from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations
from pathlib import Path
import re
from typing import Any, Optional, Sequence, TypedDict, cast
import xml.dom.minidom
from cpn_tree.access_cpn import AccessCPN
from sklearn.utils.validation import check_is_fitted
from .cpn import CPN
import pandas as pd
from .rule import Rule, Condition, CompOp
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier
from scipy.special import logit
import xml.etree.ElementTree as ET
import copy


class ModelStrategy(StrEnum):
    GBDT = "GBDT"
    LABELS = "LABELS"


@dataclass
class Model:
    name: str
    strategy: ModelStrategy


@dataclass
class Tree:
    rules: list[Rule]
    learning_rate: Optional[float] = field(default=None)


@dataclass
class GBDT(Model):
    classes: list[str] = field(default_factory=list)
    class_trees: dict[str, list[Tree]] = field(default_factory=dict)
    strategy: ModelStrategy = field(default=ModelStrategy.GBDT)

    @property
    def multiclass(self):
        return len(self.classes) > 2


class CPNTree:
    __slots__ = ["models", "cpn", "features", "instance_place_ids"]

    def __init__(self):
        self.models: dict[str, dict[str, str | Model]] = {}
        self.features = []
        self.instance_place_ids: list[str] = []
        self.__build_net()

    @staticmethod
    def format_text(text: str):
        return re.sub(r"[^a-zA-Z0-9_]", "_", text.strip())

    @staticmethod
    def __scrape_tree(
        tree, feature_names: Sequence[str], id=0, path: list[Condition] = []
    ) -> list[Rule]:
        left = cast(int, tree.children_left[id])
        right = cast(int, tree.children_right[id])
        threshold = cast(float, tree.threshold[id])
        value = cast(Any, tree.value[id, 0, 0])

        if left == right:
            return [Rule(path, value)]

        feature_i = cast(int, tree.feature[id])
        feature = CPNTree.format_text(feature_names[feature_i])

        left_path = path + [
            {"feature": feature, "comp": CompOp.LTE, "threshold": threshold}
        ]
        left_conditions = CPNTree.__scrape_tree(tree, feature_names, left, left_path)

        right_path = path + [
            {"feature": feature, "comp": CompOp.GT, "threshold": threshold}
        ]
        right_conditions = CPNTree.__scrape_tree(tree, feature_names, right, right_path)

        return left_conditions + right_conditions

    def __duplicate(self):
        return copy.deepcopy(self)

    def get_net(self):
        return self.cpn

    def predict(
        self, X: pd.DataFrame, write_cpn: Optional[str | Path] = None
    ) -> dict[str, pd.Series]:
        if not len(self.models):
            raise ValueError("You must load at least one model")
        if len(
            set(self.features) - set(self.format_text(str(col)) for col in X.columns)
        ):
            raise KeyError("X must contain every feature seen during scan and no more")

        instances = "val instances ="
        for i, (idx, row) in enumerate(X.iterrows()):
            instances += f"1`{{idx = {idx}"
            for name, value in row.items():
                instances += (
                    f", {self.format_text(str(name))} = {str(value).replace('-', '~')}"
                )
            instances += "}" + ("++\n" if i < len(X) - 1 else ";")

        new = self.__duplicate()
        new.cpn.new_constant(ml=instances)
        for instance_place_id in self.instance_place_ids:
            instances_place = new.cpn.find(instance_place_id)
            if instances_place:
                new.cpn.insert_initmark(instances_place, "instances", posattr=(50, 20))

        if write_cpn:
            new.write_cpn(write_cpn)

        access = AccessCPN()
        outputs = access.run(new.cpn)

        results = {}
        for name, model in self.models.items():
            results[name] = pd.Series(
                {
                    int(cast(dict, d)["idx"]): cast(dict, d)["label"]
                    for d in outputs[cast(str, model["out_id"])]
                }
            )

        return results

    def __rule_to_guard(self, rule: Rule):
        return (
            "["
            + ",\n".join(
                f"#{cond['feature']} instance {cond['comp']} {str(cond['threshold']).replace('-', '~')}"
                for cond in rule.conditions
            )
            + "]"
        )

    def __build_net(self):
        self.cpn = CPN()
        self.cpn.new_color(name="LABELLED", type_={"idx": "INT", "label": "STRING"})
        self.cpn.new_variable(name="idx", type_="INT")

    def add_from_GradientBoostingClassifier(
        self, name: str, gbdt: GradientBoostingClassifier
    ):
        check_is_fitted(gbdt)

        new = self.__duplicate()

        if new.format_text(name) in new.models:
            raise ValueError("There already is a model with this name")

        params = gbdt.get_params()
        if params["loss"] != "log_loss":
            raise ValueError("Loss function must be log_loss")

        features = [new.format_text(feature) for feature in gbdt.feature_names_in_]

        if not len(new.features):
            new.features = features
            new.cpn.new_color(
                name="INSTANCE",
                type_={
                    "idx": "INT",
                    **{feature: "REAL" for feature in new.features},
                },
            )
            new.cpn.new_variable(name="instance", type_="INSTANCE")
        elif len(set(features) - set(new.features)):
            raise KeyError("Feature names must be equal between all models")

        init = gbdt.init_
        model = GBDT(
            name=new.format_text(name),
            classes=[new.format_text(class_) for class_ in gbdt.classes_],
        )

        for i, class_ in enumerate(
            gbdt.classes_ if model.multiclass else [gbdt.classes_[1]]
        ):
            class_ = new.format_text(class_)
            model.class_trees[class_] = []
            if init == "zero":
                model.class_trees[class_].append(Tree(rules=[Rule([], 0)]))
            else:
                init = cast(DummyClassifier, init)
                init_params = init.get_params()
                if (
                    init_params["strategy"] in ["stratified", "uniform"]
                    and init_params["random_state"] is None
                ):
                    raise ValueError("DummyClassifier must be deterministic")
                model.class_trees[class_].append(
                    Tree(
                        rules=[
                            Rule([], gbdt._raw_predict_init([[0] * gbdt.n_features_in_])[0][i if model.multiclass else 0])  # type: ignore
                        ],
                    )
                )

            for estimator in gbdt.estimators_[:, i if model.multiclass else 0]:
                model.class_trees[class_].append(
                    Tree(
                        rules=new.__scrape_tree(estimator.tree_, features),
                        learning_rate=params["learning_rate"],
                    )
                )

        new.__add_model(model)
        return new

    def __add_model(self, model: Model):
        self.cpn.new_page(name=model.name)
        self.instance_place_ids.append(
            str(
                self.cpn.new_place(
                    page=model.name,
                    posattr=(0, 0),
                    name="Instances",
                    type_="INSTANCE",
                    size=(100, 40),
                ).get("id")
            )
        )

        out_id = ""
        match model.strategy:
            case ModelStrategy.GBDT:
                out_id = self.__add_gbdt(cast(GBDT, model))

        self.models[model.name] = {"model": model, "out_id": out_id}

    def __add_gbdt(self, model: GBDT) -> str:
        self.cpn.new_declaration_block(
            block=model.name, text=f"{model.name} Declarations"
        )
        self.cpn.new_color(
            name=f"{model.name.upper()}_RESULT",
            type_={"idx": "INT", "result": "REAL"},
            block=model.name,
        )
        self.cpn.new_trans(
            page=model.name, posattr=(200, 0), name="Load instance", size=(100, 40)
        )
        self.cpn.new_arc(
            page=model.name,
            orientation="PTOT",
            place="Instances",
            trans="Load instance",
            annot="instance",
            annot_pos=(100, 10),
        )
        for i, [class_, trees] in enumerate(model.class_trees.items()):
            self.cpn.new_variable(
                name=f"{model.name}_{class_}_result", type_="REAL", block=model.name
            )
            self.cpn.new_place(
                page=model.name,
                posattr=(400, -100 * i),
                name=f"{class_} Input",
                size=(100, 40),
                type_="INSTANCE",
            )
            self.cpn.new_arc(
                page=model.name,
                orientation="TTOP",
                trans="Load instance",
                place=f"{class_} Input",
                annot="instance",
                bend_points=[(200, -100 * i)] if i > 0 else [],
                annot_pos=(300, -100 * i + 10),
            )
            self.cpn.new_trans(
                page=model.name,
                posattr=(600, -100 * i),
                name=f"{model.name}_{class_}",
                size=(100, 40),
            )
            self.cpn.new_arc(
                page=model.name,
                orientation="PTOT",
                place=f"{class_} Input",
                trans=f"{model.name}_{class_}",
            )
            self.cpn.new_place(
                page=model.name,
                posattr=(800, -100 * i),
                name=f"{class_} Output",
                size=(100, 40),
                type_=f"{model.name.upper()}_RESULT",
            )
            self.cpn.new_arc(
                page=model.name,
                orientation="TTOP",
                trans=f"{model.name}_{class_}",
                place=f"{class_} Output",
            )
            self.cpn.new_page(name=f"{model.name}_{class_}")

            self.cpn.new_place(
                page=f"{model.name}_{class_}",
                posattr=(0, 0),
                name=f"{class_} Input",
                type_="INSTANCE",
                port="In",
            )
            self.cpn.new_trans(
                page=f"{model.name}_{class_}",
                posattr=(0, -100),
                name="Load instance",
                size=(100, 40),
            )
            self.cpn.new_arc(
                page=f"{model.name}_{class_}",
                orientation="PTOT",
                place=f"{class_} Input",
                trans=f"Load instance",
                annot="instance",
                annot_pos=(50, -50)
            )
            max_rules = max(len(tree.rules) for tree in trees)
            self.cpn.new_trans(
                page=f"{model.name}_{class_}",
                name="Sum",
                posattr=(0, -450 - 100 * max_rules),
            )
            for i, tree in enumerate(trees):
                self.cpn.new_variable(
                    name=f"{model.name}_{class_}_t{i}_result",
                    type_="REAL",
                    block=model.name,
                )
                self.cpn.new_place(
                    page=f"{model.name}_{class_}",
                    posattr=(600 * i, -250),
                    name=f"T{i} Input",
                    type_="INSTANCE",
                    size=(100, 40),
                )
                self.cpn.new_arc(
                    page=f"{model.name}_{class_}",
                    orientation="TTOP",
                    trans="Load instance",
                    place=f"T{i} Input",
                    annot="instance",
                    bend_points=[(0, -150), (600 * i, -150)],
                    annot_pos=(600 * i + 50, -200),
                )
                self.cpn.new_place(
                    page=f"{model.name}_{class_}",
                    posattr=(600 * i + 500, -350 - 100 * max_rules),
                    name=f"T{i} Output",
                    type_=f"{model.name.upper()}_RESULT",
                    size=(100, 40),
                )
                self.cpn.new_arc(
                    page=f"{model.name}_{class_}",
                    orientation="PTOT",
                    place=f"T{i} Output",
                    trans="Sum",
                    annot=f"{{idx = idx,\nresult = {model.name}_{class_}_t{i}_result}}",
                    bend_points=[
                        (500, -450 - 100 * max_rules),
                        (600 * i + 500, -450 - 100 * max_rules),
                    ],
                    annot_pos=(
                        (600 * i + 200, -430 - 100 * max_rules)
                    ),
                )
                for j, rule in enumerate(tree.rules):
                    self.cpn.new_trans(
                        page=f"{model.name}_{class_}",
                        name=f"T{i} Rule {j + 1}",
                        posattr=(600 * i + 200, -350 - 100 * j),
                        cond=self.__rule_to_guard(rule),
                    )
                    self.cpn.new_arc(
                        page=f"{model.name}_{class_}",
                        orientation="PTOT",
                        place=f"T{i} Input",
                        trans=f"T{i} Rule {j + 1}",
                        annot="instance",
                        bend_points=[(600 * i, -350 - 100 * j)],
                        annot_pos=(600 * i + 100, -340 - 100 * j),
                    )
                    result = str(rule.result).replace('-', '~')
                    if tree.learning_rate is not None:
                        result += f' *\n{str(tree.learning_rate).replace('-', '~')}'
                    self.cpn.new_arc(
                        page=f"{model.name}_{class_}",
                        orientation="TTOP",
                        trans=f"T{i} Rule {j + 1}",
                        place=f"T{i} Output",
                        annot=f"{{idx = #idx instance,\nresult = {result}}}",
                        bend_points=[(600 * i + 500, -350 - 100 * j)],
                        annot_pos=(600 * i + 500, -300 - 100 * j),
                    )
            self.cpn.new_place(
                page=f"{model.name}_{class_}",
                name=f"{class_} Output",
                type_=f"{model.name.upper()}_RESULT",
                posattr=(0, -550 - 100 * max_rules),
                size=(100, 40),
                port="Out",
            )
            self.cpn.new_arc(
                page=f"{model.name}_{class_}",
                orientation="TTOP",
                trans="Sum",
                place=f"{class_} Output",
                annot=f"{{idx = idx,\nresult = {'+\n'.join(f'{model.name}_{class_}_t{i}_result' for i in range(len(trees)))}}}",
                annot_pos=(250, -550 - 100 * max_rules),
            )
            self.cpn.instantiate_page(
                page=model.name,
                trans=f"{model.name}_{class_}",
                subpage=f"{model.name}_{class_}",
                ports=[
                    {
                        "main_page_place": f"{class_} Input",
                        "subpage_place": f"{class_} Input",
                    },
                    {
                        "main_page_place": f"{class_} Output",
                        "subpage_place": f"{class_} Output",
                    },
                ],
            )
        out_id = str(
            self.cpn.new_place(
                page=model.name,
                name=f"{model.name} Output",
                type_="LABELLED",
                posattr=(1800, 0),
                size=(100, 40),
            ).get("id")
        )
        if len(model.classes) > 2:
            for i, class_ in enumerate(model.classes):
                others = [
                    f"{model.name}_{class_}_result {'>=' if j > i else '>'} {model.name}_{other}_result"
                    for j, other in enumerate(model.classes)
                    if i != j
                ]
                condition = " andalso\n".join(others)
                self.cpn.new_trans(
                    page=model.name,
                    posattr=(1400, -100 * i),
                    name=f"Classify {class_}",
                    cond=f"[{condition}]",
                    size=(100, 40),
                )
                for j, other in enumerate(model.classes):
                    self.cpn.new_arc(
                        page=model.name,
                        orientation="PTOT",
                        place=f"{other} Output",
                        trans=f"Classify {class_}",
                        annot=f"{{idx = idx, result = {model.name}_{other}_result}}",
                        annot_pos=(1100, -100 * i + 10 * j + 10),
                    )
                self.cpn.new_arc(
                    page=model.name,
                    orientation="TTOP",
                    trans=f"Classify {class_}",
                    place=f"{model.name} Output",
                    annot=f'{{idx = idx, label = "{class_}"}}',
                    bend_points=[(1800, -100 * i)] if i > 0 else [],
                    annot_pos=(1600, -100 * i + 10),
                )
        else:
            neg_class, pos_class = list(model.classes)
            self.cpn.new_trans(
                page=model.name, posattr=(1400, 0), name=f"Classify", size=(100, 40)
            )
            self.cpn.new_arc(
                page=model.name,
                orientation="PTOT",
                place=f"{pos_class} Output",
                trans=f"Classify",
                annot=f"{{idx = idx, result = {model.name}_{pos_class}_result}}",
                annot_pos=(1100, 10),
            )
            self.cpn.new_arc(
                page=model.name,
                orientation="TTOP",
                trans=f"Classify",
                place=f"{model.name} Output",
                annot=f'{{idx = idx, label = if {model.name}_{pos_class}_result >= 0.0 then "{pos_class}" else "{neg_class}"}}',
                annot_pos=(1600, 10),
            )
        return out_id

    def write_cpn(self, path: str | Path):
        with open(path, "w") as f:
            xml.dom.minidom.parseString(str(self.cpn)).writexml(
                f, addindent="  ", newl="\n"
            )
        return self
