from dataclasses import dataclass, field
from enum import StrEnum
from itertools import combinations
from pathlib import Path
from typing import Any, Optional, Sequence, TypedDict, cast
import xml.dom.minidom
from cpn_tree.access_cpn import AccessCPN
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
    GROUND_TRUTH = "GROUND_TRUTH"


@dataclass
class Model:
    name: str
    strategy: ModelStrategy


@dataclass
class Tree:
    rules: list[Rule]
    learning_rate: Optional[float]


@dataclass
class TreeBasedModel(Model):
    trees: list[Tree] = field(default_factory=list)

    def add_tree(self, tree: Tree):
        self.trees.append(tree)


@dataclass
class LabelledInstance:
    idx: int
    label: bool


@dataclass
class GroundTruth(Model):
    instances: list[LabelledInstance] = field(default_factory=list)


class InternalModelsComparisons(TypedDict):
    equal: str
    different: str


class InternalModels(TypedDict):
    model: Model
    comparisons: dict[str, InternalModelsComparisons]


class CPNTree:
    __slots__ = ["models", "features", "cpn", "instances_place_id", 'comparator_place_id']

    def __init__(self, features: list[str]):
        self.models: InternalModels = cast(InternalModels, {})
        self.features: list[str] = [
            CPNTree.__format_text(feature) for feature in features
        ]
        self.__build_net()

    @staticmethod
    def __format_text(feature: str):
        return "_".join(feature.strip().lower().split())

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
        feature = CPNTree.__format_text(feature_names[feature_i])

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

    def predict(self, X: pd.DataFrame) :
        if not len(self.models):
            raise ValueError("You must load at least one model")

        if len(
            set(self.features)
            - set(CPNTree.__format_text(str(col)) for col in X.columns)
        ):
            raise KeyError("X must contain every feature seen during scan and no more")

        instances = "val instances ="
        for i, (idx, row) in enumerate(X.iterrows()):
            instances += f"1`{{idx = {idx}"
            for name, value in row.items():
                instances += f", {CPNTree.__format_text(str(name))} = {str(value).replace('-', '~')}"
            instances += "}" + ("++\n" if i < len(X) - 1 else ";")

        indices = "val indices ="
        for i, idx in enumerate(X.index):
            indices += f"1`{idx}"
            indices += "++\n" if i < len(X) - 1 else ";"

        new = self.__duplicate()
        new.cpn.new_constant(ml=instances)
        instances_place = new.cpn.find(self.instances_place_id)
        if instances_place:
            new.cpn.insert_initmark(instances_place, "instances", posattr=(50, 20))
        new.cpn.new_constant(ml=indices)
        comparator_place = new.cpn.find(self.comparator_place_id)
        if comparator_place:
            new.cpn.insert_initmark(comparator_place, "indices", posattr=(50, 20))

        access = AccessCPN()
        outputs = access.run(new.cpn)

        results = {}
        for name, model in self.models.items():
            results.setdefault(name, {})
            for other, comparisons in cast(dict, model)['comparisons'].items():
                results[name][other] = {
                    "equal": cast(dict, outputs[comparisons["equal"]])["quantity"],
                    "different": cast(dict, outputs[comparisons["different"]])["quantity"],
                }

        return results

    def __rule_to_guard(self, rule: Rule):
        return (
            "["
            + " andalso\n ".join(
                f"#{cond['feature']} instance {cond['comp']} {cond['threshold']}"
                for cond in rule.conditions
            )
            + "]"
        )

    def __build_net(self):
        self.cpn = CPN()
        self.cpn.new_color(name="LABELLED", _type={"idx": "INT", "label": "BOOL"})
        self.cpn.new_color(
            name="INSTANCE",
            _type={
                "idx": "INT",
                **{feature: "REAL" for feature in self.features},
            },
        )
        self.cpn.new_variable(name="instance", _type="INSTANCE")
        self.cpn.new_variable(name="idx", _type="INT")
        self.cpn.new_variable(name="labelled", _type="LABELLED")
        self.cpn.new_variable(name="labelled_other", _type="LABELLED")
        self.cpn.new_page(name="Main")
        self.instances_place_id = str(self.cpn.new_place(
            page="Main",
            posattr=(0, 0),
            name="Dataset",
            _type="INSTANCE",
        ).get("id"))
        self.cpn.new_trans(
            page="Main", posattr=(200, 0), name="Load instance", size=(100, 40)
        )
        self.cpn.new_arc(
            page="Main",
            orientation="PTOT",
            place="Dataset",
            trans="Load instance",
            annot="instance",
        )
        self.cpn.new_page(name='Comparator')
        self.cpn.new_place(page='Comparator', posattr=(0, 0), name='Model 1 Input', _type='LABELLED', size=(100,40), port='I/O')
        self.cpn.new_place(page='Comparator', posattr=(0, -200), name='Model 2 Input', _type='LABELLED', size=(100,40), port='I/O')
        self.comparator_place_id = str(self.cpn.new_place(page='Comparator', posattr=(400, -100), name='Indices', _type='INT').get('id'))
        self.cpn.new_trans(page='Comparator', posattr=(400, 0), name='Model 1 eq Model 2', size=(150,40), cond='[#idx labelled = idx andalso #idx labelled_other = idx andalso #label labelled = #label labelled_other]')
        self.cpn.new_trans(page='Comparator', posattr=(400, -200), name='Model 1 neq Model 2', size=(150,40), cond='[#idx labelled = idx andalso #idx labelled_other = idx andalso #label labelled <> #label labelled_other]', cond_pos=(400, -250))
        self.cpn.new_arc(page='Comparator', orientation='BOTHDIR', place='Model 1 Input', trans='Model 1 eq Model 2', annot='labelled')
        self.cpn.new_arc(page='Comparator', orientation='BOTHDIR', place='Model 2 Input', trans='Model 1 eq Model 2', annot='labelled_other')
        self.cpn.new_arc(page='Comparator', orientation='BOTHDIR', place='Model 1 Input', trans='Model 1 neq Model 2', annot='labelled', annot_pos=(265, -165))
        self.cpn.new_arc(page='Comparator', orientation='BOTHDIR', place='Model 2 Input', trans='Model 1 neq Model 2', annot='labelled_other', annot_pos=(265, -35))
        self.cpn.new_arc(page='Comparator', orientation='PTOT', place='Indices', trans='Model 1 eq Model 2', annot='idx', annot_pos=(400, -50))
        self.cpn.new_arc(page='Comparator', orientation='PTOT', place='Indices', trans='Model 1 neq Model 2', annot='idx', annot_pos=(400, -150))
        self.cpn.new_place(page='Comparator', posattr=(600, 0), name='Equal', port='Out')
        self.cpn.new_place(page='Comparator', posattr=(600, -200), name='Different', port='Out')
        self.cpn.new_arc(page='Comparator', orientation='TTOP', trans='Model 1 eq Model 2', place='Equal', annot='1`()')
        self.cpn.new_arc(page='Comparator', orientation='TTOP', trans='Model 1 neq Model 2', place='Different', annot='1`()')

    def compare_models(self, X: pd.DataFrame) -> dict[tuple[str, str], pd.Series]:
        outputs = self.predict(X)

        comparisons = {}
        for (name1, out1), (name2, out2) in combinations(outputs.items(), 2):
            comparisons[(name1, name2)] = out1 == out2.reindex(out1.index)

        return comparisons

    def add_from_GroundTruth(self, name: str, y_true: pd.Series, label_true: Any):
        new = self.__duplicate()

        if CPNTree.__format_text(name) in new.models:
            raise ValueError("There already is a model with this name")

        model = GroundTruth(
            CPNTree.__format_text(name),
            ModelStrategy.GROUND_TRUTH,
            [
                LabelledInstance(idx=cast(int, idx), label=label == label_true)
                for idx, label in y_true.items()
            ],
        )

        new.__add_model(model)

        return new

    def add_from_GradientBoostingClassifier(
        self, name: str, gbdt: GradientBoostingClassifier
    ):
        new = self.__duplicate()

        if CPNTree.__format_text(name) in self.models:
            raise ValueError("There already is a model with this name")

        model = TreeBasedModel(CPNTree.__format_text(name), ModelStrategy.GBDT)

        params = gbdt.get_params()
        if params["loss"] != "log_loss":
            raise ValueError("Loss function must be log_loss")

        if len(
            set(CPNTree.__format_text(feature) for feature in gbdt.feature_names_in_)
            - set(new.features)
        ):
            raise KeyError("Feature names must be equal between all models")

        init = gbdt.init_

        if init == "zero":
            model.add_tree(Tree(rules=[Rule([], 0)], learning_rate=1.0))
        else:
            init = cast(DummyClassifier, init)
            init_params = init.get_params()
            if (
                init_params["strategy"] in ["stratified", "uniform"]
                and init_params["random_state"] is None
            ):
                raise ValueError("DummyClassifier must be deterministic")
            model.add_tree(
                Tree(
                    rules=[
                        Rule([], logit(init.predict_proba(pd.DataFrame([1]))[0][1]))
                    ],
                    learning_rate=1.0,
                )
            )

        for estimator in gbdt.estimators_:
            model.add_tree(
                Tree(
                    rules=CPNTree.__scrape_tree(estimator[0].tree_, new.features),
                    learning_rate=params["learning_rate"],
                )
            )

        new.__add_model(model)

        return new

    def __add_model(self, model: Model):
        m = len(self.models)
        origin = (400, -m * 200)

        self.cpn.new_place(
            page="Main",
            posattr=origin,
            name=f"{model.name} Input",
            _type="INSTANCE",
            size=(100, 40),
        )
        self.cpn.new_arc(
            page="Main",
            orientation="TTOP",
            trans="Load instance",
            place=f"{model.name} Input",
            annot="instance",
            bend_points=[(origin[0] - 200, origin[1])] if m != 0 else [],
        )
        self.cpn.new_trans(
            page="Main",
            posattr=(origin[0] + 200, origin[1]),
            name=model.name,
            size=(100, 40),
        )
        self.cpn.new_arc(
            page="Main",
            orientation="PTOT",
            place=f"{model.name} Input",
            trans=model.name,
        )
        self.cpn.new_place(
            page="Main",
            name=f"{model.name} Output",
            _type="LABELLED",
            posattr=(origin[0] + 400, origin[1]),
            size=(100, 40),
        )
        self.cpn.new_arc(
            page="Main",
            orientation="TTOP",
            trans=model.name,
            place=f"{model.name} Output",
        )
        self.cpn.new_page(name=model.name)
        self.cpn.new_place(
            page=model.name,
            posattr=(0, 0),
            name=f"{model.name} Input",
            _type="INSTANCE",
            size=(100, 40),
            port="In",
        )

        match model.strategy:
            case ModelStrategy.GBDT:
                self.__add_gbdt(cast(TreeBasedModel, model))
            case ModelStrategy.GROUND_TRUTH:
                self.__add_ground_truth(cast(GroundTruth, model))

        self.cpn.instantiate_page(
            page="Main",
            subpage=model.name,
            trans=model.name,
            ports=[
                {
                    "main_page_place": f"{model.name} Input",
                    "subpage_place": f"{model.name} Input",
                },
                {
                    "main_page_place": f"{model.name} Output",
                    "subpage_place": f"{model.name} Output",
                },
            ],
        )

        for i, other in enumerate(self.models):
            coord1 = (origin[0] + 600 + 250 * i, -200 * len(self.models) + 50)
            coord2 = (origin[0] + 600 - 250 * i + 250 * (len(self.models) - 1), -200 * i + 50)
            self.cpn.new_trans(page='Main', posattr=coord2, name=f'{other} vs {model.name}', size=(100, 40))
            self.cpn.new_arc(page="Main", orientation="BOTHDIR", place=f"{model.name} Output", trans=f"{other} vs {model.name}", bend_points=[(coord2[0] - 100, coord1[1] - 50), (coord2[0] - 100, coord2[1])])
            self.cpn.new_arc(page="Main", orientation="BOTHDIR", place=f"{other} Output", trans=f"{other} vs {model.name}", bend_points=[(coord2[0], coord2[1] - 50)])
            eq = self.cpn.new_place(page="Main", name=f"{other} eq {model.name}", posattr=(coord2[0] + 90, coord2[1] + 100), size=(100, 40))
            neq = self.cpn.new_place(page="Main", name=f"{other} neq {model.name}", posattr=(coord2[0] + 90, coord2[1] + 50), size=(100, 40))
            self.cpn.new_arc(page="Main", orientation="TTOP", trans=f"{other} vs {model.name}", place=f"{other} eq {model.name}", bend_points=[(coord2[0], coord2[1] + 100)])
            self.cpn.new_arc(page="Main", orientation="TTOP", trans=f"{other} vs {model.name}", place=f"{other} neq {model.name}", bend_points=[(coord2[0], coord2[1] + 50)])
            self.cpn.instantiate_page(page='Main', trans=f'{other} vs {model.name}', subpage='Comparator', ports=[{"main_page_place": f'{other} Output', "subpage_place": 'Model 1 Input'}, {"main_page_place": f'{model.name} Output', 'subpage_place': 'Model 2 Input'}, {"main_page_place": f'{other} eq {model.name}', "subpage_place": 'Equal'}, {'main_page_place': f'{other} neq {model.name}', 'subpage_place': 'Different'}])
            self.models[other]['comparisons'] = {model.name: {'equals': eq, 'different': neq}}

        self.models[model.name] = {"model": model, 'comparisons': {}}

    def __add_ground_truth(self, model: GroundTruth):
        self.cpn.new_declaration_block(
            block=model.name, text=f"{model.name} Declarations"
        )

        instances = (
            f"val {model.name}_instances = "
            + "++\n".join(
                [
                    f"1`{{idx = {instance.idx}, label={str(instance.label).lower()}}}"
                    for instance in model.instances
                ]
            )
            + ";"
        )
        self.cpn.new_constant(ml=instances, block=model.name)
        self.cpn.new_variable(
            name=f"{model.name}_labelled_instance", _type="LABELLED", block=model.name
        )

        self.cpn.new_place(
            page=model.name,
            name="Instances",
            _type="LABELLED",
            initmark=f"{model.name}_instances",
            posattr=(0, -100),
            size=(100, 40),
        )
        self.cpn.new_trans(
            page=model.name,
            name="Match",
            posattr=(300, 0),
            cond=f"[#idx instance = #idx {model.name}_labelled_instance]",
        )
        self.cpn.new_arc(
            page=model.name,
            orientation="PTOT",
            place=f"{model.name} Input",
            trans="Match",
            annot="instance",
        )
        self.cpn.new_arc(
            page=model.name,
            orientation="PTOT",
            place="Instances",
            trans="Match",
            annot=f"{model.name}_labelled_instance",
            bend_points=[(300, -100)],
        )
        self.cpn.new_place(
            page=model.name,
            name=f"{model.name} Output",
            _type="LABELLED",
            posattr=(600, 0),
            size=(100, 40),
            port="Out",
        )
        self.cpn.new_arc(
            page=model.name,
            orientation="TTOP",
            trans="Match",
            place=f"{model.name} Output",
            annot=f"{model.name}_labelled_instance",
        )

    def __add_gbdt(self, model: TreeBasedModel):
        self.cpn.new_declaration_block(
            block=model.name, text=f"{model.name} Declarations"
        )
        self.cpn.new_color(
            name=f"{model.name.upper()}_RESULT",
            _type={"idx": "INT", "result": "REAL"},
            block=model.name,
        )
        self.cpn.new_variable(
            name=f"{model.name}_result",
            _type=f"{model.name.upper()}_RESULT",
            block=model.name,
        )
        self.cpn.new_trans(
            page=model.name, posattr=(200, 0), name="Load instance", size=(100, 40)
        )
        self.cpn.new_arc(
            page=model.name,
            orientation="PTOT",
            trans=f"Load instance",
            place=f"{model.name} Input",
            annot="instance",
        )
        self.cpn.new_trans(page=model.name, name="Sum", posattr=(1200, 0))
        for i, tree in enumerate(model.trees):
            self.cpn.new_variable(
                name=f"{model.name}_t{i}_result", _type="REAL", block=model.name
            )
            self.cpn.new_place(
                page=model.name,
                posattr=(400, -i * 50),
                name=f"T{i} Input",
                _type="INSTANCE",
                size=(100, 40),
            )
            self.cpn.new_arc(
                page=model.name,
                orientation="TTOP",
                trans="Load instance",
                place=f"T{i} Input",
                annot="instance",
                bend_points=[(200, -i * 50)] if i != 0 else [],
            )
            self.cpn.new_trans(
                page=model.name,
                posattr=(600, -i * 50),
                name=f"{model.name} Tree {i}",
                size=(100, 40),
            )
            self.cpn.new_arc(
                page=model.name,
                orientation="PTOT",
                place=f"T{i} Input",
                trans=f"{model.name} Tree {i}",
            )
            self.cpn.new_place(
                page=model.name,
                name=f"T{i} Output",
                posattr=(800, -i * 50),
                _type=f"{model.name.upper()}_RESULT",
                size=(100, 40),
            )
            self.cpn.new_arc(
                page=model.name,
                orientation="TTOP",
                trans=f"{model.name} Tree {i}",
                place=f"T{i} Output",
            )
            self.cpn.new_page(name=f"{model.name} Tree {i}")
            self.cpn.new_place(
                page=f"{model.name} Tree {i}",
                name=f"T{i} Input",
                posattr=(0, 0),
                _type="INSTANCE",
                size=(100, 40),
                port="In",
            )
            self.cpn.new_place(
                page=f"{model.name} Tree {i}",
                name=f"T{i} Output",
                posattr=(800, 0),
                _type=f"{model.name.upper()}_RESULT",
                size=(100, 40),
                port="Out",
            )
            self.cpn.instantiate_page(
                page=model.name,
                subpage=f"{model.name} Tree {i}",
                trans=f"{model.name} Tree {i}",
                ports=[
                    {
                        "main_page_place": f"T{i} Input",
                        "subpage_place": f"T{i} Input",
                    },
                    {
                        "main_page_place": f"T{i} Output",
                        "subpage_place": f"T{i} Output",
                    },
                ],
            )

            for j, rule in enumerate(tree.rules):
                self.cpn.new_trans(
                    page=f"{model.name} Tree {i}",
                    name=f"Rule {j}",
                    posattr=(300, -j * 100),
                    cond=self.__rule_to_guard(rule),
                )
                self.cpn.new_arc(
                    page=f"{model.name} Tree {i}",
                    orientation="PTOT",
                    place=f"T{i} Input",
                    trans=f"Rule {j}",
                    annot="instance",
                    bend_points=[(0, -j * 100)] if j != 0 else [],
                )
                self.cpn.new_arc(
                    page=f"{model.name} Tree {i}",
                    orientation="TTOP",
                    trans=f"Rule {j}",
                    place=f"T{i} Output",
                    annot=f"{{idx = #idx instance, result = {str(rule.result).replace('-', '~')} * {str(tree.learning_rate).replace('-', '~')}}}",
                    bend_points=[(800, -j * 100)] if j != 0 else [],
                )
            self.cpn.new_arc(
                page=model.name,
                orientation="PTOT",
                place=f"T{i} Output",
                trans="Sum",
                annot=f"{{idx = idx, result = {model.name}_t{i}_result}}",
                bend_points=[(1200, -i * 50)] if i != 0 else [],
            )
        self.cpn.new_place(
            page=model.name,
            name="Result",
            posattr=(1600, 0),
            _type=f"{model.name.upper()}_RESULT",
        )
        self.cpn.new_arc(
            page=model.name,
            orientation="TTOP",
            trans="Sum",
            place="Result",
            annot=f"{{idx = idx, result = {'+\n'.join(f'{model.name}_t{i}_result' for i in range(len(model.trees)))}}}",
        )
        self.cpn.new_trans(page=model.name, name="Sigmoid", posattr=(1800, 0))
        self.cpn.new_arc(
            page=model.name,
            orientation="PTOT",
            place="Result",
            trans="Sigmoid",
            annot=f"{model.name}_result",
        )
        self.cpn.new_place(
            page=model.name,
            name="Probability",
            _type=f"{model.name.upper()}_RESULT",
            posattr=(2400, 0),
            size=(100, 40),
        )
        self.cpn.new_arc(
            page=model.name,
            orientation="TTOP",
            trans="Sigmoid",
            place="Probability",
            annot=f"{{idx = #idx {model.name}_result,\nresult = 1.0 / (1.0 + Math.exp(~1.0 * #result {model.name}_result))}}",
        )
        self.cpn.new_trans(page=model.name, name="Classify", posattr=(2600, 0))
        self.cpn.new_arc(
            page=model.name,
            orientation="PTOT",
            place="Probability",
            trans="Classify",
            annot=f"{model.name}_result",
        )
        self.cpn.new_place(
            page=model.name,
            name=f"{model.name} Output",
            _type="LABELLED",
            posattr=(3000, 0),
            size=(100, 40),
            port="Out",
        )
        self.cpn.new_arc(
            page=model.name,
            orientation="TTOP",
            trans="Classify",
            place=f"{model.name} Output",
            annot=f"{{idx = #idx {model.name}_result, label = (#result {model.name}_result) >= 0.5}}",
        )

    def write_cpn(self, path: str | Path):
        with open(path, "w") as f:
            xml.dom.minidom.parseString(str(self.cpn)).writexml(
                f, addindent="  ", newl="\n"
            )
