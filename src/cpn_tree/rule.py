from enum import StrEnum
from typing import Sequence, TypedDict


class CompOp(StrEnum):
    GT = ">"
    LT = "<"
    GTE = ">="
    LTE = "<="


class Condition(TypedDict):
    feature: str
    comp: CompOp
    threshold: float


class Rule:
    __slots__ = [
        "result",
        "conditions",
    ]

    def __init__(self, conditions: Sequence[Condition], result: str | float):
        self.conditions = conditions
        self.result = result

    def __repr__(self) -> str:
        if not self.conditions:
            return f'true: {self.result}'
        return (
            " and ".join(
                [
                    f'{cond["feature"]} {cond["comp"]} {cond["threshold"]}'
                    for cond in self.conditions
                ]
            )
            + f": {self.result}"
        )
