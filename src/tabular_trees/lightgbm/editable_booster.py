"""Module containing editable version of LightGBM Booster objects."""

from dataclasses import dataclass, field
from typing import Union

import lightgbm as lgb

from .booster_string import BoosterString


@dataclass
class FeatureRange:
    """Feature range information from feature_infos line in Booster text."""

    min: float
    max: float

    def __str__(self) -> str:
        """Export feature range to string."""
        return f"[{self.min}:{self.max}]"

    def __repr__(self) -> str:
        """Export feature range to string."""
        return f"[{self.min}:{self.max}]"


@dataclass
class BoosterHeader:
    """Dataclass for the metadata in header section of a Booster."""

    header: str
    version: str
    num_class: int
    num_tree_per_iteration: int
    label_index: int
    max_feature_idx: int
    objective: str
    feature_names: list[str]
    feature_infos: list[FeatureRange]
    tree_sizes: list[int]

    list_delimiter: str = field(init=False, repr=False, default=" ")
    new_line: str = field(init=False, repr=False, default="\n")
    header_attributes: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Set the header_attributes value."""
        self.header_attributes = [
            "version",
            "num_class",
            "num_tree_per_iteration",
            "label_index",
            "max_feature_idx",
            "objective",
            "feature_names",
            "feature_infos",
            "tree_sizes",
        ]

    def to_string(self) -> str:
        """Concatenate header information to a single string."""
        return self.new_line.join(self.get_booster_string_rows())

    def get_booster_string_rows(self) -> list[str]:
        """Append the booster header as a list of strings."""
        header_attribute_values = [
            self.__getattribute__(attribute_name)
            for attribute_name in self.header_attributes
        ]

        header_attributes_with_equals_signs = [
            f"{name}={self._concatendate_list_to_string(value)}"
            if type(value) is list
            else f"{name}={value}"
            for name, value in zip(self.header_attributes, header_attribute_values)
        ]

        return [self.header] + header_attributes_with_equals_signs + [""]

    def _concatendate_list_to_string(self, value: list) -> str:
        """Concatenate list using list_delimiter as the separator."""
        return self.list_delimiter.join([str(list_item) for list_item in value])


@dataclass
class BoosterTree:
    """Data class for individual LightGBM trees."""

    tree: int
    num_leaves: int
    num_cat: int
    split_feature: list[int]
    split_gain: list[Union[int, float]]
    threshold: list[Union[int, float]]
    decision_type: list[int]
    left_child: list[int]
    right_child: list[int]
    leaf_value: list[Union[int, float]]
    leaf_weight: list[Union[int, float]]
    leaf_count: list[int]
    internal_value: list[Union[int, float]]
    internal_weight: list[Union[int, float]]
    internal_count: list[int]
    is_linear: int
    shrinkage: Union[int, float]

    list_delimiter: str = field(init=False, repr=False, default=" ")
    new_line: str = field(init=False, repr=False, default="\n")
    tree_attributes: list[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Set the tree_attributes value."""
        self.tree_attributes = [
            "tree",
            "num_leaves",
            "num_cat",
            "split_feature",
            "split_gain",
            "threshold",
            "decision_type",
            "left_child",
            "right_child",
            "leaf_value",
            "leaf_weight",
            "leaf_count",
            "internal_value",
            "internal_weight",
            "internal_count",
            "is_linear",
            "shrinkage",
        ]

        self.tree_attributes_for_booster_string = ["Tree"] + self.tree_attributes[1:]

    def get_booster_sting(self) -> str:
        """Concatenate tree information to a single string."""
        return self.new_line.join(self.get_booster_string_rows())

    def get_booster_string_rows(self) -> list[str]:
        """Convert data to rows that could be concatenated to part of BoosterString."""
        tree_attribute_values = [
            self.__getattribute__(attribute_name)
            for attribute_name in self.tree_attributes
        ]

        tree_attributes_with_equals_signs = [
            f"{name}={self._concatendate_list_to_string(value)}"
            if type(value) is list
            else f"{name}={value}"
            for name, value in zip(
                self.tree_attributes_for_booster_string, tree_attribute_values
            )
        ]

        return tree_attributes_with_equals_signs + ["", ""]

    def _concatendate_list_to_string(self, value: list) -> str:
        """Concatenate list using list_delimiter as the separator."""
        return self.list_delimiter.join([str(list_item) for list_item in value])


@dataclass
class EditableBooster:
    """Editable LightGBM booster."""

    header: BoosterHeader
    trees: list[BoosterTree] = field(repr=False)
    bottom_rows: list[str]

    def to_booster(self) -> lgb.Booster:
        """Convert EditableBooster to Lightgbm Booster."""
        booster_string = self._to_booster_string()
        return booster_string.to_booster()

    def _to_booster_string(self) -> BoosterString:
        """Convert EditableBooster to BoosterString."""
        booster_string_rows = self.header.get_booster_string_rows()

        for tree in self.trees:
            booster_string_rows += tree.get_booster_string_rows()

        booster_string_rows += self.bottom_rows

        return BoosterString(rows=booster_string_rows)
