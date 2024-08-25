"""Module containing editable version of LightGBM Booster objects."""

from dataclasses import dataclass, field
from typing import Union

import lightgbm as lgb

from .booster_string import BoosterString
from .helpers import (
    FloatFixedString,
    convert_string_to_int_or_float,
    remove_surrounding_brackets,
)


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
        """Concatenate tree information to a single string.

        Returns
        -------
        model : str
            Booster as a string.

        """
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


class BoosterStringConverter:
    """Logic for converting BoosterString objects to EditableBooster objects."""

    def convert(self, booster_string: BoosterString) -> "EditableBooster":
        """Convert a BoosterString to EditableBooster object."""
        header = self._export_header(booster_string)
        trees = self._export_trees(booster_string)
        bottom_rows = booster_string.extract_bottom_rows()

        return EditableBooster(header=header, trees=trees, bottom_rows=bottom_rows)

    def _export_header(self, booster_string: BoosterString) -> BoosterHeader:
        """Export the header information from BoosterString as BoosterHeader object."""
        header_rows = booster_string.extract_header_rows()
        return self._rows_to_header(header_rows)

    def _rows_to_header(self, rows: list[str]) -> BoosterHeader:
        """Convert string booster rows to BoosterHeader object."""
        return BoosterHeader(
            header=rows[0],
            version=self._extract_string(rows[1]),
            num_class=self._extract_int(rows[2]),
            num_tree_per_iteration=self._extract_int(rows[3]),
            label_index=self._extract_int(rows[4]),
            max_feature_idx=self._extract_int(rows[5]),
            objective=self._extract_string(rows[6]),
            feature_names=self._extract_list_of_strings(rows[7]),
            feature_infos=self._extract_list_of_feature_ranges(rows[8]),
            tree_sizes=self._extract_list_of_ints(rows[9]),
        )

    def _export_trees(self, booster_string: BoosterString) -> list[BoosterTree]:
        """Extract trees from booster string to list of BoosterTree objects."""
        booster_trees = []

        for tree_number in booster_string.tree_rows:
            tree_rows = booster_string.extract_tree_rows(tree_number)
            booster_trees.append(self._rows_to_tree(tree_rows))

        return booster_trees

    def _rows_to_tree(self, rows: list[str]) -> BoosterTree:
        """Convert a list of strings to BoosterTree object."""
        return BoosterTree(
            tree=self._extract_int(rows[0]),
            num_leaves=self._extract_int(rows[1]),
            num_cat=self._extract_int(rows[2]),
            split_feature=self._extract_list_of_ints(rows[3]),
            split_gain=self._extract_list_of_ints_or_floats(rows[4]),
            threshold=self._extract_list_of_ints_or_floats(rows[5]),
            decision_type=self._extract_list_of_ints(rows[6]),
            left_child=self._extract_list_of_ints(rows[7]),
            right_child=self._extract_list_of_ints(rows[8]),
            leaf_value=self._extract_list_of_ints_or_floats(rows[9]),
            leaf_weight=self._extract_list_of_ints_or_floats(rows[10]),
            leaf_count=self._extract_list_of_ints(rows[11]),
            internal_value=self._extract_list_of_ints_or_floats(rows[12]),
            internal_weight=self._extract_list_of_ints_or_floats(rows[13]),
            internal_count=self._extract_list_of_ints(rows[14]),
            is_linear=self._extract_int(rows[15]),
            shrinkage=self._extract_int_or_float(rows[16]),
        )

    def _split_at_equals(self, line: str) -> str:
        """Extract the part of the input line after the equals line."""
        return line.split("=")[1]

    def _extract_string(self, line: str) -> str:
        """Extract line contents after equals sign and return as string."""
        return self._split_at_equals(line)

    def _extract_int(self, line: str) -> int:
        """Extract line contents after equals sign and return as int."""
        return int(self._split_at_equals(line))

    def _extract_int_or_float(self, line: str) -> Union[int, float]:
        """Extract line contents after equals sign and return as int or float."""
        return convert_string_to_int_or_float(self._split_at_equals(line))

    def _extract_list_of_strings(self, line: str, delimiter: str = " ") -> list[str]:
        """Extract line contents after equals sign and return as list of strings."""
        return self._split_at_equals(line).split(delimiter)

    def _extract_list_of_ints(self, line: str, delimiter: str = " ") -> list[int]:
        """Extract line contents after equals sign and return as list of ints."""
        return [int(value) for value in self._extract_list_of_strings(line, delimiter)]

    def _extract_list_of_ints_or_floats(
        self, line: str, delimiter: str = " "
    ) -> list[Union[int, float]]:
        """Extract line contents after equals sign and return as list numbers."""
        return [
            convert_string_to_int_or_float(value)
            for value in self._extract_list_of_strings(line, delimiter)
        ]

    def _extract_list_of_feature_ranges(
        self, line: str, delimiter: str = " "
    ) -> list[FeatureRange]:
        """Extract line contents after equals sign and return as FeatureRanges list."""
        return [
            self._feature_range_from_string(value)
            for value in self._extract_list_of_strings(line, delimiter)
        ]

    def _feature_range_from_string(self, string: str) -> FeatureRange:
        """Convert string of the form '[x:y]' to FeatureRanges(x, y) object."""
        string_without_brackets = remove_surrounding_brackets(string)
        string_split = string_without_brackets.split(":")

        return FeatureRange(
            min=FloatFixedString(string_split[0], string_split[0]),
            max=FloatFixedString(string_split[1], string_split[1]),
        )


@dataclass
class EditableBooster:
    """Editable LightGBM booster."""

    header: BoosterHeader
    trees: list[BoosterTree] = field(repr=False)
    bottom_rows: list[str]

    @classmethod
    def from_booster(cls, booster: lgb.Booster) -> "EditableBooster":
        """Create an EditableBooster from a LightGBM Booster.

        Parameters
        ----------
        booster : lgb.Booster
            LightGBM model to convert to EditableBooster.

        Returns
        -------
        model : EditableBooster
            Booster as an EditableBooster.

        """
        booster_string = BoosterString.from_booster(booster)
        return BoosterStringConverter().convert(booster_string)

    def to_booster(self) -> lgb.Booster:
        """Convert EditableBooster to LightGBM Booster.

        Returns
        -------
        model : lgb.Booster
            EditableBooster as a lgb.Booster object.

        """
        booster_string = self.to_booster_string()
        return booster_string.to_booster()

    def to_booster_string(self) -> BoosterString:
        """Convert EditableBooster to BoosterString.

        Returns
        -------
        model : BoosterString
            EditableBooster as a BoosterString object.

        """
        booster_string_rows = self.header.get_booster_string_rows()

        for tree in self.trees:
            booster_string_rows += tree.get_booster_string_rows()

        booster_string_rows += self.bottom_rows

        return BoosterString(rows=booster_string_rows)
