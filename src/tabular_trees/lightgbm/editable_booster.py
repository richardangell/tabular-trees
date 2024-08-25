"""Module containing editable version of LightGBM Booster objects."""

import warnings
from dataclasses import dataclass, field, fields
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
    """Major version of the LightGBM package."""

    num_class: int
    """Number of (response) classes."""

    num_tree_per_iteration: int
    label_index: int

    max_feature_idx: int
    """Number of features."""

    objective: str
    """Objective for the model."""

    feature_names: list[str]
    """Name of each feature."""

    feature_infos: list[FeatureRange]
    """Range of each feature."""

    tree_sizes: list[int]
    """Number of characters for each tree.

    This is equal to len(BoosterTree.get_booster_sting()) + 1 for each tree in the
    model.

    """

    _list_delimiter = " "
    _new_line = "\n"

    def to_string(self) -> str:
        """Concatenate header information to a single string.

        Returns
        -------
        header : str
            Header information concatenated into one string.

        """
        return self._new_line.join(self.get_booster_string_rows())

    def get_booster_string_rows(self) -> list[str]:
        """Append the booster header as a list of strings.

        Returns
        -------
        header : list[str]
            Header information as a list of strings.

        """
        header_attributes = [field_.name for field_ in fields(self)]

        header_attribute_values = [
            self.__getattribute__(name) for name in header_attributes
        ]

        header_attributes_with_equals_signs = [
            f"{name}={self._concatendate_list_to_string(value)}"
            if type(value) is list
            else f"{name}={value}"
            for name, value in zip(header_attributes, header_attribute_values)
        ]

        return [self.header] + header_attributes_with_equals_signs + [""]

    def _concatendate_list_to_string(self, value: list) -> str:
        """Concatenate list using list_delimiter as the separator."""
        return self._list_delimiter.join([str(list_item) for list_item in value])


@dataclass
class BoosterTree:
    """Data class for individual LightGBM trees."""

    tree: int
    """Tree index."""

    num_leaves: int
    """Number of leaves in tree."""

    num_cat: int

    split_feature: list[int]
    """Split feature indexes for internal nodes."""

    split_gain: list[Union[int, float]]
    """Split gain for internal nodes."""

    threshold: list[Union[int, float]]
    """Split threshold for internal nodes."""

    decision_type: list[int]
    """2 for ordered splits."""

    left_child: list[int]
    """Left child node indexes.

    Leaf nodes indexes are negative and indexed from -1.

    """

    right_child: list[int]
    """Right child node indexes.

    Leaf nodes indexes are negative and indexed from -1.

    """

    leaf_value: list[Union[int, float]]
    """Leaf predictions."""

    leaf_weight: list[Union[int, float]]
    """Sum of Hessian for rows in the leaf node."""

    leaf_count: list[int]
    """Number of rows in the leaf node."""

    internal_value: list[Union[int, float]]
    """Prediction for internal nodes."""

    internal_weight: list[Union[int, float]]
    """Sum of Hessian for rows in the internal node."""

    internal_count: list[int]
    """Number of rows in the internal node."""

    is_linear: int
    shrinkage: Union[int, float]

    _list_delimiter = " "
    _new_line = "\n"

    def __post_init__(self) -> None:
        """Set the tree_attributes value."""
        self._tree_attributes = [field_.name for field_ in fields(self)]

        self._tree_attributes_for_booster_string = ["Tree"] + self._tree_attributes[1:]

    def get_booster_sting(self) -> str:
        """Concatenate tree information to a single string.

        Returns
        -------
        model : str
            Booster as a string.

        """
        return self._new_line.join(self.get_booster_string_rows())

    def get_booster_string_rows(self) -> list[str]:
        """Convert data to rows that could be concatenated to part of BoosterString."""
        tree_attribute_values = [
            self.__getattribute__(attribute_name)
            for attribute_name in self._tree_attributes
        ]

        tree_attributes_with_equals_signs = [
            f"{name}={self._concatendate_list_to_string(value)}"
            if type(value) is list
            else f"{name}={value}"
            for name, value in zip(
                self._tree_attributes_for_booster_string, tree_attribute_values
            )
        ]

        return tree_attributes_with_equals_signs + ["", ""]

    def _concatendate_list_to_string(self, value: list) -> str:
        """Concatenate list using list_delimiter as the separator."""
        return self._list_delimiter.join([str(list_item) for list_item in value])


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
    """Header section for the Booster."""

    trees: list[BoosterTree] = field(repr=False)
    """List of trees in the Booster."""

    bottom_rows: list[str]
    """Unspecified rows at the end of a Booster's string representation.

    These are the rows after the 'end of trees' part of the Booster, once is has been
    converted to strings.

    """

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
        warnings.warn(
            "EditableBooster is experimental and has not been tested with every "
            "option that is available in LightGBM.",
            stacklevel=2,
        )

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
