"""Logic to convert from BoosterString to EditableBooster."""

from typing import Union

from .booster_string import BoosterString
from .editable_booster import BoosterHeader, BoosterTree, EditableBooster, FeatureRange
from .helpers import (
    FloatFixedString,
    convert_string_to_int_or_float,
    remove_surrounding_brackets,
)


def convert_booster_string_to_editable_booster(
    booster_string: BoosterString,
) -> EditableBooster:
    """Convert BoosterString to EditableBooster."""
    converter = BoosterStringConverter()

    return converter.convert(booster_string)


class BoosterStringConverter:
    """Logic for converting BoosterString objects to EditableBooster objects."""

    def convert(self, booster_string: BoosterString) -> EditableBooster:
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
