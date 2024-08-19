"""LightGBM trees in tabular format."""

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Union

import lightgbm as lgb
import pandas as pd

from . import checks
from .trees import BaseModelTabularTrees, TabularTrees, export_tree_data


def lightgbm_get_root_node_given_tree(tree: int) -> str:
    """Return the name of the root node of a given tree."""
    return f"{tree}-S0"


@dataclass
class LightGBMTabularTrees(BaseModelTabularTrees):
    """Class to hold the xgboost trees in tabular format."""

    trees: pd.DataFrame
    """Tree data."""

    REQUIRED_COLUMNS = [
        "tree_index",
        "node_depth",
        "node_index",
        "left_child",
        "right_child",
        "parent_index",
        "split_feature",
        "split_gain",
        "threshold",
        "decision_type",
        "missing_direction",
        "missing_type",
        "value",
        "weight",
        "count",
    ]
    """List of columns required in tree data."""

    SORT_BY_COLUMNS = ["tree_index", "node_depth", "node_index"]
    """List of columns to sort tree data by."""

    COLUMN_MAPPING = {
        "tree_index": "tree",
        "node_index": "node",
        "left_child": "left_child",
        "right_child": "right_child",
        "missing_direction": "missing",
        "split_feature": "feature",
        "threshold": "split_condition",
        "leaf": "leaf",
        "count": "count",
        "value": "prediction",
    }
    """Column name mapping between LightGBMTabularTrees and TabularTrees tree data."""

    def __init__(self, trees: pd.DataFrame):
        """Initialise the LightGBMTabularTrees object.

        Parameters
        ----------
        trees : pd.DataFrame
            LightGBM tree data output from Booster.trees_to_dataframe.

        Examples
        --------
        >>> import lightgbm as lgb
        >>> from sklearn.datasets import load_diabetes
        >>> from tabular_trees import export_tree_data
        >>> # get data in Dataset
        >>> diabetes = load_diabetes()
        >>> data = lgb.Dataset(diabetes["data"], label=diabetes["target"])
        >>> # build model
        >>> params = {"max_depth": 3, "verbosity": -1}
        >>> model = lgb.train(params, train_set=data, num_boost_round=10)
        >>> # export to LightGBMTabularTrees
        >>> lightgbm_tabular_trees = export_tree_data(model)
        >>> type(lightgbm_tabular_trees)
        <class 'tabular_trees.lightgbm.LightGBMTabularTrees'>

        """
        self.trees = trees

        self.__post_init__()

    def convert_to_tabular_trees(self) -> TabularTrees:
        """Convert the tree data to a TabularTrees object."""
        trees = self.trees.copy()

        trees = self._derive_leaf_node_flag(trees)

        tree_data_converted = trees[self.COLUMN_MAPPING.keys()].rename(
            columns=self.COLUMN_MAPPING
        )

        return TabularTrees(
            trees=tree_data_converted,
            get_root_node_given_tree=lightgbm_get_root_node_given_tree,
        )

    def _derive_leaf_node_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive a leaf node indiciator flag column."""
        df["leaf"] = (df["split_feature"].isnull()).astype(int)

        return df


@export_tree_data.register(lgb.Booster)
def _export_tree_data__lgb_booster(model: lgb.Booster) -> LightGBMTabularTrees:
    """Export tree data from Booster object.

    Parameters
    ----------
    model : Booster
        Model to export tree data from.

    """
    checks.check_type(model, lgb.Booster, "model")

    tree_data = model.trees_to_dataframe()

    return LightGBMTabularTrees(tree_data)


class FloatFixedString(float):
    """Float with a defined string representation."""

    def __new__(cls, value, string_representation: str):
        """Create and return a new object."""
        return float.__new__(cls, value)

    def __init__(self, value, string_representation: str):
        float.__init__(value)
        self.string_representation = string_representation

    def __str__(self):
        """Return the fixed string representation for the float."""
        return self.string_representation

    def __repr__(self):
        """Return the fixed string representation for the float."""
        return self.string_representation


def try_convert_string_to_int_or_float(value: str) -> Union[int, FloatFixedString, str]:
    """Convert a string value to int, FloatFixedString or return the input string.

    Try to convert to int or FloatFixedString first, if this results in a ValueError
    then return the original string value.

    """
    try:
        return convert_string_to_int_or_float(value)
    except ValueError:
        return value


def convert_string_to_int_or_float(value: str) -> Union[int, FloatFixedString]:
    """Try to convert a string to int or FloatFixedString if int conversion fails."""
    try:
        return int(value)
    except ValueError:
        return FloatFixedString(value, value)


def remove_surrounding_brackets(value: str) -> str:
    """Remove surrounding square brackets from string."""
    return value[1:-1]


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

    def __post_init__(self):
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

    def __post_init__(self):
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


class BoosterString:
    """Editable lightgbm Booster."""

    new_line = "\n"

    def __init__(
        self, booster: Optional[lgb.Booster] = None, rows: Optional[list[str]] = None
    ):
        checks.check_condition(
            booster is not None or rows is not None,
            "either booster or rows must be supplied",
        )

        checks.check_condition(
            not (booster is not None and rows is not None),
            "booster and rows cannot be both supplied",
        )

        if booster is not None:
            self.booster_data = self._booster_to_string(booster)

        elif rows is not None:
            self.booster_data = rows[:]

            try:
                self.to_booster()
            except Exception as err:
                raise ValueError(
                    "supplied rows do not produce a valid booster"
                ) from err

        self.row_markers, self.tree_rows = self._gather_line_markers()

    def _booster_to_string(self, booster: lgb.Booster) -> list[str]:
        """Export Booster object to string and split by line breaks."""
        booster_string = booster.model_to_string()
        booster_string_split = booster_string.split(self.new_line)

        return booster_string_split

    def _get_number_trees_from_tree_sizes_line(self, tree_sizes_line: str) -> int:
        """Get the number of trees in the booster from the tree sizes line."""
        return len(tree_sizes_line.split("=")[1].split(" "))

    def _get_tree_number_from_line(self, tree_line: str) -> int:
        """Extract the tree index from the first line in a tree section."""
        return int(tree_line.replace("Tree=", ""))

    def _gather_line_markers(self) -> tuple[OrderedDict, OrderedDict]:
        """Find specific lines in the booster string data."""
        row_to_find = [
            "tree_sizes=",
            "end of trees",
            "feature_importances:",
            "parameters:",
            "end of parameters",
            "pandas_categorical:",
        ]

        row_to_find_description = [
            "end_header",
            "end_of_trees",
            "feature_importances",
            "parameters",
            "end_of_parameters",
            "pandas_categorical",
        ]

        row_found_indexes: list[int] = []

        tree_rows = OrderedDict()

        tree_sizes_line = self._find_text_in_booster_data(row_to_find[0])
        row_found_indexes.append(tree_sizes_line)

        n_trees = self._get_number_trees_from_tree_sizes_line(
            self.booster_data[tree_sizes_line]
        )

        find_tree_from_row = tree_sizes_line

        for _ in range(n_trees):
            tree_line = self._find_text_in_booster_data("Tree=", find_tree_from_row)
            tree_number = self._get_tree_number_from_line(self.booster_data[tree_line])
            tree_rows[tree_number] = tree_line
            find_tree_from_row = tree_line + 1

        find_from_row = find_tree_from_row

        for string_to_find in row_to_find[1:]:
            string_found_row = self._find_text_in_booster_data(
                string_to_find, find_from_row
            )
            row_found_indexes.append(string_found_row)
            find_from_row = string_found_row + 1

        row_markers = OrderedDict()
        for name, row_index in zip(row_to_find_description, row_found_indexes):
            row_markers[name] = row_index

        return row_markers, tree_rows

    def _find_text_in_booster_data(self, text: str, start: int = 0) -> int:
        for subet_row_number, row in enumerate(self.booster_data[start:]):
            if row.find(text, 0, len(text)) == 0:
                row_number = subet_row_number + start

                return row_number

        raise ValueError(
            f"""unable to find row starting with text '{text}' in booster_data """
            f"starting from index {start}"
        )

    def _get_number_of_rows(self) -> int:
        return len(self.booster_data)

    def to_booster(self) -> lgb.Booster:
        """Convert the BoosterString back to a Booster."""
        booster_string = self.new_line.join(self.booster_data)

        return lgb.Booster(model_str=booster_string)

    def to_editable_booster(self):
        """Export the BoosterString an EditableBooster object."""
        return BoosterStringToEditableBoosterConverter.convert(self)

    def extract_header_rows(self) -> list[str]:
        """Extract the header rows from BoosterString object."""
        end_row_index = self.row_markers["end_header"] + 1
        return self._extract_rows(0, end_row_index)

    def _extract_rows(self, start: int, end: int) -> list[str]:
        """Extract booster rows within range."""
        return self.booster_data[start:end]

    def extract_tree_rows(self, tree_number: int) -> list[str]:
        """Extract rows for given tree number."""
        try:
            start_row_index = self.tree_rows[tree_number]
        except KeyError as err:
            raise ValueError(
                f"requested tree number {tree_number} does not exist in model"
            ) from err

        return self._extract_rows(start_row_index, start_row_index + 19)

    def extract_bottom_rows(self) -> list[str]:
        """Return all rows after the 'end of trees' line to the end."""
        return self._extract_rows(
            self.row_markers["end_of_trees"], self._get_number_of_rows()
        )


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


def convert_booster_string_to_editable_booster(
    booster_string: BoosterString,
) -> EditableBooster:
    """Convert BoosterString to EditableBooster."""
    converter = BoosterStringToEditableBoosterConverter()

    return converter.convert(booster_string)


class BoosterStringToEditableBoosterConverter:
    """Logic for converting BoosterString objects to EditableBooster objects."""

    def convert(self, booster_string: BoosterString) -> EditableBooster:
        """Extract the components from a string representation of lgb.Booster object."""
        header = self._export_header(booster_string)
        trees = self._export_trees(booster_string)
        bottom_rows = booster_string.extract_bottom_rows()

        return EditableBooster(header=header, trees=trees, bottom_rows=bottom_rows)

    def _export_header(self, booster_string: BoosterString) -> BoosterHeader:
        """Export the header information from BoosterString as BoosterHeader object."""
        header_rows = booster_string.extract_header_rows()
        return self._rows_to_header(header_rows)

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
