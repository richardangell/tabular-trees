"""LightGBM trees in tabular format."""

from collections import OrderedDict
from dataclasses import dataclass, field

import lightgbm as lgb
import pandas as pd

from . import checks
from .trees import BaseModelTabularTrees, TabularTrees, export_tree_data


def lightgbm_get_root_node_given_tree(tree: int) -> str:
    """Return the name of the root node of a given tree."""
    return f"{tree}-S0"


@dataclass
class LightGBMTabularTrees(BaseModelTabularTrees):
    """Class to hold the xgboost trees in tabular format.

    Parameters
    ----------
    trees : pd.DataFrame
        LightGBM tree data output from Booster.trees_to_dataframe.

    """

    trees: pd.DataFrame

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

    SORT_BY_COLUMNS = ["tree_index", "node_depth", "node_index"]

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
def export_tree_data__lgb_booster(model: lgb.Booster) -> LightGBMTabularTrees:
    """Export tree data from Booster object.

    Parameters
    ----------
    model : Booster
        Model to export tree data from.

    """
    checks.check_type(model, lgb.Booster, "model")

    tree_data = model.trees_to_dataframe()

    return LightGBMTabularTrees(tree_data)


@dataclass
class FeatureRanges:
    """Feature range information from feature_infos line in Booster text."""

    min: float
    max: float

    def to_string(self) -> str:
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
    feature_infos: list[FeatureRanges]
    tree_sizes: list[int]
    delimiter: str = field(repr=False)

    def to_string(self) -> str:
        """Concatenate header information to a single string."""
        return self.delimiter.join(self.to_list())

    def to_list(self) -> list[str]:
        """Append the booster header as a list of strings."""
        return [
            self.header,
            self.version,
            str(self.num_class),
            str(self.num_tree_per_iteration),
            str(self.label_index),
            str(self.max_feature_idx),
            self.objective,
            " ".join(self.feature_names),
            " ".join(
                [feature_range.to_string() for feature_range in self.feature_infos]
            ),
            " ".join([str(tree_size) for tree_size in self.tree_sizes]),
        ]


class EditableBooster:
    """LightGBM booster object that can be edited."""

    def __init__(self):

        pass


class BoosterString:
    """Editable lightgbm Booster."""

    new_line = "\n"

    def __init__(self, booster: lgb.Booster):

        self.booster_data = self._booster_to_string(booster)
        self.tree_rows, self.row_markers = self._gather_line_markers()

    def _booster_to_string(self, booster: lgb.Booster) -> list[str]:
        """Export Booster object to string and split by line breaks."""
        booster_string = booster.model_to_string()
        booster_string_split = booster_string.split(self.new_line)

        return booster_string_split

    def _get_number_trees_from_tree_sizes_line(self, tree_sizes_line: str) -> int:
        """Get the number of trees in the booster from the tree sizes line."""
        return len(tree_sizes_line.split("=")[1].split(" "))

    def _get_tree_index_from_line(self, tree_line: str) -> int:
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
            tree_index = self._get_tree_index_from_line(self.booster_data[tree_line])
            tree_rows[tree_index] = tree_line
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
            f"""unable to find row starting with text '{text}' in booster_data starting from index {start}"""
        )

    def _get_number_of_rows(self) -> int:

        return len(self.booster_data)

    def to_booster(self) -> lgb.Booster:
        """Convert the BoosterString back to a Booster."""
        booster_string = self.new_line.join(self.booster_data)

        return lgb.Booster(model_str=booster_string)

    def to_editable_booster(self):
        """Export the BoosterString an EditableBooster object."""
        pass

    def _export_header(self) -> BoosterHeader:

        header_rows = self._get_header_rows()
        return self._rows_to_header(header_rows)

    def _get_header_rows(self) -> list[str]:

        end_row_index = self.tree_rows["end_header"] + 1
        return self.booster_data[:end_row_index]

    @staticmethod
    def _split_at_equals(line: str) -> str:

        return line.split("=")[1]

    @staticmethod
    def _extract_str_value(line: str) -> str:

        return BoosterString._split_at_equals(line)

    @staticmethod
    def _extract_int_value(line: str) -> int:

        return int(BoosterString._split_at_equals(line))

    @staticmethod
    def _extract_list_values(line: str, delimiter: str = " ") -> list[str]:

        return BoosterString._split_at_equals(line).split(delimiter)

    @staticmethod
    def _feature_range_from_string(string: str) -> FeatureRanges:

        string_without_brackets = string.replace("[", "").replace("]", "")
        string_split = string_without_brackets.split(":")

        return FeatureRanges(min=float(string_split[0]), max=float(string_split[1]))

    def _rows_to_header(self, rows) -> BoosterHeader:

        return BoosterHeader(
            header=rows[0],
            version=BoosterString._extract_str_value(rows[1]),
            num_class=BoosterString._extract_int_value(rows[2]),
            num_tree_per_iteration=BoosterString._extract_int_value(rows[3]),
            label_index=BoosterString._extract_int_value(rows[4]),
            max_feature_idx=BoosterString._extract_int_value(rows[5]),
            objective=BoosterString._extract_str_value(rows[6]),
            feature_names=BoosterString._extract_list_values(rows[7]),
            feature_infos=[
                BoosterString._feature_range_from_string(v)
                for v in BoosterString._extract_list_values(rows[8])
            ],
            tree_sizes=[int(v) for v in BoosterString._extract_list_values(rows[9])],
            delimiter=self.new_line,
        )
