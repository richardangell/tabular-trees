"""LightGBM trees in tabular format."""

from dataclasses import dataclass

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
