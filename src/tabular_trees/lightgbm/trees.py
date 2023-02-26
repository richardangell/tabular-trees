"""LightGBM trees in tabular format."""

from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd

from .. import checks
from ..trees import BaseModelTabularTrees, TabularTrees, export_tree_data


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
        "value": "prediction",
    }

    def convert_to_tabular_trees(self) -> TabularTrees:
        """Convert the tree data to a TabularTrees object."""
        trees = self.trees.copy()

        trees = self._derive_leaf_node_flag(trees)

        tree_data_converted = trees[self.COLUMN_MAPPING.keys()].rename(
            columns=self.COLUMN_MAPPING
        )

        return TabularTrees(tree_data_converted)

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
