"""LightGBM trees in tabular format."""

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from numpy.typing import NDArray

from .. import checks
from ..trees import BaseModelTabularTrees, TabularTrees, export_tree_data


def lightgbm_get_root_node_given_tree(tree: int) -> str:
    """Return the name of the root node of a given tree."""
    return f"{tree}-S0"


@dataclass
class LightGBMTabularTrees(BaseModelTabularTrees):
    """Class to hold the xgboost trees in tabular format."""

    data: pd.DataFrame
    """Tree data."""

    tree_index: NDArray[np.int_] = field(init=False, repr=False)
    node_depth: NDArray[np.int_] = field(init=False, repr=False)
    node_index: NDArray[np.object_] = field(init=False, repr=False)
    left_child: NDArray[np.object_] = field(init=False, repr=False)
    right_child: NDArray[np.object_] = field(init=False, repr=False)
    parent_index: NDArray[np.object_] = field(init=False, repr=False)
    split_feature: NDArray[np.object_] = field(init=False, repr=False)
    split_gain: NDArray[np.float64] = field(init=False, repr=False)
    threshold: NDArray[np.float64] = field(init=False, repr=False)
    decision_type: NDArray[np.object_] = field(init=False, repr=False)
    missing_direction: NDArray[np.object_] = field(init=False, repr=False)
    missing_type: NDArray[np.object_] = field(init=False, repr=False)
    value: NDArray[np.float64] = field(init=False, repr=False)
    weight: NDArray[np.int_] = field(init=False, repr=False)
    count: NDArray[np.int_] = field(init=False, repr=False)

    @classmethod
    def from_booster(cls, booster: lgb.Booster) -> "LightGBMTabularTrees":
        """Create LightGBMTabularTrees from a lgb.Booster object.

        Parameters
        ----------
        booster : lgb.Booster
            LightGBM model to pull tree data from.

        Returns
        -------
        trees : LightGBMTabularTrees
            Model trees in tabular format.

        Examples
        --------
        >>> import lightgbm as lgb
        >>> from sklearn.datasets import load_diabetes
        >>> from tabular_trees import LightGBMTabularTrees
        >>> # get data in Dataset
        >>> diabetes = load_diabetes()
        >>> data = lgb.Dataset(diabetes["data"], label=diabetes["target"])
        >>> # build model
        >>> params = {"max_depth": 3, "verbosity": -1}
        >>> model = lgb.train(params, train_set=data, num_boost_round=10)
        >>> # export to LightGBMTabularTrees
        >>> lightgbm_tabular_trees = LightGBMTabularTrees.from_booster(model)
        >>> type(lightgbm_tabular_trees)
        <class 'tabular_trees.lightgbm.lightgbm_tabular_trees.LightGBMTabularTrees'>

        """
        checks.check_type(booster, lgb.Booster, "booster")
        tree_data = booster.trees_to_dataframe()

        return LightGBMTabularTrees(tree_data)

    def to_tabular_trees(self) -> TabularTrees:
        """Convert the tree data to a TabularTrees object.

        Returns
        -------
        trees : TabularTrees
            Model trees in TabularTrees form.

        """
        trees = self.data.copy()

        # derive leaf node flag
        trees["leaf"] = (trees["split_feature"].isnull()).astype(int)

        column_mapping = {
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

        tree_data_converted = trees[column_mapping.keys()].rename(
            columns=column_mapping
        )

        return TabularTrees(
            trees=tree_data_converted,
            get_root_node_given_tree=lightgbm_get_root_node_given_tree,
        )


@export_tree_data.register(lgb.Booster)
def _export_tree_data__lgb_booster(model: lgb.Booster) -> LightGBMTabularTrees:
    """Export tree data from Booster object.

    Parameters
    ----------
    model : Booster
        Model to export tree data from.

    """
    checks.check_type(model, lgb.Booster, "model")

    return LightGBMTabularTrees.from_booster(model)
