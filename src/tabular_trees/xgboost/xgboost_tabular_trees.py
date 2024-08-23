"""XGBoost trees in tabular format."""

import json
from dataclasses import dataclass, field, fields

import numpy as np
import pandas as pd
import xgboost as xgb
from numpy.typing import NDArray

from .. import checks
from ..trees import BaseModelTabularTrees, TabularTrees, export_tree_data


def xgboost_get_root_node_given_tree(tree: int) -> str:
    """Return the name of the root node of a given tree."""
    return f"{tree}-0"


@dataclass
class XGBoostTabularTrees(BaseModelTabularTrees):
    """Class to hold the xgboost trees in tabular format."""

    """Class to hold the xgboost trees in tabular format.
    
    Attributes
    ----------
    Tree : NDArray[np.int_]
        blah balh
    Node : NDArray[np.int_]
        dsksddsn
    
    """

    data: pd.DataFrame

    Tree: NDArray[np.int_] = field(init=False, repr=False)
    Node: NDArray[np.int_] = field(init=False, repr=False)
    ID: NDArray[np.object_] = field(init=False, repr=False)
    Feature: NDArray[np.object_] = field(init=False, repr=False)
    Split: NDArray[np.float64] = field(init=False, repr=False)
    Yes: NDArray[np.object_] = field(init=False, repr=False)
    No: NDArray[np.object_] = field(init=False, repr=False)
    Missing: NDArray[np.object_] = field(init=False, repr=False)
    Gain: NDArray[np.float64] = field(init=False, repr=False)
    Cover: NDArray[np.float64] = field(init=False, repr=False)
    Category: NDArray[np.float64] = field(init=False, repr=False)
    G: NDArray[np.float64] = field(init=False, repr=False)
    H: NDArray[np.float64] = field(init=False, repr=False)
    weight: NDArray[np.float64] = field(init=False, repr=False)

    def __post_init__(self):
        """Set attributes from data(frame) columns."""
        for field_ in fields(self):
            if not field_.init:
                setattr(self, field_.name, self.data[field_.name].values)

    @classmethod
    def from_booster(cls, booster: xgb.Booster) -> "XGBoostTabularTrees":
        """Export XGBoostTabularTrees from an xgboost.Booster."""
        checks.check_type(booster, xgb.Booster, "booster")

        model_config = json.loads(booster.save_config())
        train_params = model_config["learner"]["gradient_booster"]["tree_train_param"]

        model_alpha = float(train_params["alpha"])
        model_lambda = float(train_params["lambda"])

        if model_alpha != 0:
            raise ValueError("Only Booster objects with alpha = 0 are supported.")

        tree_data = booster.trees_to_dataframe()

        tree_data_with_predictions = XGBoostTabularTrees.derive_predictions(
            df=tree_data, lambda_=model_lambda
        )

        return XGBoostTabularTrees(trees=tree_data_with_predictions)

    def to_dataframe(self) -> pd.DataFrame:
        """Return data as pd.DataFrame."""
        return self.data

    def to_tabular_trees(self) -> TabularTrees:
        """Convert the tree data to a TabularTrees object."""
        trees = self.data.copy()

        # derive leaf node flag
        trees["leaf"] = (trees["Feature"] == "Leaf").astype(int)

        column_mapping = {
            "Tree": "tree",
            "ID": "node",
            "Yes": "left_child",
            "No": "right_child",
            "Missing": "missing",
            "Feature": "feature",
            "Split": "split_condition",
            "weight": "prediction",
            "leaf": "leaf",
            "Cover": "count",
        }

        tree_data_converted = trees[column_mapping.keys()].rename(
            columns=column_mapping
        )

        return TabularTrees(
            trees=tree_data_converted,
            get_root_node_given_tree=xgboost_get_root_node_given_tree,
        )

    @staticmethod
    def derive_predictions(df: pd.DataFrame, lambda_: float) -> pd.DataFrame:
        """Derive predictons for internal nodes in trees.

        Predictions will be available in 'weight' column in the output.

        Returns
        -------
        pd.DataFrame
            Tree data (trees attribute) with 'weight', 'H' and 'G' columns
            added.

        """
        n_trees = df["Tree"].max()

        # identify leaf and internal nodes
        leaf_nodes = df["Feature"] == "Leaf"
        internal_nodes = ~leaf_nodes

        df["H"] = df["Cover"]
        df["G"] = 0.0

        # column to hold predictions
        df["weight"] = 0.0
        df.loc[leaf_nodes, "weight"] = df.loc[leaf_nodes, "Gain"]

        df.loc[leaf_nodes, "G"] = -df.loc[leaf_nodes, "weight"] * (
            df.loc[leaf_nodes, "H"] + lambda_
        )

        # propagate G up from the leaf nodes to internal nodes, for each tree
        df_g_list = [
            XGBoostTabularTrees._derive_internal_node_g(df.loc[df["Tree"] == n])
            for n in range(n_trees + 1)
        ]

        # append all updated trees
        df_g = pd.concat(df_g_list, axis=0)

        # update weight values for internal nodes
        df_g.loc[internal_nodes, "weight"] = -df_g.loc[internal_nodes, "G"] / (
            df_g.loc[internal_nodes, "H"] + lambda_
        )

        return df_g

    @staticmethod
    def _derive_internal_node_g(tree_df: pd.DataFrame) -> pd.DataFrame:
        """Derive predictons for internal nodes in a single tree.

        This involves starting at each leaf node in the tree and propagating
        the G value back up through the tree, adding this leaf node G to each
        node that is travelled to.

        Parameters
        ----------
        tree_df : pd.DataFrame
            Rows from corresponding to a single tree, from _derive_predictions.

        Returns
        -------
        pd.DataFrame
            Updated tree_df with G propagated up the tree s.t. each internal
            node's g value is the sum of G for it's child nodes.

        """
        tree_df = tree_df.copy()

        leaf_df = tree_df.loc[tree_df["Feature"] == "Leaf"]

        # loop through each leaf node
        for i in leaf_df.index:
            leaf_row = leaf_df.loc[[i]]

            leaf_g = leaf_row["G"].item()
            current_node = leaf_row["Node"].item()
            current_tree_node = leaf_row["ID"].item()

            # if the current node is not also the first node in the tree
            # traverse the tree bottom from bottom to top and propagate the G
            # value upwards
            while current_node > 0:
                # find parent node row
                parent = (tree_df["Yes"] == current_tree_node) | (
                    tree_df["No"] == current_tree_node
                )

                # get parent node G
                tree_df.loc[parent, "G"] = tree_df.loc[parent, "G"] + leaf_g

                # update the current node to be the parent node
                leaf_row = tree_df.loc[parent]
                current_node = leaf_row["Node"].item()
                current_tree_node = leaf_row["ID"].item()

        return tree_df


@export_tree_data.register(xgb.Booster)
def _export_tree_data__xgb_booster(model: xgb.Booster) -> XGBoostTabularTrees:
    """Export tree data from Booster object.

    Parameters
    ----------
    model : Booster
        XGBoost booster to export tree data from.

    """
    checks.check_type(model, xgb.Booster, "model")

    return XGBoostTabularTrees.from_booster(model)
