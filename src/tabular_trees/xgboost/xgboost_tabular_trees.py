"""XGBoost trees in tabular format."""

import json
from dataclasses import dataclass

import pandas as pd
import xgboost as xgb

from .. import checks
from ..trees import BaseModelTabularTrees, TabularTrees, export_tree_data


def xgboost_get_root_node_given_tree(tree: int) -> str:
    """Return the name of the root node of a given tree."""
    return f"{tree}-0"


@dataclass
class XGBoostTabularTrees(BaseModelTabularTrees):
    """Class to hold the xgboost trees in tabular format."""

    trees: pd.DataFrame
    """Tree data."""

    lambda_: float = 1.0
    """Lambda parameter value from XGBoost model."""

    alpha: float = 0.0
    """Alpha parameter value from XGBoost model."""

    REQUIRED_COLUMNS = [
        "Tree",
        "Node",
        "ID",
        "Feature",
        "Split",
        "Yes",
        "No",
        "Missing",
        "Gain",
        "Cover",
        "Category",
        "G",
        "H",
        "weight",
    ]
    """List of columns required in tree data."""

    SORT_BY_COLUMNS = ["Tree", "Node"]
    """List of columns to sort tree data by."""

    COLUMN_MAPPING = {
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
    """Column name mapping between XGBoostTabularTrees and TabularTrees tree data."""

    def __init__(self, trees: pd.DataFrame, lambda_: float = 1.0, alpha: float = 0.0):
        """Initialise the XGBoostTabularTrees object.

        Parameters
        ----------
        trees : pd.DataFrame
            XGBoost tree data output from Booster.trees_to_dataframe.

        lambda_ : float, default = 1.0
            Lambda parameter value from XGBoost model.

        alpha : float default = 0.0
            Alpha parameter value from XGBoost model. Only alpha values of 0 are
            supported. Specifically the internal node prediction logic is only
            defined for alpha = 0.

        Raises
        ------
        ValueError
            If alpha is not 0.

        Examples
        --------
        >>> import xgboost as xgb
        >>> from sklearn.datasets import load_diabetes
        >>> from tabular_trees import export_tree_data
        >>> # get data in DMatrix
        >>> diabetes = load_diabetes()
        >>> data = xgb.DMatrix(diabetes["data"], label=diabetes["target"])
        >>> # build model
        >>> params = {"max_depth": 3, "verbosity": 0}
        >>> model = xgb.train(params, dtrain=data, num_boost_round=10)
        >>> # export to XGBoostTabularTrees
        >>> xgboost_tabular_trees = export_tree_data(model)
        >>> type(xgboost_tabular_trees)
        <class 'tabular_trees.xgboost.XGBoostTabularTrees'>

        """
        self.trees = trees
        self.lambda_ = lambda_
        self.alpha = alpha

        self.__post_init__()

    def __post_init__(self) -> None:
        """Post init checks on regularisation parameters.

        Raises
        ------
        TypeError
            If self.lambda_ is not a float.

        TypeError
            If self.alpha is not a float.

        ValueError
            If self.alpha is not 0.

        """
        checks.check_type(self.lambda_, float, "lambda_")
        checks.check_type(self.alpha, float, "alpha")
        checks.check_condition(self.alpha == 0, "alpha = 0")

        self.trees = self.derive_predictions()

        super().__post_init__()

    def convert_to_tabular_trees(self) -> TabularTrees:
        """Convert the tree data to a TabularTrees object."""
        trees = self.trees.copy()

        trees = self._derive_leaf_node_flag(trees)

        tree_data_converted = trees[self.COLUMN_MAPPING.keys()].rename(
            columns=self.COLUMN_MAPPING
        )

        return TabularTrees(
            trees=tree_data_converted,
            get_root_node_given_tree=xgboost_get_root_node_given_tree,
        )

    def _derive_leaf_node_flag(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive a leaf node indiciator flag column."""
        df["leaf"] = (df["Feature"] == "Leaf").astype(int)

        return df

    def derive_predictions(self) -> pd.DataFrame:
        """Derive predictons for internal nodes in trees.

        Predictions will be available in 'weight' column in the output.

        Returns
        -------
        pd.DataFrame
            Tree data (trees attribute) with 'weight', 'H' and 'G' columns
            added.

        """
        df = self.trees.copy()
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
            df.loc[leaf_nodes, "H"] + self.lambda_
        )

        # propagate G up from the leaf nodes to internal nodes, for each tree
        df_g_list = [
            self._derive_internal_node_g(df.loc[df["Tree"] == n])
            for n in range(n_trees + 1)
        ]

        # append all updated trees
        df_g = pd.concat(df_g_list, axis=0)

        # update weight values for internal nodes
        df_g.loc[internal_nodes, "weight"] = -df_g.loc[internal_nodes, "G"] / (
            df_g.loc[internal_nodes, "H"] + self.lambda_
        )

        return df_g

    def _derive_internal_node_g(self, tree_df: pd.DataFrame) -> pd.DataFrame:
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

    model_config = json.loads(model.save_config())
    train_params = model_config["learner"]["gradient_booster"]["tree_train_param"]

    tree_data = model.trees_to_dataframe()

    return XGBoostTabularTrees(
        trees=tree_data,
        lambda_=float(train_params["lambda"]),
        alpha=float(train_params["alpha"]),
    )
