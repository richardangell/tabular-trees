"""XGBoost trees in tabular format."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import xgboost as xgb

from .. import checks
from ..trees import BaseModelTabularTrees, TabularTrees, export_tree_data


@dataclass
class XGBoostTabularTrees(BaseModelTabularTrees):
    """Class to hold the xgboost trees in tabular format.

    Parameters
    ----------
    trees : pd.DataFrame
        XGBoost tree data output from Booster.trees_to_dataframe.

    lambda_ : float = 1.0
        Lambda value used in the xgboost model.

    alpha : float = 0.0
        Alpha value used in the xgboost model. An exception will be raised if
        alpha is non-zero.

    """

    trees: pd.DataFrame
    lambda_: float = 1.0
    alpha: float = 0.0

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

    SORT_BY_COLUMNS = ["Tree", "Node"]

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
    }

    def __post_init__(self):
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

        return TabularTrees(tree_data_converted)

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
        df["G"] = 0

        # column to hold predictions
        df["weight"] = 0
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
def export_tree_data__xgb_booster(model: xgb.Booster) -> XGBoostTabularTrees:
    """Export tree data from Booster object.

    Parameters
    ----------
    model : Booster
        XGBoost booster to export tree data from.

    """
    checks.check_type(model, xgb.Booster, "model")

    tree_data = model.trees_to_dataframe()

    return XGBoostTabularTrees(tree_data)


@dataclass
class ParsedXGBoostTabularTrees(BaseModelTabularTrees):
    """Dataclass for xgboost models that have been parsed from model dump.

    Data maybe have been parsed from text or json file dump.

    Parameters
    ----------
    trees : pd.DataFrame
        XGBoost tree data parsed from the output of Booster.dump_model.

    Attributes
    ----------
    has_stats : bool
        Does the tree data contain 'gain' and 'cover' columns?

    """

    trees: pd.DataFrame

    REQUIRED_COLUMNS = [
        "tree",
        "nodeid",
        "depth",
        "yes",
        "no",
        "missing",
        "split",
        "split_condition",
        "leaf",
        "gain",
        "cover",
    ]

    SORT_BY_COLUMNS = ["tree", "nodeid"]

    STATS_COLUMNS = [
        "gain",
        "cover",
    ]

    # mapping between column names in this class and the column names in the
    # XGBoostTabularTrees class
    COLUMNS_MAPPING = {
        "tree": "Tree",
        "nodeid": "Node",
        "yes": "Yes",
        "no": "No",
        "missing": "Missing",
        "split": "Feature",
        "split_condition": "Split",
        "gain": "Gain",
        "cover": "Cover",
    }

    def __post_init__(self):
        """Check that STATS_COLUMNS are present in the data."""
        if not all(
            [column in self.trees.columns.values for column in self.STATS_COLUMNS]
        ):

            raise ValueError(
                "Cannot create ParsedXGBoostTabularTrees object unless statistics"
                " are output. Rerun dump_model with with_stats = True."
            )

        super().__post_init__()

    def convert_to_xgboost_tabular_trees(self) -> XGBoostTabularTrees:
        """Return the tree structures as XGBoostTabularTrees class.

        Raises
        ------
        ValueError
            If both gain and cover columns are not present in the trees data.

        """
        converted_data = self._create_same_columns_as_xgboost_output(self.trees)

        return XGBoostTabularTrees(converted_data)

    def _create_same_columns_as_xgboost_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert parsed DataFrame dump to xgb.Booster.trees_to_dataframe format.

        This involves the following steps;
        - converting the cover column to float type
        - creating the ID column by combining tree and nodeid columns
        - creating the Categroy column
        - populating the split column for leaf nodes
        - combining the leaf node predictions stored in leaf into Gain
        - converting the yes, no and missing columns to the same format as ID
        - dropping the depth and leaf columns
        - renaming most of the columns according to the COLUMNS_MAPPING

        Parameters
        ----------
        df : pd.DataFrame
            Tree data to convert.

        """
        df = self._convert_cover_dtype(df)
        df = self._create_id_columns(df)
        df = self._create_category_column(df)
        df = self._populate_leaf_node_split_column(df)
        df = self._combine_leaf_and_gain(df)
        df = self._convert_node_columns(df)
        df = self._drop_columns(df)
        df = self._rename_columns(df)

        return df

    def _convert_cover_dtype(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert the cover column to float type."""
        df["cover"] = df["cover"].astype(float)
        return df

    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns according to the mapping defined in COLUMNS_MAPPING."""
        return df.rename(columns=self.COLUMNS_MAPPING)

    def _drop_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop depth, leaf columns not needed in XGBoostTabularTrees structure."""
        return df.drop(columns=["depth", "leaf"])

    def _create_id_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add an ID column onto df by concatenating tree and nodeid."""
        df["ID"] = df["tree"].astype(str) + "-" + df["nodeid"].astype(str)
        return df

    def _create_category_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create the Category column in df."""
        df["Category"] = np.NaN
        return df

    def _populate_leaf_node_split_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Populate leaf node rows of the split column with the value 'Leaf'."""
        leaf_nodes = df["gain"].isnull()
        df.loc[leaf_nodes, "split"] = "Leaf"
        return df

    def _combine_leaf_and_gain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine the values in the leaf column into the gain column.

        The leaf column should only be populated for leaf nodes (giving their
        predicted value) and gain should only be populated for interval nodes.

        """
        leaf_nodes = df["gain"].isnull()

        df.loc[leaf_nodes, "gain"] = df.loc[leaf_nodes, "leaf"]

        if df["gain"].isnull().sum() > 0:

            null_gain_indexes = ",".join(
                [str(x) for x in df.loc[df["gain"].isnull()].index.values.tolist()]
            )

            raise ValueError(
                f"gain column has null values in these indexes after combining leaf predictions; {null_gain_indexes}"
            )

        return df

    def _convert_node_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert yes, no and missing columns into tree-node format."""
        columns = ["yes", "no", "missing"]

        for column in columns:

            df = self._convert_node_column_to_tree_node_format(df, column)

        return df

    def _convert_node_column_to_tree_node_format(
        self, df: pd.DataFrame, column: str
    ) -> pd.DataFrame:
        """Convert a given column into tree-node format."""
        null_rows = df[column].isnull()

        df[column] = (
            df["tree"].astype(str) + "-" + df[column].astype("Int64").astype(str)
        )

        df.loc[null_rows, column] = np.NaN

        return df
