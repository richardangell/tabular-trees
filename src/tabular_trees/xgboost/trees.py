import pandas as pd
import numpy as np
from dataclasses import dataclass, field

from .. import checks


@dataclass
class XGBoostTabularTrees:
    """Class to hold the xgboost models in tabular format."""

    trees: pd.DataFrame
    n_trees: int = field(init=False)

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
    ]

    def __post_init__(self):

        checks.check_type(self.trees, pd.DataFrame, "trees")
        checks.check_df_columns(self.trees, self.REQUIRED_COLUMNS)

        self.n_trees = int(self.trees["Tree"].max())

        # reorder columns and sort
        self.trees = self.trees[self.REQUIRED_COLUMNS]
        self.trees = self.trees.sort_values(["Tree", "Node"])

    def get_trees(self, tree_indexes: list[int]):
        """Return the tabular data for specified tree(s) from model."""

        checks.check_type(tree_indexes, list, "tree_indexes")

        for i, tree_index in enumerate(tree_indexes):

            checks.check_type(tree_index, int, f"tree_indexes[{i}]")
            checks.check_condition(tree_index >= 0, f"tree_indexes[{i}] >= 0")
            checks.check_condition(
                tree_index <= self.n_trees,
                f"tree_indexes[{i}] in range for number of trees ({self.n_trees})",
            )

        return self.trees.loc[self.trees["Tree"].isin(tree_indexes)].copy()

    def derive_predictions(self, df):
        """Derive predictons for all nodes in trees.

        Leaf node predictions are available in the leaf column.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame output from either _read_dump_text or _read_dump_json
            functions.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with 'weight' (node predictions), 'H', 'G' and
            'node_type' columns added.

        """

        # identify leaf and internal nodes
        df["node_type"] = "internal"
        df.loc[df["Split"].isnull(), "node_type"] = "leaf"
        leaf_nodes = df["node_type"] == "leaf"

        df["H"] = df["Cover"]
        df["G"] = 0

        # column to hold predictions
        df["weight"] = 0
        df.loc[leaf_nodes, "weight"] = df.loc[leaf_nodes, "Gain"]

        df.loc[leaf_nodes, "G"] = (
            -df.loc[leaf_nodes, "weight"] * df.loc[leaf_nodes, "H"]
        )

        df.reset_index(inplace=True, drop=True)

        # propagate G up from the leaf nodes to internal nodes, for each tree
        df_G_list = [
            self._derive_internal_node_G(df.loc[df["Tree"] == n])
            for n in range(self.n_trees + 1)
        ]

        # append all updated trees
        df_G = pd.concat(df_G_list, axis=0)

        internal_nodes = df_G["node_type"] == "internal"

        # update weight values for internal nodes
        df_G.loc[internal_nodes, "weight"] = (
            -df_G.loc[internal_nodes, "G"] / df_G.loc[internal_nodes, "H"]
        )

        return df_G

    def _derive_internal_node_G(self, tree_df):
        """Function to derive predictons for internal nodes in a single tree.

        This involves starting at each leaf node in the tree and propagating the
        G value back up through the tree, adding this leaf node G to each node
        that is travelled to.

        Parameters:
        -----------
        tree_df : pd.DataFrame
            Rows from corresponding to a single tree, from _derive_predictions.

        Returns
        -------
        pd.DataFrame
            Updated tree_df with G propagated up the tree s.t. each internal node's G value
            is the sum of G for it's child nodes.

        """

        tree_df = tree_df.copy()

        leaf_df = tree_df.loc[tree_df["node_type"] == "leaf"]

        # loop through each leaf node
        for i in leaf_df.index:

            leaf_row = leaf_df.loc[[i]]
            current_node = leaf_row["ID"].item()

            leaf_G = leaf_row["G"].item()

            # if the leaf node is not also the first node in the tree
            if int(current_node.split("-")[1]) > 0:

                # traverse the tree bottom from bottom to top and propagate the G value upwards
                while True:

                    # find parent node row
                    parent = (tree_df["Yes"] == current_node) | (
                        tree_df["No"] == current_node
                    )

                    # get parent node G
                    tree_df.loc[parent, "G"] = tree_df.loc[parent, "G"] + leaf_G

                    # update the current node to be the parent node
                    leaf_row = tree_df.loc[parent]
                    current_node = leaf_row["ID"].item()

                    # if we have made it back to the top node in the tree then stop
                    if int(current_node.split("-")[1]) == 0:

                        break

        return tree_df


@dataclass
class ParsedXGBoostTabularTrees:
    """Class to hold the xgboost models that have been parsed from either text
    or json file, in tabular format."""

    trees: pd.DataFrame
    has_stats: bool = field(init=False)

    REQUIRED_BASE_COLUMNS = [
        "tree",
        "nodeid",
        "depth",
        "yes",
        "no",
        "missing",
        "split",
        "split_condition",
        "leaf",
    ]

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

        checks.check_type(self.trees, pd.DataFrame, "trees")

        if all([column in self.trees.columns.values for column in self.STATS_COLUMNS]):

            all_columns = self.REQUIRED_BASE_COLUMNS + self.STATS_COLUMNS

            checks.check_df_columns(self.trees, all_columns)
            self.trees = self.trees[all_columns]
            self.has_stats = True

        else:

            checks.check_df_columns(self.trees, self.REQUIRED_BASE_COLUMNS)
            self.trees = self.trees[self.REQUIRED_BASE_COLUMNS]
            self.has_stats = False

        self.trees = self.trees.sort_values(["tree", "nodeid"])

    def convert_to_xgboost_tabular_trees(self) -> XGBoostTabularTrees:
        """Return the tree structures as XGBoostTabularTrees class.

        Raises
        ------
        ValueError
            If both gain and cover columns are not present in the trees data.

        """

        if not self.has_stats:

            raise ValueError(
                "Cannot convert to XGBoostTabularTrees class unless statistics"
                " are output. Rerun dump_model with with_stats = True."
            )

        else:

            converted_data = self._create_same_columns_as_xgboost_output(self.trees)

            return XGBoostTabularTrees(converted_data)

    def _create_same_columns_as_xgboost_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """This method converts the DataFrame structure that has been created
        from importing a dump of an xgb.Booster to the same format as that
        output from xgb.Booster.trees_to_dataframe.

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
        """Drop columns not needed in XGBoostTabularTrees structure.

        The columns to be dropped are; depth, leaf.
        """

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
