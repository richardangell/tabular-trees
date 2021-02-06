import pandas as pd

import pygbmexpl.helpers as h


class TabularTrees:
    """Class to hold tree structures from different packages in tabular format.

    Different packages containing tree based algorithms need different approaches to convert
    the models into tabular structure. Once the models have been converted to tabular
    structure they can be stored within the a TabularTrees object. TabularTrees does
    various checks to ensure the trees are in a set format.

    Parameters
    ----------
    data : pd.DataFrame
        Trees in tabular structure.

    package : str
        The name of the package which generated the trees.

    package_version : str
        The version of the package which generated the trees.

    """

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
        "node_prediction",
    ]

    INTEGER_COLUMNS = ["tree", "nodeid", "depth"]

    FLOAT_COLUMNS = ["yes", "no", "missing", "leaf", "node_prediction"]

    # columns that should only take null values when the leaf columns takes nulls values
    # and vice versa
    LEAF_INVERTED_NULLS_COLUMNS = ["yes", "no", "missing", "split", "split_condition"]

    def __init__(self, data, package, package_version):

        h.check_type(data, "data", pd.DataFrame)
        h.check_type(package, "package", str)
        h.check_type(package_version, "package_version", str)

        self.package = package
        self.package_version = package_version

        self.check_tree_data(data)
        self.tree_data = data

        self.n_trees = self.calculate_number_trees(self.tree_data)
        self.max_depth = self.calculate_max_depth_trees(self.tree_data)
        self.n_nodes = self.calculate_number_nodes(self.tree_data)
        self.n_leaf_nodes = self.calculate_number_leaf_nodes(self.tree_data)

    def check_tree_data(self, tree_df):
        """Checks on input tree_df to ensure it is in a consistent, expected format.

        Raises section below lists the checks that are performed.

        Raises
        ------
        ValueError
            If tree_df has no rows.

        ValueError
            If any of the checks in h.check_df_columns fail.

        ValueError
            If there are nulls in any of the INTEGER_COLUMNS.

        TypeError
            If any of the INTEGER_COLUMNS are not integer dtypes.

        TypeError
            If any of the FLOAT_COLUMNS are not float dtypes.

        ValueError
            If node_prediction column contains any null values.

        ValueError
            If there are any nulls in LEAF_INVERTED_NULLS_COLUMNS when 'leaf' column
            takes null values. LEAF_INVERTED_NULLS_COLUMNS should only take nulls
            when 'leaf' is not null an vice versa.

        ValueError
            If there are any non-null values in LEAF_INVERTED_NULLS_COLUMNS when 'leaf'
            column is not null. LEAF_INVERTED_NULLS_COLUMNS should only take nulls
            when 'leaf' is not null an vice versa.

        ValueError
            If the first tree in tree_df is not indexed 0.

        ValueError
            If any of the trees in tree_df have 0 rows. Trees with index 0:max(tree_df['tree'])
            are expected in tree_df, i.e. there are no missing trees within that range.

        ValueError
            If any of the trees do not have depth column indexed from 0.

        ValueError
            If any of the trees have 0 rows for a given depth between 0:max(tree_df['depth']).

        ValueError
            If any of the trees do not have nodeid column indexed from 0.

        ValueError
            If any of the trees do not have nodeid column taking unique integer values 0:max(tree_df['nodeid']).

        """

        if not tree_df.shape[0] > 0:
            raise ValueError("tree data has not rows")

        h.check_df_columns(
            df=tree_df,
            expected_columns=self.REQUIRED_COLUMNS,
            allow_unspecified_columns=True,
        )

        if tree_df[self.INTEGER_COLUMNS].isnull().sum().sum() > 0:
            raise ValueError(
                f"nulls present in the following columns; {self.INTEGER_COLUMNS}"
            )

        for column in self.INTEGER_COLUMNS:
            if not pd.api.types.is_integer_dtype(tree_df[column]):
                raise TypeError(f"{column} column should be integer dtype")

        for column in self.FLOAT_COLUMNS:
            if not pd.api.types.is_float_dtype(tree_df[column]):
                raise TypeError(f"{column} column should be float dtype")

        if tree_df["node_prediction"].isnull().sum() > 0:
            raise ValueError("node_prediction column has null values")

        leaf_column_nulls = tree_df["leaf"].isnull()
        leaf_column_non_nulls = ~leaf_column_nulls

        # check that LEAF_INVERTED_NULLS_COLUMNS have no nulls where 'leaf' is null
        if (
            tree_df.loc[leaf_column_nulls, self.LEAF_INVERTED_NULLS_COLUMNS]
            .isnull()
            .sum()
            .sum()
            > 0
        ):
            raise ValueError(
                f"null values present in the following columns; {self.LEAF_INVERTED_NULLS_COLUMNS} when leaf column is null"
            )

        expected_null_values_where_leaf_not_null = (
            len(self.LEAF_INVERTED_NULLS_COLUMNS) * leaf_column_non_nulls.sum()
        )

        # check that LEAF_INVERTED_NULLS_COLUMNS have no non-null values where 'leaf' is not null
        if (
            tree_df.loc[leaf_column_non_nulls, self.LEAF_INVERTED_NULLS_COLUMNS]
            .isnull()
            .sum()
            .sum()
            < expected_null_values_where_leaf_not_null
        ):
            raise ValueError(
                f"non null values present in the following columns; {self.LEAF_INVERTED_NULLS_COLUMNS} when leaf column is not null"
            )

        first_tree = tree_df["tree"].min()

        # check tree column start index is 0
        if not first_tree == 0:
            raise ValueError(f"""first tree is not index 0, got {first_tree}""")

        # for each tree in [0:max(tree_df['tree'])]
        for tree_no in range(tree_df["tree"].max() + 1):

            tree_no_rows = tree_df["tree"] == tree_no

            # check the tree has rows in tree_df
            if not (tree_no_rows).sum() > 0:
                raise ValueError(f"no rows in data for tree {tree_no}")

            tree_min_depth = tree_df.loc[tree_no_rows, "depth"].min()
            tree_max_depth = tree_df.loc[tree_no_rows, "depth"].max()

            # check depth column start index is 0, for given tree
            if not tree_min_depth == 0:
                raise ValueError(
                    f"""first depth index for tree {tree_no} is not 0, got {tree_min_depth}"""
                )

            # check there are rows for the given tree, for each depth in [0:max(tree_df['depth'])]
            for tree_depth in range(tree_max_depth + 1):
                tree_depth_rows = (tree_df["tree"] == tree_no) & (
                    tree_df["depth"] == tree_depth
                )
                if not tree_depth_rows.sum() > 0:
                    raise ValueError(
                        f"no rows in data for tree {tree_no} and depth {tree_depth}"
                    )

            tree_min_nodeid = tree_df.loc[tree_no_rows, "nodeid"].min()
            tree_max_nodeid = tree_df.loc[tree_no_rows, "nodeid"].max()

            # check depth column start index is 0, for given tree
            if not tree_min_nodeid == 0:
                raise ValueError(
                    f"""first nodeid index for tree {tree_no} is not 0, got {tree_min_nodeid}"""
                )

            if not tree_df.loc[tree_no_rows, "nodeid"].nunique() == tree_max_nodeid + 1:
                raise ValueError(
                    f"nodeid is not unique increasing from {tree_min_nodeid} to {tree_max_nodeid}"
                )

    def calculate_number_trees(self, tree_df):
        """Caculate the number of trees.

        This is the max of the tree column plus 1, as the tree index starts at 0.
        """

        return tree_df["tree"].max() + 1

    def calculate_max_depth_trees(self, tree_df):
        """Caculate the max tree depth across all trees.

        This is the max of the depth column plus 1, as the depth index starts at 0.
        """

        return tree_df["depth"].max() + 1

    def calculate_number_leaf_nodes(self, tree_df):
        """Caculate the total number of leaf nodes in the trees.

        This is the total number of rows in the passed DataFrame where the leaf column is not null.
        """

        return (~tree_df["leaf"].isnull()).sum()

    def calculate_number_nodes(self, tree_df):
        """Caculate the total number of nodes in the trees.

        This is the total number of rows in the passed DataFrame.
        """

        return tree_df.shape[0]

    def __repr__(self):
        """Return a string with key info from the object."""

        print_str = (
            f"TabularTrees representation of {self.package} ({self.package_version}) model"
            + f"\n  n trees: {self.n_trees}\n  max depth: {self.max_depth}\n  n nodes: {self.n_nodes}\n  n leaf nodes: {self.n_leaf_nodes}"
        )

        return print_str
