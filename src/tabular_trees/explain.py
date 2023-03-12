"""Explanations for xgboost models."""

import itertools
import warnings
from dataclasses import dataclass, field
from math import factorial
from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from .checks import check_condition, check_type
from .trees import TabularTrees


@dataclass
class PredictionDecomposition:
    """Prediction decomposition results."""

    summary: pd.DataFrame
    """Prediction contribution for each feature."""

    nodes: pd.DataFrame = field(repr=False)
    """Node level prediction contributions for all trees."""

    def __init__(self, summary: pd.DataFrame, nodes: pd.DataFrame):
        """Initialise the PredictionDecomposition object.

        Parameters
        ----------
        summary : pd.DataFrame
            Prediction contribution for each feature.

        nodes : pd.DataFrame
            Node level prediction contributions for all trees.

        """
        self.summary = summary
        self.nodes = nodes


def decompose_prediction(
    tabular_trees: TabularTrees, row: pd.DataFrame
) -> PredictionDecomposition:
    """Decompose prediction from tree based model with Saabas method[1].

    This method attributes the change in prediction from moving to a lower node to the
    variable that was split on. This can then be summed over all splits in a tree and
    all trees in a model.

    Parameters
    ----------
    tabular_trees : TabularTrees
        Tree based model to explain prediction for.

    row : pd.DataFrame
        Single row of data to explain prediction from tabular_trees object.

    Notes
    -----
    [1] Saabas, Ando (2014) 'Interpreting random forests', Diving into data blog, 19
    October. Available at http://blog.datadive.net/interpreting-random-forests/
    (Accessed 26 February 2023).

    """
    check_type(tabular_trees, TabularTrees, "tabular_trees")
    check_type(row, pd.DataFrame, "row")
    check_condition(row.shape[0] == 1, "row is not 1 row")

    return _decompose_prediction(
        trees_df=tabular_trees.trees,
        row=row,
        calculate_root_node=tabular_trees.get_root_node_given_tree,
    )


def _decompose_prediction(
    trees_df: pd.DataFrame, row: pd.DataFrame, calculate_root_node: Callable
):
    """Decompose prediction from tree based model with Saabas method.

    Parameters
    ----------
    tree_df : pd.DataFrame
        Tree data from TabularTrees object.

    row : pd.DataFrame
        Single row of data to explain prediction.

    calculate_root_node : callable
        Function that can return the root node id when passed tree index.

    """
    n_trees = trees_df.tree.max()

    prediction_decompositions = []

    for n in range(n_trees + 1):

        leaf_node_path = _find_path_to_leaf_node(
            tree_df=trees_df.loc[trees_df.tree == n],
            row=row,
            calculate_root_node=calculate_root_node,
        )

        tree_prediction_decomposition = _calculate_change_in_node_predictions(
            path=leaf_node_path
        )

        prediction_decompositions.append(tree_prediction_decomposition)

    return _format_prediction_decomposition_results(prediction_decompositions)


def _find_path_to_leaf_node(
    tree_df: pd.DataFrame, row: pd.DataFrame, calculate_root_node: Callable
) -> pd.DataFrame:
    """Traverse tree down to leaf for given row of data.

    Parameters
    ----------
    tree_df : pd.DataFrame
        Subset of tree data for a single tree.

    row : pd.DataFrame
        Single row of data (observation) to send through tree to leaf node.

    calculate_root_node : callable
        Function that can return the root node id when passed tree index.

    Returns
    -------
    pd.DataFrame
        DataFrame where each successive row shows the path of row through the tree.

    """
    # get column headers no rows
    path = tree_df.loc[tree_df["node"] == -1]

    root_node_index_for_tree = calculate_root_node(tree_df["tree"].values[0])

    # get the first node in the tree
    current_node = tree_df.loc[tree_df["node"] == root_node_index_for_tree].copy()

    # for internal nodes record the value of the variable that will be used to split
    if current_node["leaf"].item() != 1:

        current_node["value"] = row[current_node["feature"]].values[0]

    else:

        current_node["value"] = np.NaN

    path = path.append(current_node)

    # as long as we are not at a leaf node already
    if current_node["leaf"].item() != 1:

        # determine if the value of the split variable sends the row left (yes) or right (no)
        if (
            row[current_node["feature"]].values[0]
            < current_node["split_condition"].values[0]
        ):

            next_node = current_node["left_child"].item()

        else:

            next_node = current_node["right_child"].item()

        # (loop) traverse the tree until a leaf node is reached
        while True:

            current_node = tree_df.loc[tree_df["node"] == next_node].copy()

            # for internal nodes record the value of the variable that will be used to split
            if current_node["leaf"].item() != 1:

                current_node["value"] = row[current_node["feature"]].values[0]

            path = path.append(current_node)

            if current_node["leaf"].item() != 1:

                # determine if the value of the split variable sends the row left (yes) or right (no)
                if (
                    row[current_node["feature"]].values[0]
                    < current_node["split_condition"].values[0]
                ):

                    next_node = current_node["left_child"].item()

                else:

                    next_node = current_node["right_child"].item()

            else:

                break

    return path


def _calculate_change_in_node_predictions(path: pd.DataFrame):
    """Calcualte change in node prediction through a particular path through the tree.

    Parameters
    ----------
    path : pd.DataFrame
        DataFrame where each successive row shows the next node visited though a tree.
        Must have feautre and prediction columns.

    """
    # shift features down by 1 to get the variable which is contributing to the change
    # in prediction
    path["contributing_feature"] = path["feature"].shift(1)

    # calculate the change in prediction
    path["contribution"] = path["prediction"] - path["prediction"].shift(1).fillna(0)

    path.loc[path["contributing_feature"].isnull(), "contributing_feature"] = "base"

    return path


def _format_prediction_decomposition_results(
    list_decomposition_results: list[pd.DataFrame],
) -> PredictionDecomposition:
    """Combine results for each tree into PredictionDecomposition object.

    The list of individual prediction deomposition results is combined into a single
    DataFrame and contirbutions are summed over trees.

    """
    prediction_decompositions_df = pd.concat(list_decomposition_results, axis=0)

    keep_columns = ["tree", "node", "contributing_feature", "contribution"]
    prediction_decompositions_df_subset = prediction_decompositions_df[
        keep_columns
    ].rename({"node": "node_path"})

    decomposition_summary = pd.DataFrame(
        prediction_decompositions_df_subset.groupby(
            "contributing_feature"
        ).contribution.sum()
    ).reset_index()

    return PredictionDecomposition(
        summary=decomposition_summary, nodes=prediction_decompositions_df_subset
    )


@dataclass
class ShapleyValues:
    """Shapley values results."""

    summary: pd.DataFrame
    """Shapley values at feature level. Average of permutations over features."""

    permutations: pd.DataFrame = field(repr=False)
    """Shapley values for each feature permutation and tree."""

    def __init__(self, summary: pd.DataFrame, permutations: pd.DataFrame):
        """Initialise the ShapleyValues object.

        Parameters
        ----------
        summary : pd.DataFrame
            Shapley values at feature level. Average of permutations over features.

        permutations : pd.DataFrame
            Shapley values for each feature permutation and tree.

        """
        self.summary = summary
        self.permutations = permutations


def calculate_shapley_values(
    tabular_trees: TabularTrees, row: pd.Series
) -> ShapleyValues:
    """Calculate shapley values from TabularTrees model for row of data.

    This is Algorithm 1 as presented in https://arxiv.org/pdf/1802.03888.pdf.

    Parameters
    ----------
    tree_df : pd.DataFrame
        Model (multiple trees) in tabular structure. Should be the output of
        pygbm.expl.xgb.extract_model_predictions.

    row : pd.Series
        Single row of data to explain prediction for. It is the users responsibility to
        pass the relevant columns in row (i.e. the columns used by the model). If extra
        columns are added this will exponentially increase the number of runs - even if
        they are not relevant to the model.

    Notes
    -----
    This algorithm has O(TL2^M) complexity (where M is the number of features) and this
    implementation is not intended to be efficient - rather it is intended to
    illustrate the algorithm. Beware of using this on models or datasets (specifically
    columns) of any significant size.

    """
    warnings.warn(
        "This algorithm has very long runtime. It will produce the same results as "
        "treeSHAP but will take much longer to run."
    )

    check_type(tabular_trees, TabularTrees, "tabular_trees")
    check_type(row, pd.Series, "row")

    return _calculate_shapley_values(tree_df=tabular_trees.trees, row=row)


def _calculate_shapley_values(tree_df: pd.DataFrame, row: pd.Series) -> ShapleyValues:
    """Calculate shapley values for each tree and combine.

    Parameters
    ----------
    tree_df : pd.DataFrame
        Tree data.

    row : pd.Series
        Row of data to explain prediction for.

    """
    # TODO: do not modify TabularTrees data
    tree_df.reset_index(drop=True, inplace=True)

    n_trees = tree_df["tree"].max()

    results_list = []

    for tree_no in range(n_trees + 1):

        tree_rows = tree_df.loc[tree_df["tree"] == tree_no]

        tree_shapley_values = _shapley_values_tree(tree_rows, row)

        results_list.append(tree_shapley_values)

    return _combine_shapley_values_across_trees(results_list)


def _convert_node_columns_to_integer(tree_df: pd.DataFrame) -> pd.DataFrame:
    """Convert node, left_child and right_child columns to integer."""
    node_mapping = {node: i for i, node in enumerate(tree_df["node"].tolist())}

    tree_df["node"] = tree_df["node"].map(node_mapping).astype(int)
    tree_df["left_child"] = tree_df["left_child"].map(node_mapping)
    tree_df["right_child"] = tree_df["right_child"].map(node_mapping)

    return tree_df


def _shapley_values_tree(tree_df, row) -> ShapleyValues:
    """Calculate shapley values for single tree.

    Parameters
    ----------
    tree_df : pd.DataFrame
        Single tree in tabular structure. Should be subset of output of
        pygbm.expl.xgb.extract_model_predictions.

    row : pd.Series
        Single row of data to explain prediction.

    """
    tree_number = tree_df["tree"].values[0]

    internal_nodes = tree_df["leaf"] == 0

    mean_prediction = (
        tree_df.loc[~internal_nodes, "count"]
        * tree_df.loc[~internal_nodes, "prediction"]
    ).sum() / tree_df.loc[~internal_nodes, "count"].sum()

    keep_cols = [
        "node",
        "left_child",
        "right_child",
        "split_condition",
        "feature",
        "count",
        "leaf",
        "prediction",
    ]

    tree_df_cols_subset = tree_df[keep_cols].copy()

    tree_df_cols_subset = _convert_node_columns_to_integer(tree_df_cols_subset)

    tree_df_cols_subset.loc[internal_nodes, "prediction"] = "internal"

    cols_rename = {
        "left_child": "a",
        "right_child": "b",
        "split_condition": "t",
        "feature": "d",
        "count": "r",
        "prediction": "v",
    }

    tree_df_cols_subset.rename(columns=cols_rename, inplace=True)

    tree_df_cols_subset.drop(columns=["leaf", "node"], inplace=True)

    # convert child node index column to pandas int type with nulls
    # this is to prevent error in G when trying to select items from list by index (with float)
    tree_df_cols_subset["a"] = pd.Series(tree_df_cols_subset["a"], dtype="Int64")
    tree_df_cols_subset["b"] = pd.Series(tree_df_cols_subset["b"], dtype="Int64")

    features = row.index.values.tolist()

    results_list = []

    i = 0

    for feature_permutation in tqdm(
        itertools.permutations(features), total=factorial(len(features))
    ):

        feature_permutation = list(feature_permutation)

        selected_features = []

        current_prediction = mean_prediction

        contributions = {
            "bias": mean_prediction,
            "permutation": str(feature_permutation),
        }

        for feature in feature_permutation:

            selected_features.append(feature)

            expected_value_given_features = _expvalue(
                row, selected_features, tree_df_cols_subset
            )

            contributions[feature] = expected_value_given_features - current_prediction

            current_prediction = expected_value_given_features

        contributions_df = pd.DataFrame(contributions, index=[i])

        results_list.append(contributions_df)

        i += 1

    return _format_shapley_value_for_tree(results_list, tree_number)


def _expvalue(x, s, tree):
    """Estimate E[f(x)|x_S].

    Algorithm 1 from Consistent Individualized Feature Attribution for Tree Ensembles.

    Note, the node indices start at 0 for this implementation whereas in the paper they
    start at 1.

    Note, the arguments [v, a, b, t, r, d] define the tree under consideration.

    Parameters
    ----------
    x : pd.Series
        Single row of data to estimate E[f(x)|x_S] for.

    s : list
        subset of features

    tree : pd.DataFrame
        tree structure in tabular form with the following columns;
        v - node values which for internal nodes take the value 'internal' otherwise the predicted value
        for the leaf node
        a - left child node indices
        b - right child node indices
        t - thresholds for split in internal nodes
        r - cover for each node (# samples)
        d - split variable for each node

    """
    return _g(0, 1, x, s, tree)


def _g(j, w, x, s, tree):
    """Recusively traverse down tree and return prediction for x.

    This algorithm follows the route allowed by the features in s, if a node is
    encountered that is made on a feature not in s then an average of predictions
    for both child nodes is used.

    Parameters
    ----------
    j : int
        Node index

    w : float
        Proportion of training samples that meet node considtion

    x : pd.Series
        Row of data to explain prediction for

    s : list
        Subset of features being considered

    tree : pd.DataFrame
        Tree structure in tabular form

    """
    v = tree["v"].tolist()
    a = tree["a"].tolist()
    b = tree["b"].tolist()
    t = tree["t"].tolist()
    r = tree["r"].tolist()
    d = tree["d"].tolist()

    if v[j] != "internal":

        return w * v[j]

    else:

        if d[j] in s:

            if x[d[j]] <= t[j]:

                return _g(a[j], w, x, s, tree)

            else:

                return _g(b[j], w, x, s, tree)

        else:

            return _g(a[j], w * r[a[j]] / r[j], x, s, tree) + _g(
                b[j], w * r[b[j]] / r[j], x, s, tree
            )


def _format_shapley_value_for_tree(
    results_list: list[pd.DataFrame], tree_no: int
) -> ShapleyValues:
    """Format shapley values for a single tree.

    The summary results give the average contribution across all permutations.

    Parameters
    ----------
    results_list : list[pd.DataFrame]
        List of contributions for different feature permutations.

    tree_no: int
        Tree number.

    """
    permutations_df = pd.concat(results_list, axis=0, sort=True)

    permutations_df.insert(0, "tree", tree_no)

    results_feature_level = pd.DataFrame(
        permutations_df.drop(columns="permutation").mean(axis=0)
    ).T

    return ShapleyValues(summary=results_feature_level, permutations=permutations_df)


def _combine_shapley_values_across_trees(
    tree_shapley_values: list[ShapleyValues],
) -> ShapleyValues:
    """Combine shapley values across all trees in the model.

    Permutation results from each ShapleyValues object are simply appended. For the
    summary results, these are appended and then summed over trees.

    Parameters
    ----------
    tree_shapley_values : list[ShapleyValues]
        Shapley values for each tree.

    Notes
    -----
    The summary attribute on the returned ShapleyValues object does not have the 'tree'
    column - as results are summed over all trees.

    """
    summary_appended = pd.concat(
        [shapley_values.summary for shapley_values in tree_shapley_values],
        axis=0,
        sort=True,
    )

    summary_summed_over_trees = pd.DataFrame(summary_appended.sum(axis=0)).T.drop(
        columns="tree"
    )

    permutations_appended = pd.concat(
        [shapley_values.permutations for shapley_values in tree_shapley_values],
        axis=0,
        sort=True,
    )

    return ShapleyValues(
        summary=summary_summed_over_trees, permutations=permutations_appended
    )
