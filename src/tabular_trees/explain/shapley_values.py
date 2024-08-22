"""Module implementing shapley values calculation."""

import itertools
import warnings
from dataclasses import dataclass, field
from math import factorial

import numpy as np
import pandas as pd
from tqdm import tqdm

from ..checks import check_type
from ..trees import TabularTrees


@dataclass
class ShapleyValues:
    """Shapley values results."""

    summary: pd.DataFrame
    """Shapley values at feature level.

    Average of permutations over features.

    """

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

    Examples
    --------
    >>> import xgboost as xgb
    >>> import pandas as pd
    >>> from sklearn.datasets import load_diabetes
    >>> from tabular_trees import export_tree_data
    >>> from tabular_trees.explain import calculate_shapley_values
    >>> # get data in DMatrix
    >>> diabetes = load_diabetes()
    >>> data = xgb.DMatrix(
    ...     diabetes["data"][:,:3],
    ...     label=diabetes["target"],
    ...     feature_names=diabetes["feature_names"][:3]
    ... )
    >>> # build model
    >>> params = {"max_depth": 3, "verbosity": 0}
    >>> model = xgb.train(params, dtrain=data, num_boost_round=10)
    >>> # export to TabularTrees
    >>> xgboost_tabular_trees = export_tree_data(model)
    >>> tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()
    >>> # get data to score
    >>> scoring_data = pd.DataFrame(
    ...     diabetes["data"][:,:3],
    ...     columns=diabetes["feature_names"][:3]
    ... )
    >>> row_to_score = scoring_data.iloc[0]
    >>> # calculate shapley values
    >>> results = calculate_shapley_values(tabular_trees, row=row_to_score)
    >>> type(results)
    <class 'tabular_trees.explain.ShapleyValues'>

    """
    warnings.warn(
        "This algorithm has very long runtime. It will produce the same results as "
        "treeSHAP but will take much longer to run.",
        stacklevel=2,
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


def convert_node_columns_to_integer(tree_df: pd.DataFrame) -> pd.DataFrame:
    """Convert node, left_child and right_child columns to integer."""
    node_mapping = {node: i for i, node in enumerate(tree_df["node"].tolist())}

    tree_df["node"] = tree_df["node"].map(node_mapping).astype(int)
    tree_df["left_child"] = tree_df["left_child"].map(node_mapping)
    tree_df["right_child"] = tree_df["right_child"].map(node_mapping)

    return tree_df


def _shapley_values_tree(tree_df: pd.DataFrame, row: pd.Series) -> ShapleyValues:
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

    tree_df_cols_subset = convert_node_columns_to_integer(tree_df_cols_subset)

    tree_df_cols_subset.loc[internal_nodes, "prediction"] = np.nan

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
    # this is to prevent error in G when trying to select items from list by
    # index (with float)
    tree_df_cols_subset["a"] = pd.Series(tree_df_cols_subset["a"], dtype="Int64")
    tree_df_cols_subset["b"] = pd.Series(tree_df_cols_subset["b"], dtype="Int64")

    features = row.index.values.tolist()

    results_list = []

    for i, feature_permutation in enumerate(
        tqdm(itertools.permutations(features), total=factorial(len(features)))
    ):
        selected_features = []

        current_prediction = mean_prediction

        contributions = {
            "bias": mean_prediction,
            "permutation": str(list(feature_permutation)),
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

    return _format_shapley_value_for_tree(results_list, tree_number)


def _expvalue(x: pd.Series, s: list, tree: pd.DataFrame) -> float:
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
        v - node values which for internal nodes take the value 'internal' otherwise
        the predicted value for the leaf node
        a - left child node indices
        b - right child node indices
        t - thresholds for split in internal nodes
        r - cover for each node (# samples)
        d - split variable for each node

    """
    return _g(0, 1, x, s, tree)


def _g(j: int, w: float, x: pd.Series, s: list, tree: pd.DataFrame) -> float:
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
    v: list[float] = tree["v"].tolist()
    a = tree["a"].tolist()
    b = tree["b"].tolist()
    t = tree["t"].tolist()
    r = tree["r"].tolist()
    d = tree["d"].tolist()

    if not np.isnan(v[j]):
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
