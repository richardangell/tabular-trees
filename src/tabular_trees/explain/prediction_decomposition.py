"""Module implementing prediction decomposition method."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from ..checks import check_condition, check_type
from ..trees import TabularTrees


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

    Examples
    --------
    >>> import xgboost as xgb
    >>> import pandas as pd
    >>> from sklearn.datasets import load_diabetes
    >>> from tabular_trees import export_tree_data
    >>> from tabular_trees.explain import decompose_prediction
    >>> # get data in DMatrix
    >>> diabetes = load_diabetes()
    >>> data = xgb.DMatrix(
    ...     diabetes["data"],
    ...     label=diabetes["target"],
    ...     feature_names=diabetes["feature_names"]
    ... )
    >>> # build model
    >>> params = {"max_depth": 3, "verbosity": 0}
    >>> model = xgb.train(params, dtrain=data, num_boost_round=10)
    >>> # export to TabularTrees
    >>> xgboost_tabular_trees = export_tree_data(model)
    >>> tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()
    >>> # get data to score
    >>> scoring_data = pd.DataFrame(diabetes["data"], columns=diabetes["feature_names"])
    >>> row_to_score = scoring_data.iloc[[0]]
    >>> # decompose prediction
    >>> results = decompose_prediction(tabular_trees, row=row_to_score)
    >>> type(results)
    <class 'tabular_trees.explain.PredictionDecomposition'>

    """
    check_type(tabular_trees, TabularTrees, "tabular_trees")
    check_type(row, pd.DataFrame, "row")
    check_condition(row.shape[0] == 1, "row is a single pd.DataFrame row")

    return _decompose_prediction(
        trees_df=tabular_trees.trees,
        row=row,
        calculate_root_node=tabular_trees.get_root_node_given_tree,
    )


def _decompose_prediction(
    trees_df: pd.DataFrame, row: pd.DataFrame, calculate_root_node: Callable
) -> PredictionDecomposition:
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
        current_node["value"] = np.nan

    path = pd.concat([path, current_node], axis=0)

    # as long as we are not at a leaf node already
    if current_node["leaf"].item() != 1:
        # determine if the value of the split variable sends the row left
        # (yes) or right (no)
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

            # for internal nodes record the value of the variable that will be
            # used to split
            if current_node["leaf"].item() != 1:
                current_node["value"] = row[current_node["feature"]].values[0]

            path = pd.concat([path, current_node], axis=0)

            if current_node["leaf"].item() != 1:
                # determine if the value of the split variable sends the row left
                # (yes) or right (no)
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


def _calculate_change_in_node_predictions(path: pd.DataFrame) -> pd.DataFrame:
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
