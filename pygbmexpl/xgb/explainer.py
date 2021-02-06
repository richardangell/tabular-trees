import pandas as pd
import numpy as np
import warnings
import itertools
from math import factorial
from tqdm import tqdm


def decompose_prediction(trees_df, row):

    n_trees = trees_df.tree.max()

    # run terminal_node_path for each tree
    terminal_node_paths = [
        terminal_node_path(tree_df=trees_df.loc[trees_df.tree == n], row=row)
        for n in range(n_trees + 1)
    ]

    # append the paths for each tree
    terminal_node_paths = pd.concat(terminal_node_paths, axis=0)

    return terminal_node_paths


def terminal_node_path(tree_df, row):
    """Traverse tree according to the values in the given row of data.

    Args:
        tree_df (pd.DataFrame): df subset of output from pygbm.expl.xgb.extract_model_predictions
            for single tree.
        row (pd.DataFrame): single row df to explain prediction.

    Returns:
        pd.DataFrame: df where each successive row shows the path of row through the tree.

    """

    # get column headers no rows
    path = tree_df.loc[tree_df.nodeid == -1]

    # get the first node in the tree
    current_node = tree_df.loc[tree_df.nodeid == 0].copy()

    # for internal nodes record the value of the variable that will be used to split
    if current_node["node_type"].item() != "leaf":

        current_node["value"] = row[current_node["split"]].values[0]

    else:

        current_node["value"] = np.NaN

    path = path.append(current_node)

    # as long as we are not at a leaf node already
    if current_node["node_type"].item() != "leaf":

        # determine if the value of the split variable sends the row left (yes) or right (no)
        if (
            row[current_node["split"]].values[0]
            < current_node["split_condition"].values[0]
        ):

            next_node = current_node["yes"].item()

        else:

            next_node = current_node["no"].item()

        # (loop) traverse the tree until a leaf node is reached
        while True:

            current_node = tree_df.loc[tree_df.nodeid == next_node].copy()

            # for internal nodes record the value of the variable that will be used to split
            if current_node["node_type"].item() != "leaf":

                current_node["value"] = row[current_node["split"]].values[0]

            path = path.append(current_node)

            if current_node["node_type"].item() != "leaf":

                # determine if the value of the split variable sends the row left (yes) or right (no)
                if (
                    row[current_node["split"]].values[0]
                    < current_node["split_condition"].values[0]
                ):

                    next_node = current_node["yes"].item()

                else:

                    next_node = current_node["no"].item()

            else:

                break

    # shift the split_vars down by 1 to get the variable which is contributing to the change in prediction
    path["contributing_var"] = path["split"].shift(1)

    # get change in predicted value due to split i.e. contribution for that variable
    path["contribution"] = path["node_prediction"] - path["node_prediction"].shift(
        1
    ).fillna(0)

    path.loc[path.contributing_var.isnull(), "contributing_var"] = "base"

    cols_order = [
        "tree",
        "nodeid",
        "yes",
        "no",
        "missing",
        "split",
        "split_condition",
        "cover",
        "node_prediction",
        "node_type",
        "H",
        "G",
        "value",
        "contributing_var",
        "contribution",
    ]

    return path[cols_order]


def shapley_values(tree_df, row, return_permutations=False):
    """Function to calculate shapley values for xgboost model.

    This is Algorithm 1 as presented in https://arxiv.org/pdf/1802.03888.pdf.

    Note, the algorithm has O(TL2^M) complexity (where M is the number of features)
    and this implementation is not intended to be efficient - rather it is intended
    to illustrate the algorithm - so will likely run very slow for models of any
    significant size.

    Note, it is the users responsibility to pass the relevant columns in row (i.e. the
    columns that were present in the training data available to the model). If extra
    columns are added this will exponentially increase the number of runs - even if
    they are not relevant to the model.

    Parameters
    ----------
    tree_df : pd.DataFrame
        Model (multiple trees) in tabular structure. Should be the output of pygbm.expl.xgb.extract_model_predictions.

    row : pd.Series
        Single row of data to explain prediction.

    return_permutations : bool, default = False
        Should contributions for each feature permutation for each tree (i.e. most granular level) be
        returned? If not overall shapley values are returned.

    """

    warnings.warn(
        "This algorithm is likely to run very slow, it gives the same results but is not the more efficient treeSHAP algorithm."
    )

    tree_df.reset_index(drop=True, inplace=True)

    n_trees = tree_df["tree"].max()

    results_list = []

    for tree_no in range(n_trees + 1):

        tree_rows = tree_df.loc[tree_df["tree"] == tree_no].copy()

        if not tree_rows.shape[0] > 0:

            raise ValueError(f"tree number {tree_no} has no rows")

        tree_values = shapley_values_tree(tree_rows, row, return_permutations)

        tree_values.insert(0, "tree", tree_no)

        results_list.append(tree_values)

    results_df = pd.concat(results_list, axis=0, sort=True)

    if return_permutations:

        return results_df

    else:

        results_df = pd.DataFrame(results_df.sum(axis=0)).T.drop(columns="tree")

        return results_df


def shapley_values_tree(tree_df, row, return_permutations=False):
    """Function to calculate shapley values for single tree.

    Parameters
    ----------
    tree_df : pd.DataFrame
        Single tree in tabular structure. Should be subset of output of pygbm.expl.xgb.extract_model_predictions.

    row : pd.Series
        Single row of data to explain prediction.

    return_permutations : bool, default = False
        Should contributions for each feature permutation be returned? If not overall shapley values (i.e. average
        over all permutations) are returned.

    """

    internal_nodes = tree_df["node_type"] == "internal"

    mean_prediction = (
        tree_df.loc[~internal_nodes, "cover"]
        * tree_df.loc[~internal_nodes, "node_prediction"]
    ).sum() / tree_df.loc[~internal_nodes, "cover"].sum()

    keep_cols = [
        "yes",
        "no",
        "split_condition",
        "split",
        "cover",
        "node_type",
        "node_prediction",
    ]

    tree_df_cols_subset = tree_df[keep_cols].copy()

    tree_df_cols_subset.loc[internal_nodes, "node_prediction"] = "internal"

    cols_rename = {
        "yes": "a",
        "no": "b",
        "split_condition": "t",
        "split": "d",
        "cover": "r",
        "node_prediction": "v",
    }

    tree_df_cols_subset.rename(columns=cols_rename, inplace=True)

    tree_df_cols_subset.drop(columns="node_type", inplace=True)

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

            expected_value_given_features = EXPVALUE(
                row, selected_features, tree_df_cols_subset
            )

            contributions[feature] = expected_value_given_features - current_prediction

            current_prediction = expected_value_given_features

        contributions = pd.DataFrame(contributions, index=[i])

        results_list.append(contributions)

        i += 1

    results_df = pd.concat(results_list, axis=0, sort=True)

    if return_permutations:

        return results_df

    else:

        results_feature_level = pd.DataFrame(
            results_df.drop(columns="permutation").mean(axis=0)
        ).T

        return results_feature_level


def EXPVALUE(x, S, tree):
    """Function to estimate E[f(x)|x_S] - Algorithm 1 from Consistent Individualized Feature Attribution
    for TreeEnsembles.

    Note, the node indices start at 0 for this implementation whereas in the paper they start at 1.

    Note, the arguments [v, a, b, t, r, d] define the tree under consideration.

    Parameters
    ----------
    x : pd.Series
        Single row of data to estimate E[f(x)|x_S] for.

    S : list
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

    return G(0, 1, x, S, tree)


def G(j, w, x, S, tree):
    """Function to recusively traverse down tree and return prediction for x from tree using only
    the selected features in S.

    This algorithm follows the route allowed by the features in S, if a node is encountered that
    is made on a feature not in S then an average of predictions fro both child nodes is used.

    Parameters
    ----------
    j : int
        Node index

    w : float
        Proportion of training samples that meet node considtion

    x : pd.Series
        Row of data to explain prediction for

    S : list
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

        if d[j] in S:

            if x[d[j]] <= t[j]:

                return G(a[j], w, x, S, tree)

            else:

                return G(b[j], w, x, S, tree)

        else:

            return G(a[j], w * r[a[j]] / r[j], x, S, tree) + G(
                b[j], w * r[b[j]] / r[j], x, S, tree
            )
