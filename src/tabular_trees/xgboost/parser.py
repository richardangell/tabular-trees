"""Module for parsing xgboost models."""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import tempfile
from pathlib import Path
from copy import deepcopy

from .. import checks
from .. import trees as t


EXPECTED_COLUMNS = {
    "model_dump_no_stats": [
        "tree",
        "nodeid",
        "depth",
        "yes",
        "no",
        "missing",
        "split",
        "split_condition",
        "leaf",
    ],
    "model_dump_with_stats": [
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
    ],
    "tree_df_with_node_predictions": [
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
        "node_type",
        "gain",
        "cover",
        "H",
        "G",
    ],
}


def parse_model(model):
    """Extract predictions for all nodes in an xgboost model.

    Parameters
    ----------
    model : xgb.core.booster
        Xgboost model to parse into tabular structure.

    Returns
    -------
    tabular_trees : TabularTrees
        Model parsed into tabular structure.

    """

    if type(model) is xgb.core.Booster:

        with tempfile.TemporaryDirectory() as tmp_dir:

            tmp_model_dump = str(Path(tmp_dir).joinpath("temp_model_dump.json"))

            model.dump_model(tmp_model_dump, with_stats=True, dump_format="json")

            trees_df = _read_dump_json(tmp_model_dump, False)

        trees_preds_df = _derive_predictions(trees_df)

        checks.check_df_columns(
            df=trees_preds_df,
            expected_columns=EXPECTED_COLUMNS["tree_df_with_node_predictions"],
        )

        tabular_trees = t.TabularTrees(trees_preds_df, "xgboost", xgb.__version__)

        return tabular_trees

    else:

        raise TypeError(
            f"unexpected type for model, expected xgboost.core.Booster but got {type(model)}"
        )


def read_dump(file, return_raw_lines=False):
    """Read an xgboost model dumped to txt or json file.

    Parameters
    ----------
    file : str
        Full path, filename with extension of xgboost model dumped to txt or json format.

    return_raw_lines : bool
        Should the raw contents of file also be returned?

    """

    checks.check_type(file, str, "file")
    checks.check_type(return_raw_lines, bool, "return_raw_lines")

    if file.lower().endswith("txt"):

        return _read_dump_text(file, return_raw_lines)

    elif file.lower().endswith("json"):

        return _read_dump_json(file, return_raw_lines)

    else:

        raise ValueError(
            "file should be the full path, filename with extension of an xgboost model dumped to json or txt"
        )


def _read_dump_json(file, return_raw_lines=False):
    """Reads an xgboost model dump json file and parses it into a tabular structure.

    Json file to read must be the output from xgboost.Booster.dump_model with dump_format = 'json'.
    Note this argument was only added in 0.81 and the default prior to this release was dump in
    text format.

    Parameters
    ----------
        file : str
            Xgboost model dump json file.

        return_raw_lines : bool, default = False
            Should lines read from the json file be returned in a dict as well?

    Returns
    -------
        pd.DataFrame: df with columns; tree, nodeid, depth, yes, no, missing, split, split_condition, leaf.
        If the model dump file was output with with_stats = True then gain and cover columns are also
        in the output DataFrame.
        dict : if return_raw_lines is True then the raw contents of the json file are also returned.

    """

    with open(file) as f:

        j = json.load(f)

    j_copy = deepcopy(j)

    tree_list = []

    for i in range(len(j)):

        results_list = []

        _recursive_pop_children(_dict=j[i], _list=results_list, verbose=False)

        tree_df = pd.concat(results_list, axis=0, sort=True)

        tree_df["tree"] = i

        tree_df = _fill_depth_for_terminal_nodes(tree_df)

        tree_list.append(tree_df)

    trees_df = pd.concat(tree_list, axis=0, sort=True)

    trees_df = _reorder_tree_df(trees_df)

    trees_df.reset_index(inplace=True, drop=True)

    if return_raw_lines:

        return trees_df, j_copy

    else:

        return trees_df


def _recursive_pop_children(_dict, _list, verbose=False):
    """Function to recursively extract nodes from nested structure and append to list.

    Procedure is as follows;
    - if no children item in dict, append items (in pd.DataFrame) to list
    - or remove children item from dict, then append remaining items (in pd.DataFrame) to list
    - then call function on left and right children.

    """

    if "children" in _dict.keys():

        children = _dict.pop("children")

        if verbose:

            print(_dict)

        _list.append(pd.DataFrame(_dict, index=[_dict["nodeid"]]))

        _recursive_pop_children(children[0], _list, verbose)

        _recursive_pop_children(children[1], _list, verbose)

    else:

        if verbose:

            print(_dict)

        _list.append(pd.DataFrame(_dict, index=[_dict["nodeid"]]))


def _fill_depth_for_terminal_nodes(df):
    """Function to fill in the depth column for terminal nodes.

    The json dump from xgboost does not contain this information.
    """

    for i, row in df.iterrows():

        if np.isnan(row["depth"]):

            if (df["yes"] == row["nodeid"]).sum() > 0:

                parent_col = "yes"

            else:

                parent_col = "no"

            df.at[i, "depth"] = df.loc[df[parent_col] == row["nodeid"], "depth"] + 1

    df["depth"] = df["depth"].astype(int)

    return df


def _reorder_tree_df(df):
    """Function to sort and reorder columns df of trees."""

    checks.check_type(df, pd.DataFrame, "df")

    if not df.shape[0] > 0:
        raise ValueError("df has no rows")

    if df.shape[1] == 9:
        col_order = EXPECTED_COLUMNS["model_dump_no_stats"]

    elif df.shape[1] == 11:
        col_order = EXPECTED_COLUMNS["model_dump_with_stats"]

    elif df.shape[1] == 15:
        col_order = EXPECTED_COLUMNS["tree_df_with_node_predictions"]

    else:

        raise ValueError(
            f"Expected 9, 11 or 15 columns in parsed model dump but got {df.shape[1]} ({str(df.columns.values)})"
        )

    # reorder columns
    df = df.loc[:, col_order]
    df.sort_values(["tree", "nodeid"], inplace=True)

    return df


def _read_dump_text(file, return_raw_lines=False):
    """Reads an xgboost model dump text file and parses it into a tabular structure.

    Text file to read must be the output from xgboost.Booster.dump_model with dump_format = 'text'.
    Note this argument was only added in 0.81 and text dump was the default prior to this release.

    Parameters
    ----------
        file : str
            Xgboost model dump json file.

        return_raw_lines : bool, default = False
            Should lines read from the json file be returned in a dict as well?

    Returns
    -------
    pd.DataFrame
        DataFrame with columns; tree, nodeid, depth, yes, no, missing, split, split_condition, leaf. If
        the model dump file was output with with_stats = True then gain and cover columns are also in
        the output DataFrame.
    list
        If return_raw_lines is True then the raw contents of the txt file are also returned.

    """

    with open(file) as f:

        lines = f.readlines()

    tree_no = -1

    lines_list = []

    for i in range(len(lines)):

        # if line is a new tree
        if lines[i][:7] == "booster":

            tree_no += 1

        # else if node row
        else:

            line_dict = {}

            # remove \n from end and any \t from start
            node_str = lines[i][: len(lines[i]) - 1].replace("\t", "")

            line_dict["tree"] = tree_no

            # note this will get tree depth for all nodes, which is not consistent with the json dump output
            # xgb json model dumps only contain depth for the non-terminal nodes
            line_dict["depth"] = lines[i].count("\t")

            # split by :
            node_str_split1 = node_str.split(":")

            # get the node number before the :
            line_dict["nodeid"] = int(node_str_split1[0])

            # else if leaf node
            if node_str_split1[1][:4] == "leaf":

                node_str_split2 = node_str_split1[1].split(",")

                line_dict["leaf"] = float(node_str_split2[0].split("=")[1])

                # if model is dumped with the arg with_stats = False then cover will not be included
                # in the dump for terminal nodes
                try:

                    line_dict["cover"] = float(node_str_split2[1].split("=")[1])

                except IndexError:

                    pass

            # else non terminal node
            else:

                node_str_split2 = node_str_split1[1].split(" ")

                node_str_split3 = (
                    node_str_split2[0].replace("[", "").replace("]", "").split("<")
                )

                # extract split variable name before the <
                line_dict["split"] = node_str_split3[0]

                # extract split point after the <
                line_dict["split_condition"] = float(node_str_split3[1])

                node_str_split4 = node_str_split2[1].split(",")

                # get the child nodes
                line_dict["yes"] = int(node_str_split4[0].split("=")[1])
                line_dict["no"] = int(node_str_split4[1].split("=")[1])
                line_dict["missing"] = int(node_str_split4[2].split("=")[1])

                # if model is dumped with the arg with_stats = False then gain and cover will not
                # be included in the dump for non-terminal nodes
                try:

                    # get the child nodes
                    line_dict["gain"] = float(node_str_split4[3].split("=")[1])
                    line_dict["cover"] = float(node_str_split4[4].split("=")[1])

                except IndexError:

                    pass

            lines_list = lines_list + [line_dict]

    lines_df = pd.DataFrame.from_dict(lines_list)

    if "cover" in lines_df.columns.values:

        lines_df["cover"] = lines_df["cover"].astype(int)

    lines_df = _reorder_tree_df(lines_df)

    lines_df.reset_index(inplace=True, drop=True)

    if return_raw_lines:

        return lines_df, lines

    else:

        return lines_df


def _derive_predictions(df):
    """Function to derive predictons for all nodes in trees.

    Leaf node predictions are available in the leaf column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame output from either _read_dump_text or _read_dump_json functions.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with 'node_prediction', 'H', 'G' and 'node_type' columns added.

    """

    # identify leaf and internal nodes
    df["node_type"] = "internal"
    df.loc[df["split_condition"].isnull(), "node_type"] = "leaf"
    leaf_nodes = df["node_type"] == "leaf"

    df["H"] = df["cover"]
    df["G"] = 0

    # column to hold predictions
    df["weight"] = 0
    df.loc[leaf_nodes, "weight"] = df.loc[leaf_nodes, "leaf"]

    df.loc[leaf_nodes, "G"] = -df.loc[leaf_nodes, "weight"] * df.loc[leaf_nodes, "H"]

    df.reset_index(inplace=True, drop=True)

    n_trees = df.tree.max()

    # propagate G up from the leaf nodes to internal nodes, for each tree
    df_G_list = [
        _derive_internal_node_G(df.loc[df["tree"] == n]) for n in range(n_trees + 1)
    ]

    # append all updated trees
    df_G = pd.concat(df_G_list, axis=0)

    internal_nodes = df_G["node_type"] == "internal"

    # update weight values for internal nodes
    df_G.loc[internal_nodes, "weight"] = (
        -df_G.loc[internal_nodes, "G"] / df_G.loc[internal_nodes, "H"]
    )

    df_G.rename(columns={"weight": "node_prediction"}, inplace=True)

    df_G = _reorder_tree_df(df_G)

    return df_G


def _derive_internal_node_G(tree_df):
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

        # print(i, 'leaf------------------')

        leaf_row = leaf_df.loc[[i]]
        current_node = leaf_row["nodeid"].item()

        leaf_G = leaf_row["G"].item()

        # print('current_node', current_node)
        # print(tree_df)
        # print('----')

        # if the leaf node is not also the first node in the tree
        if current_node > 0:

            # traverse the tree bottom from bottom to top and propagate the G value upwards
            while True:

                # find parent node row
                parent = (tree_df["yes"] == current_node) | (
                    tree_df["no"] == current_node
                )

                # get parent node G
                tree_df.loc[parent, "G"] = tree_df.loc[parent, "G"] + leaf_G

                # update the current node to be the parent node
                leaf_row = tree_df.loc[parent]
                current_node = leaf_row["nodeid"].item()

                # print('current_node', current_node)
                # print(tree_df)
                # print('----')

                # if we have made it back to the top node in the tree then stop
                if current_node == 0:

                    break

    return tree_df
