"""Module for parsing xgboost models."""

import pandas as pd
import numpy as np
import xgboost as xgb
import json
import tempfile
import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from .. import checks
from .trees import ParsedXGBoostTabularTrees


class XGBoostParser:
    """Class that dumps and xgboost Booster and then parses the dumped file."""

    def __init__(self, model, dump_type="json"):

        checks.check_type(model, xgb.core.Booster, "model")
        self.model = model

        checks.check_type(dump_type, str, "dump_type")
        checks.check_condition(
            dump_type in ["json", "text"], "dump_type in ['json', 'text']"
        )
        self.dump_type = dump_type

        if dump_type == "json":
            self.reader = JsonDumpReader()
        else:
            self.reader = TextDumpReader()

        warnings.warn(
            "XGBoostDumpParser class is depreceated, "
            "Booster.trees_to_dataframe is available instead",
            FutureWarning,
        )

    def parse_model(self) -> ParsedXGBoostTabularTrees:
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

        with tempfile.TemporaryDirectory() as tmp_dir:

            tmp_model_dump = str(
                Path(tmp_dir).joinpath(f"temp_model_dump.{self.dump_type}")
            )

            self.model.dump_model(tmp_model_dump, with_stats=True, dump_format="json")

            trees_df = self.reader.read_dump(tmp_model_dump)

        return ParsedXGBoostTabularTrees(trees_df)


class DumpReader(ABC):
    """Abstract base class for parsers."""

    def __init__(self) -> None:

        pass

    @abstractmethod
    def read_dump(self, file: str) -> pd.DataFrame:

        checks.check_type(file, str, "file")
        checks.check_condition(Path(file).exists(), f"{file} exists")


class JsonDumpReader(DumpReader):
    """Class to read xgboost model (json) file dumps."""

    def read_dump(self, file: str) -> pd.DataFrame:
        """Reads an xgboost model dump json file and parses it into a tabular
        structure.

        Json file to read must be the output from xgboost.Booster.dump_model
        with dump_format = 'json'. Note this argument was only added in 0.81
        and the default prior to this release was dump in text format.

        Returns
        -------
            pd.DataFrame: df with columns; tree, nodeid, depth, yes, no, missing, split, split_condition, leaf.
            If the model dump file was output with with_stats = True then gain and cover columns are also
            in the output DataFrame.
            dict : if return_raw_lines is True then the raw contents of the json file are also returned.

        """

        super().read_dump(file)

        with open(file) as f:

            j = json.load(f)

        tree_list = []

        for i in range(len(j)):

            results_list: list[pd.DataFrame] = []

            self._recursive_pop_children(_dict=j[i], _list=results_list, verbose=False)

            tree_df = pd.concat(results_list, axis=0, sort=True)

            tree_df["tree"] = i

            tree_df = self._fill_depth_for_terminal_nodes(tree_df)

            tree_list.append(tree_df)

        trees_df = pd.concat(tree_list, axis=0, sort=True)

        trees_df.reset_index(inplace=True, drop=True)

        return trees_df

    def _recursive_pop_children(self, _dict, _list, verbose=False):
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

            self._recursive_pop_children(children[0], _list, verbose)

            self._recursive_pop_children(children[1], _list, verbose)

        else:

            if verbose:

                print(_dict)

            _list.append(pd.DataFrame(_dict, index=[_dict["nodeid"]]))

    def _fill_depth_for_terminal_nodes(self, df):
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


class TextDumpReader(DumpReader):
    """Class to read xgboost model (json) file dumps."""

    def read_dump(self, file: str) -> pd.DataFrame:
        """Reads an xgboost model dump text file and parses it into a tabular structure.

        Text file to read must be the output from xgboost.Booster.dump_model with dump_format = 'text'.
        Note this argument was only added in 0.81 and text dump was the default prior to this release.

        Parameters
        ----------
        file : str
            Xgboost model dump text file.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns; tree, nodeid, depth, yes, no, missing, split, split_condition, leaf. If
            the model dump file was output with with_stats = True then gain and cover columns are also in
            the output DataFrame.

        """

        super().read_dump(file)

        with open(file) as f:

            lines = f.readlines()

        tree_no = -1

        lines_list: list[dict[str, Union[int, float, str]]] = []

        for i in range(len(lines)):

            # if line is a new tree
            if lines[i][:7] == "booster":

                tree_no += 1

            # else if node row
            else:

                line_dict: dict[str, Union[int, float, str]] = {}

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

                lines_list.append(line_dict)

        lines_df = pd.DataFrame.from_dict(lines_list)

        if "cover" in lines_df.columns.values:

            lines_df["cover"] = lines_df["cover"].astype(int)

        lines_df.reset_index(inplace=True, drop=True)

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
