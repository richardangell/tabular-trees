"""XGBoost trees in tabular format."""

import contextlib
import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import numpy as np
import pandas as pd

from .. import checks


class DumpType(Enum):
    """Type for a specific implementation of DumpReader."""

    TEXT = "text"
    JSON = "json"


class DumpReader(ABC):
    """Abstract base class for xgboost mode dump readers."""

    @property
    @abstractmethod
    def dump_type(self) -> DumpType:
        """Attribute indicating the model dump format supported."""
        raise NotImplementedError

    @abstractmethod
    def read_dump(self, file: str) -> pd.DataFrame:
        """Read xgboost model dump, in specific format."""
        raise NotImplementedError

    def check_file_exists(self, file: str) -> None:
        """Check passed file exists."""
        checks.check_type(file, str, "file")
        checks.check_condition(Path(file).exists(), f"{file} exists")


class JsonDumpReader(DumpReader):
    """Class to read xgboost model (json) file dumps."""

    dump_type = DumpType.JSON
    """Type of model dump file this DumpReader can read."""

    def read_dump(self, file: str) -> pd.DataFrame:
        """Read an xgboost model dump json file and parse it into a tabular structure.

        Json file to read must be the output from xgboost.Booster.dump_model with
        dump_format = 'json'.

        Parameters
        ----------
        file : str
            Xgboost model dump text file.

        Returns
        -------
        model : pd.DataFrame
            DataFrame with columns; tree, nodeid, depth, yes, no, missing,
            split, split_condition, leaf. If the model dump file was output
            with with_stats = True then gain and cover columns are also in
            the output DataFrame.

        """
        super().check_file_exists(file)

        with open(file) as f:
            j = json.load(f)

        tree_list = []

        for i in range(len(j)):
            results_list: list[pd.DataFrame] = []

            self._recursive_pop_children(_dict=j[i], _list=results_list)

            tree_df = pd.concat(results_list, axis=0, sort=True)

            tree_df["tree"] = i

            tree_df = self._fill_depth_for_terminal_nodes(tree_df)

            tree_list.append(tree_df)

        trees_df = pd.concat(tree_list, axis=0, sort=True)

        trees_df.reset_index(inplace=True, drop=True)

        return trees_df

    def _recursive_pop_children(self, _dict: dict, _list: list) -> None:
        """Recursively extract nodes from nested structure and append to list.

        The procedure is as follows; If _dict has no childen i.e. this is a leaf node
        then convert the dict to a DataFrame and append to _list. Otherwise remove
        children from _dict, convert the remaining items to a DataFrame and append to
        _list, then call function on left and right children.

        """
        if "children" in _dict:
            children = _dict.pop("children")

            _list.append(pd.DataFrame(_dict, index=[_dict["nodeid"]]))

            self._recursive_pop_children(children[0], _list)

            self._recursive_pop_children(children[1], _list)

        else:
            _list.append(pd.DataFrame(_dict, index=[_dict["nodeid"]]))

    def _fill_depth_for_terminal_nodes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill in the depth column for terminal nodes.

        The json dump from xgboost does not contain this information.

        """
        for i in df.index:
            if np.isnan(df.loc[i, "depth"].item()):
                row_nodeid = df.loc[i, "nodeid"].item()
                parent_col = "yes" if (df["yes"] == row_nodeid).sum() > 0 else "no"
                df.loc[i, "depth"] = (
                    df.loc[df[parent_col] == row_nodeid, "depth"].item() + 1
                )

        df["depth"] = df["depth"].astype(int)

        return df


class TextDumpReader(DumpReader):
    """Class to read xgboost model (text) file dumps."""

    dump_type = DumpType.TEXT
    """Type of model dump file this DumpReader can read."""

    def read_dump(self, file: str) -> pd.DataFrame:
        """Read an xgboost model dump text file dump and parse into tabular structure.

        Text file to read must be the output from xgboost.Booster.dump_model
        with dump_format = 'text'. Note this argument was only added in 0.81
        and text dump was the default prior to this release.

        Parameters
        ----------
        file : str
            Xgboost model dump text file.

        Returns
        -------
        model : pd.DataFrame
            DataFrame with columns; tree, nodeid, depth, yes, no, missing,
            split, split_condition, leaf. If the model dump file was output
            with with_stats = True then gain and cover columns are also in
            the output DataFrame.

        """
        super().check_file_exists(file)

        with open(file) as f:
            lines = f.readlines()

        tree_no = -1

        lines_list: list[dict] = []

        for i in range(len(lines)):
            # if line is a new tree
            if lines[i][:7] == "booster":
                tree_no += 1

            # else if node row
            else:
                line_dict: dict = {}

                # remove \n from end and any \t from start
                node_str = lines[i][: len(lines[i]) - 1].replace("\t", "")

                line_dict["tree"] = tree_no

                # note this will get tree depth for all nodes, which is not consistent
                # with the json dump output xgb json model dumps only contain depth for
                # the non-terminal nodes
                line_dict["depth"] = lines[i].count("\t")

                # split by :
                node_str_split1 = node_str.split(":")

                # get the node number before the :
                line_dict["nodeid"] = int(node_str_split1[0])

                # else if leaf node
                if node_str_split1[1][:4] == "leaf":
                    node_str_split2 = node_str_split1[1].split(",")

                    line_dict["leaf"] = float(node_str_split2[0].split("=")[1])

                    # if model is dumped with the arg with_stats = False then cover
                    # will not be included in the dump for terminal nodes
                    with contextlib.suppress(IndexError):
                        line_dict["cover"] = float(node_str_split2[1].split("=")[1])

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

                    # if model is dumped with the arg with_stats = False then gain
                    # and cover will not be included in the dump for non-terminal nodes
                    try:
                        # get the child nodes
                        line_dict["gain"] = float(node_str_split4[3].split("=")[1])
                        line_dict["cover"] = float(node_str_split4[4].split("=")[1])

                    except IndexError:
                        pass

                lines_list.append(line_dict)

        lines_df = pd.DataFrame.from_records(lines_list)

        if "cover" in lines_df.columns.values:
            lines_df["cover"] = lines_df["cover"].astype(int)

        lines_df.reset_index(inplace=True, drop=True)

        return lines_df
