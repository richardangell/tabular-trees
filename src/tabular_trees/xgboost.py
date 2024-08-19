"""XGBoost trees in tabular format."""

import contextlib
import json
import tempfile
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xgboost as xgb

from . import checks
from .trees import BaseModelTabularTrees, TabularTrees, export_tree_data


def xgboost_get_root_node_given_tree(tree: int) -> str:
    """Return the name of the root node of a given tree."""
    return f"{tree}-0"


@dataclass
class XGBoostTabularTrees(BaseModelTabularTrees):
    """Class to hold the xgboost trees in tabular format."""

    trees: pd.DataFrame
    """Tree data."""

    lambda_: float = 1.0
    """Lambda parameter value from XGBoost model."""

    alpha: float = 0.0
    """Alpha parameter value from XGBoost model."""

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
    """List of columns required in tree data."""

    SORT_BY_COLUMNS = ["Tree", "Node"]
    """List of columns to sort tree data by."""

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
        "Cover": "count",
    }
    """Column name mapping between XGBoostTabularTrees and TabularTrees tree data."""

    def __init__(self, trees: pd.DataFrame, lambda_: float = 1.0, alpha: float = 0.0):
        """Initialise the XGBoostTabularTrees object.

        Parameters
        ----------
        trees : pd.DataFrame
            XGBoost tree data output from Booster.trees_to_dataframe.

        lambda_ : float, default = 1.0
            Lambda parameter value from XGBoost model.

        alpha : float default = 0.0
            Alpha parameter value from XGBoost model. Only alpha values of 0 are
            supported. Specifically the internal node prediction logic is only
            defined for alpha = 0.

        Raises
        ------
        ValueError
            If alpha is not 0.

        Examples
        --------
        >>> import xgboost as xgb
        >>> from sklearn.datasets import load_diabetes
        >>> from tabular_trees import export_tree_data
        >>> # get data in DMatrix
        >>> diabetes = load_diabetes()
        >>> data = xgb.DMatrix(diabetes["data"], label=diabetes["target"])
        >>> # build model
        >>> params = {"max_depth": 3, "verbosity": 0}
        >>> model = xgb.train(params, dtrain=data, num_boost_round=10)
        >>> # export to XGBoostTabularTrees
        >>> xgboost_tabular_trees = export_tree_data(model)
        >>> type(xgboost_tabular_trees)
        <class 'tabular_trees.xgboost.XGBoostTabularTrees'>

        """
        self.trees = trees
        self.lambda_ = lambda_
        self.alpha = alpha

        self.__post_init__()

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

        return TabularTrees(
            trees=tree_data_converted,
            get_root_node_given_tree=xgboost_get_root_node_given_tree,
        )

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
def _export_tree_data__xgb_booster(model: xgb.Booster) -> XGBoostTabularTrees:
    """Export tree data from Booster object.

    Parameters
    ----------
    model : Booster
        XGBoost booster to export tree data from.

    """
    checks.check_type(model, xgb.Booster, "model")

    model_config = json.loads(model.save_config())
    train_params = model_config["learner"]["gradient_booster"]["updater"][
        "grow_colmaker"
    ]["train_param"]

    tree_data = model.trees_to_dataframe()

    return XGBoostTabularTrees(
        trees=tree_data,
        lambda_=float(train_params["lambda"]),
        alpha=float(train_params["alpha"]),
    )


@dataclass
class ParsedXGBoostTabularTrees(BaseModelTabularTrees):
    """Dataclass for XGBoost models that have been parsed from a model dump.

    Data maybe have been parsed from text or json file dump.
    """

    trees: pd.DataFrame
    """Tree data."""

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
    """List of columns required in tree data."""

    SORT_BY_COLUMNS = ["tree", "nodeid"]
    """List of columns to sort tree data by."""

    STATS_COLUMNS = [
        "gain",
        "cover",
    ]
    """Data items included in XGBoost model dump if with_stats = True in dump_model."""

    # mapping between column names in this class and the column names in the
    # XGBoostTabularTrees class
    COLUMN_MAPPING = {
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
    """Column name mapping between ParsedXGBoostTabularTrees and XGBoostTabularTrees
    tree data."""

    def __init__(self, trees: pd.DataFrame):
        """Initialise the ParsedXGBoostTabularTrees object.

        Parameters
        ----------
        trees : pd.DataFrame
            XGBoost tree data parsed from the output of Booster.dump_model.

        Raises
        ------
        ValueError
            If alpha is not 0.

        Examples
        --------
        >>> import xgboost as xgb
        >>> from sklearn.datasets import load_diabetes
        >>> from tabular_trees.xgboost import XGBoostParser
        >>> # get data in DMatrix
        >>> diabetes = load_diabetes()
        >>> data = xgb.DMatrix(diabetes["data"], label=diabetes["target"])
        >>> # build model
        >>> params = {"max_depth": 3, "verbosity": 0}
        >>> model = xgb.train(params, dtrain=data, num_boost_round=10)
        >>> # parse the model
        >>> xgbooster_parser = XGBoostParser(model)
        >>> # export to ParsedXGBoostTabularTrees
        >>> parsed_xgboost_tabular_trees = xgbooster_parser.parse_model()
        >>> type(parsed_xgboost_tabular_trees)
        <class 'tabular_trees.xgboost.ParsedXGBoostTabularTrees'>

        """
        self.trees = trees

        self.__post_init__()

    def __post_init__(self):
        """Check that STATS_COLUMNS are present in the data."""
        checks.check_condition(
            all(column in self.trees.columns.values for column in self.STATS_COLUMNS),
            "Cannot create ParsedXGBoostTabularTrees object unless statistics "
            "are output. Rerun dump_model with with_stats = True.",
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
        """Rename columns according to the mapping defined in COLUMN_MAPPING."""
        return df.rename(columns=self.COLUMN_MAPPING)

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
                "gain column has null values in these indexes after combining leaf "
                f"predictions; {null_gain_indexes}"
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


class DumpReader(ABC):
    """Abstract base class for xgboost mode dump readers."""

    @property
    @abstractmethod
    def dump_type(self):
        """Attribute indicating the model dump format supported."""
        raise NotImplementedError

    @abstractmethod
    def read_dump(self, file: str) -> None:
        """Read xgboost model dump, in specific format."""
        checks.check_type(file, str, "file")
        checks.check_condition(Path(file).exists(), f"{file} exists")


class JsonDumpReader(DumpReader):
    """Class to read xgboost model (json) file dumps."""

    dump_type = "json"
    """Type of model dump file this DumpReader can read."""

    def read_dump(self, file: str) -> pd.DataFrame:
        """Read an xgboost model dump json file and parse it into a tabular structure.

        Json file to read must be the output from xgboost.Booster.dump_model with
        dump_format = 'json'.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns; tree, nodeid, depth, yes, no, missing,
            split, split_condition, leaf. If the model dump file was output
            with with_stats = True then gain and cover columns are also in
            the output DataFrame.

        """
        super().read_dump(file)

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

    def _recursive_pop_children(self, _dict: dict, _list: list):
        """Recursively extract nodes from nested structure and append to list.

        The procedure is as follows;
        If _dict has no childen i.e. this is a leaf node then convert the dict
        to a DataFrame and append to _list.
        Otherwise remove children from _dict, convert the remaining items to a
        DataFrame and append to _list, then call function on left and right
        children.

        """
        if "children" in _dict:
            children = _dict.pop("children")

            _list.append(pd.DataFrame(_dict, index=[_dict["nodeid"]]))

            self._recursive_pop_children(children[0], _list)

            self._recursive_pop_children(children[1], _list)

        else:
            _list.append(pd.DataFrame(_dict, index=[_dict["nodeid"]]))

    def _fill_depth_for_terminal_nodes(self, df: pd.DataFrame):
        """Fill in the depth column for terminal nodes.

        The json dump from xgboost does not contain this information.

        """
        for i, row in df.iterrows():
            if np.isnan(row["depth"]):
                parent_col = "yes" if (df["yes"] == row["nodeid"]).sum() > 0 else "no"

                df.at[i, "depth"] = df.loc[df[parent_col] == row["nodeid"], "depth"] + 1

        df["depth"] = df["depth"].astype(int)

        return df


class TextDumpReader(DumpReader):
    """Class to read xgboost model (text) file dumps."""

    dump_type = "text"
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
        pd.DataFrame
            DataFrame with columns; tree, nodeid, depth, yes, no, missing,
            split, split_condition, leaf. If the model dump file was output
            with with_stats = True then gain and cover columns are also in
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

        lines_df = pd.DataFrame.from_dict(lines_list)

        if "cover" in lines_df.columns.values:
            lines_df["cover"] = lines_df["cover"].astype(int)

        lines_df.reset_index(inplace=True, drop=True)

        return lines_df


class XGBoostParser:
    """Class that dumps an xgboost Booster then parses the dumped file."""

    def __init__(
        self, model: xgb.core.Booster, reader: Optional[DumpReader] = None
    ) -> None:
        """Initialise the XGBoostParser object.

        Parameters
        ----------
        model : xgb.core.Booster
            Model to parse trees into tabular data.

        reader : Optional[DumpReader], default = None
            Object capable of reading dumped xgboost model. If no value is passed
            then JsonDumpReader() is used.

        """
        checks.check_type(model, xgb.core.Booster, "model")
        checks.check_type(reader, DumpReader, "reader", True)

        if reader is None:
            reader_object: DumpReader = JsonDumpReader()
        else:
            reader_object = reader

        self.model = model
        self.reader = reader_object

        warnings.warn(
            "XGBoostDumpParser class is depreceated, "
            "Booster.trees_to_dataframe is available instead",
            FutureWarning,
            stacklevel=2,
        )

    def parse_model(self) -> ParsedXGBoostTabularTrees:
        """Dump model and then parse into tabular structure.

        Tree data is returned in a ParsedXGBoostTabularTrees object.

        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_model_dump = str(
                Path(tmp_dir).joinpath(f"temp_model_dump.{self.reader.dump_type}")
            )

            self.model.dump_model(
                tmp_model_dump, with_stats=True, dump_format=self.reader.dump_type
            )

            trees_df = self.reader.read_dump(tmp_model_dump)

        return ParsedXGBoostTabularTrees(trees_df)
