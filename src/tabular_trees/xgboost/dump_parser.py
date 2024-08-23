"""XGBoost trees in tabular format."""

import tempfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb

from .. import checks
from ..trees import BaseModelTabularTrees
from .dump_reader import DumpReader, JsonDumpReader
from .xgboost_tabular_trees import XGBoostTabularTrees


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
        >>> from tabular_trees import XGBoostParser
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
        <class 'tabular_trees.xgboost.dump_parser.ParsedXGBoostTabularTrees'>

        """
        self.trees = trees

        self.__post_init__()

    def __post_init__(self) -> None:
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
        df["Category"] = np.nan
        return df

    def _populate_leaf_node_split_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Populate leaf node rows of the split column with the value 'Leaf'."""
        leaf_nodes = df["gain"].isnull()
        df.loc[leaf_nodes, "split"] = "Leaf"
        return df

    def _combine_leaf_and_gain(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combine the values in the leaf column into the gain column.

        The leaf column should only be populated for leaf nodes (giving their predicted
        value) and gain should only be populated for interval nodes.

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

        df.loc[null_rows, column] = np.nan

        return df


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
                tmp_model_dump, with_stats=True, dump_format=self.reader.dump_type.value
            )

            trees_df = self.reader.read_dump(tmp_model_dump)

        return ParsedXGBoostTabularTrees(trees_df)
