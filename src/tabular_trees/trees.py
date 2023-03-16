"""Module for tree structure classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Callable

import pandas as pd

from . import checks


class BaseModelTabularTrees(ABC):
    """Abstract base class for model specific TabularTrees classes."""

    trees: pd.DataFrame
    """Tree data."""

    @property
    @abstractmethod
    def REQUIRED_COLUMNS(self):  # noqa: N802
        """Attribute that must be defined in BaseModelTabularTrees subclasses."""
        raise NotImplementedError("REQUIRED_COLUMNS attribute not defined")

    @property
    @abstractmethod
    def SORT_BY_COLUMNS(self):  # noqa: N802
        """Attribute that must be defined in BaseModelTabularTrees subclasses."""
        raise NotImplementedError("SORT_BY_COLUMNS attribute not defined")

    def __post_init__(self) -> None:
        """Post init checks and processing.

        Processing on the trees attribute is as follows;
        - Columns are ordered into REQUIRED_COLUMNS order
        - Rows are sorted by SORT_BY_COLUMNS columns
        - The index is reset and original index dropped.

        Raises
        ------
        AttributeError
            If object does not have trees attribute.

        TypeError
            If trees attribute is not a pd.DataFrame.

        TypeError
            If REQUIRED_COLUMNS attribute is not a list.

        TypeError
            If SORT_BY_COLUMNS attribute is not a list.

        ValueError
            If REQUIRED_COLUMNS are not in trees attribute.

        ValueError
            If SORT_BY_COLUMNS is not a subset of REQUIRED_COLUMNS.

        """
        if not hasattr(self, "trees"):
            raise AttributeError("trees attribute not set")

        checks.check_type(self.trees, pd.DataFrame, "trees")
        checks.check_type(self.REQUIRED_COLUMNS, list, "REQUIRED_COLUMNS")
        checks.check_type(self.SORT_BY_COLUMNS, list, "SORT_BY_COLUMNS")

        checks.check_df_columns(self.trees, self.REQUIRED_COLUMNS)

        checks.check_condition(
            all([column in self.REQUIRED_COLUMNS for column in self.SORT_BY_COLUMNS]),
            "SORT_BY_COLUMNS is a subset of REQUIRED_COLUMNS",
        )

        self.trees = self.trees[self.REQUIRED_COLUMNS]
        self.trees = self.trees.sort_values(self.SORT_BY_COLUMNS)
        self.trees = self.trees.reset_index(drop=True)


@dataclass
class TabularTrees(BaseModelTabularTrees):
    """Generic tree structure in tabular format."""

    trees: pd.DataFrame
    """Tree data."""

    get_root_node_given_tree: Callable
    """Function that returns the name of the root node for a given tree index."""

    REQUIRED_COLUMNS = [
        "tree",
        "node",
        "left_child",
        "right_child",
        "missing",
        "feature",
        "split_condition",
        "leaf",
        "count",
        "prediction",
    ]
    """List of columns required in tree data."""

    SORT_BY_COLUMNS = ["tree", "node"]
    """List of columns to sort tree data by."""

    def __init__(self, trees: pd.DataFrame, get_root_node_given_tree: Callable):
        """Initialise the TabularTrees object.

        Parameters
        ----------
        trees : pd.DataFrame
            Tree data in tabular structure.

        """
        self.trees = trees
        self.get_root_node_given_tree = get_root_node_given_tree

        checks.check_condition(
            callable(self.get_root_node_given_tree),
            "get_root_node_given_tree is not callable",
        )

        self.__post_init__()


@singledispatch
def export_tree_data(model: Any) -> BaseModelTabularTrees:
    """Export tree data from model.

    The model types that are supported depend on the packages that are installed in the
    Python environment that tabular_trees is running. For example if xgboost is
    installed then xgboost Booster objects can be exported.

    Parameters
    ----------
    model : Any
        Model to export tree data from.

    Raises
    ------
    NotImplementedError
        If the type of the passed model is not supported.

    """
    raise NotImplementedError(f"model type not supported; {type(model)}")
