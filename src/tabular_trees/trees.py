"""Module for tree structure classes."""

from abc import ABC
from dataclasses import dataclass, fields
from functools import singledispatch
from typing import Any, Callable

import pandas as pd

from . import checks


@dataclass
class BaseModelTabularTrees(ABC):
    """Base class for model specific TabularTrees classes."""

    data: pd.DataFrame

    def to_dataframe(self) -> pd.DataFrame:
        """Return data for trees object."""
        return self.data.copy()

    def __post_init__(self) -> None:
        """Copy data and set attributes defined on subclass."""
        if not hasattr(self, "data"):
            raise AttributeError("data attribute not set")

        self.data = self.data.copy()

        for field_ in fields(self):
            if not field_.init:
                setattr(self, field_.name, self.data[field_.name].values)


@dataclass
class TabularTrees:
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
        self.trees = trees.copy()
        self.get_root_node_given_tree = get_root_node_given_tree

        checks.check_condition(
            callable(self.get_root_node_given_tree),
            "get_root_node_given_tree is callable",
        )


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
