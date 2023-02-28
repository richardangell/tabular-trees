"""Module for tree structure classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch
from typing import Any, Callable

import pandas as pd

from . import checks


class BaseModelTabularTrees(ABC):
    """Abstract base class for model specific TabularTrees classes.

    Parameters
    ----------
    trees : pd.DataFrame
        Model specific tree data in tabular structure.

    """

    trees: pd.DataFrame

    @abstractmethod
    def __init__(self, trees: pd.DataFrame) -> None:
        pass

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
        - Rows are sorted by tree and node columns
        - The index is reset and original index dropped.

        Raises
        ------
        TypeError
            If self.trees is not a pd.DataFrame.

        ValueError
            If REQUIRED_COLUMNS are not in self.trees.

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
    """Generic tree structure in tabular format.

    Parameters
    ----------
    data : pd.DataFrame
        Tree data in tabular structure.

    """

    trees: pd.DataFrame
    get_root_node_given_tree: Callable

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

    SORT_BY_COLUMNS = ["tree", "node"]


@singledispatch
def export_tree_data(model: Any) -> BaseModelTabularTrees:
    """Export tree data from passed model."""
    raise NotImplementedError(f"model type not supported; {type(model)}")
