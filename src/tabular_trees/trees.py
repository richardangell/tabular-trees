"""Module for tree structure classes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import singledispatch
from typing import Any

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

    def __init__(self, trees: pd.DataFrame) -> None:
        pass

    @classmethod
    @property
    @abstractmethod
    def REQUIRED_COLUMNS(cls):
        """REQUIRED_COLUMNS attribute that must be defined in classes
        inheriting from BaseModelTabularTrees."""

        raise NotImplementedError("REQUIRED_COLUMNS attribute not defined")

    @classmethod
    @property
    @abstractmethod
    def SORT_BY_COLUMNS(cls):
        """SORT_BY_COLUMNS attribute that must be defined in classes inheriting
        from BaseModelTabularTrees."""

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

        # reorder columns and sort
        self.trees = self.trees[self.REQUIRED_COLUMNS]
        self.trees = self.trees.sort_values(self.SORT_BY_COLUMNS)

        self.trees = self.trees.reset_index(drop=True)

        self.__post_post_init__()

    def __post_post_init__(self):
        """Method to be called at the end of __post_init__ for model specific
        processing."""

        pass


@singledispatch
def export_tree_data(model: Any):
    """Export tree data from passed model."""

    raise NotImplementedError(f"model type not supported; {type(model)}")


@dataclass
class TabularTrees(BaseModelTabularTrees):
    """Generic tree structure in tabular format.

    Parameters
    ----------
    data : pd.DataFrame
        Tree data in tabular structure.

    """

    trees: pd.DataFrame

    REQUIRED_COLUMNS = [
        "tree",
        "node",
        "left_child",
        "right_child",
        "missing",
        "feature",
        "split_condition",
        "leaf",
        "prediction",
    ]

    SORT_BY_COLUMNS = ["tree", "node"]
