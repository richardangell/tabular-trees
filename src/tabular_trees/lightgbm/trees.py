import pandas as pd
from dataclasses import dataclass

from ..trees import BaseModelTabularTrees


@dataclass
class LightGBMTabularTrees(BaseModelTabularTrees):
    """Class to hold the xgboost trees in tabular format.

    Parameters
    ----------
    trees : pd.DataFrame
        LightGBM tree data output from Booster.trees_to_dataframe.

    """

    trees: pd.DataFrame

    REQUIRED_COLUMNS = [
        "tree_index",
        "node_depth",
        "node_index",
        "left_child",
        "right_child",
        "parent_index",
        "split_feature",
        "split_gain",
        "threshold",
        "decision_type",
        "missing_direction",
        "missing_type",
        "value",
        "weight",
        "count",
    ]

    SORT_BY_COLUMNS = ["tree_index", "node_depth", "node_index"]

    def __post__post__init__(self) -> None:
        """No model specific post init processing."""

        pass
