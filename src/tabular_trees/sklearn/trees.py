import pandas as pd
from dataclasses import dataclass

from ..trees import BaseModelTabularTrees


@dataclass
class ScikitLearnHistTabularTrees(BaseModelTabularTrees):
    """Class to hold the scikit-learn HistGradientBoosting trees in tabular
    format.

    Parameters
    ----------
    trees : pd.DataFrame
        HistGradientBoostingRegressor or Classifier tree data extracted from
        ._predictors attribute.

    """

    trees: pd.DataFrame

    REQUIRED_COLUMNS = [
        "tree",
        "node",
        "value",
        "count",
        "feature_idx",
        "num_threshold",
        "missing_go_to_left",
        "left",
        "right",
        "gain",
        "depth",
        "is_leaf",
        "bin_threshold",
        "is_categorical",
        "bitset_idx",
    ]

    SORT_BY_COLUMNS = ["tree", "node"]

    def __post__post__init__(self) -> None:
        """No model specific post init processing."""

        pass
