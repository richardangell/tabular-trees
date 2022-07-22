import pandas as pd
from typing import Union
from dataclasses import dataclass

try:
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
except ModuleNotFoundError as err:
    raise ImportError(
        "scikit-learn must be installed to use functionality in sklearn module"
    ) from err

from .. import checks
from ..trees import BaseModelTabularTrees
from ..trees import export_tree_data


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

    def __post_post_init__(self) -> None:
        """No model specific post init processing."""

        pass


@export_tree_data.register(HistGradientBoostingClassifier)
@export_tree_data.register(HistGradientBoostingRegressor)
def export_tree_data__hist_gradient_boosting_model(
    model: Union[HistGradientBoostingClassifier, HistGradientBoostingRegressor]
) -> ScikitLearnHistTabularTrees:
    """Export tree data from HistGradientBoostingRegressor or Classifier object.

    Parameters
    ----------
    model : Union[HistGradientBoostingClassifier, HistGradientBoostingRegressor]
        Model to export tree data from.

    """

    checks.check_type(
        model, (HistGradientBoostingClassifier, HistGradientBoostingRegressor), "model"
    )

    if not model._is_fitted():
        raise ValueError("model is not fitted, cannot export trees")

    if len(model._predictors[0]) > 1:
        raise NotImplementedError("model with multiple responses not supported")

    tree_data = _extract_hist_gbm_tree_data(model)

    return ScikitLearnHistTabularTrees(tree_data)


def _extract_hist_gbm_tree_data(
    model: Union[HistGradientBoostingClassifier, HistGradientBoostingRegressor]
) -> pd.DataFrame:
    """Extract tree data from _predictors objects in HistGradientBoostingClassifier
    or HistGradientBoostingRegressor model.

    Parameters
    ----------
    model : Union[HistGradientBoostingClassifier, HistGradientBoostingRegressor]
        Model to extract tree data from.

    """

    tree_data_list = []

    for tree_no in range(model.n_iter_):

        tree_df = pd.DataFrame(model._predictors[tree_no][0].nodes)

        tree_df["tree"] = tree_no

        tree_df = tree_df.reset_index().rename(columns={"index": "node"})

        tree_data_list.append(tree_df)

    tree_data = pd.concat(tree_data_list, axis=0)

    return tree_data
