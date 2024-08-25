"""Scikit-learn histogram based GBM trees in tabular format."""

from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    from sklearn.ensemble import (  # type: ignore[import-not-found]
        HistGradientBoostingClassifier,
        HistGradientBoostingRegressor,
    )
except ModuleNotFoundError as err:
    raise ImportError(
        "scikit-learn must be installed to use functionality in sklearn module"
    ) from err

from .. import checks
from ..trees import BaseModelTabularTrees, export_tree_data


@dataclass
class ScikitLearnHistTabularTrees(BaseModelTabularTrees):
    """Scikit-Learn HistGradientBoosting trees in tabular format."""

    data: pd.DataFrame
    """Tree data."""

    tree: NDArray[np.int_] = field(init=False, repr=False)
    """Tree index."""

    node: NDArray[np.int_] = field(init=False, repr=False)
    """Node index in tree."""

    value: NDArray[np.float64] = field(init=False, repr=False)
    """Node prediction."""

    count: NDArray[np.int_] = field(init=False, repr=False)
    """Count of rows in node from training."""

    feature_idx: NDArray[np.int_] = field(init=False, repr=False)
    """Feature index for split."""

    num_threshold: NDArray[np.float64] = field(init=False, repr=False)
    """Split threshold."""

    missing_go_to_left: NDArray[np.int_] = field(init=False, repr=False)
    """Binary indicator if null values go to the left child."""

    left: NDArray[np.int_] = field(init=False, repr=False)
    """Lift child index."""

    right: NDArray[np.int_] = field(init=False, repr=False)
    """Right child index."""

    gain: NDArray[np.float64] = field(init=False, repr=False)
    """Gain for split."""

    depth: NDArray[np.int_] = field(init=False, repr=False)
    """Depth of node."""

    is_leaf: NDArray[np.int_] = field(init=False, repr=False)
    """Leaf node indicator."""

    bin_threshold: NDArray[np.int_] = field(init=False, repr=False)
    is_categorical: NDArray[np.int_] = field(init=False, repr=False)
    bitset_idx: NDArray[np.int_] = field(init=False, repr=False)

    @classmethod
    def from_hist_gradient_booster(  # type: ignore[no-any-unimported]
        cls, model: Union[HistGradientBoostingClassifier, HistGradientBoostingRegressor]
    ) -> "ScikitLearnHistTabularTrees":
        """Create ScikitLearnHistTabularTrees from hist gradient booster.

        Parameters
        ----------
        model : Union[HistGradientBoostingClassifier, HistGradientBoostingRegressor]
            Model to extract tree data from.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.ensemble import HistGradientBoostingRegressor
        >>> from tabular_trees import ScikitLearnHistTabularTrees
        >>> # load data
        >>> diabetes = load_diabetes()
        >>> # build model
        >>> model = HistGradientBoostingRegressor(max_depth=3, max_iter=10)
        >>> model.fit(diabetes["data"], diabetes["target"])
        HistGradientBoostingRegressor(max_depth=3, max_iter=10)
        >>> # export to ScikitLearnHistTabularTrees
        >>> sklearn_tabular_trees = ScikitLearnHistTabularTrees.from_hist_gradient_booster(model)
        >>> type(sklearn_tabular_trees)
        <class 'tabular_trees.sklearn.sklearn_hist_tabular_trees.ScikitLearnHistTabularTrees'>

        """  # noqa: E501
        checks.check_type(
            model,
            (HistGradientBoostingClassifier, HistGradientBoostingRegressor),
            "model",
        )

        if not model._is_fitted():
            raise ValueError("model is not fitted, cannot export trees")

        if len(model._predictors[0]) > 1:
            raise NotImplementedError("model with multiple responses not supported")

        tree_data = ScikitLearnHistTabularTrees._extract_hist_gbm_tree_data(model)

        return ScikitLearnHistTabularTrees(tree_data)

    @staticmethod
    def _extract_hist_gbm_tree_data(  # type: ignore[no-any-unimported]
        model: Union[HistGradientBoostingClassifier, HistGradientBoostingRegressor],
    ) -> pd.DataFrame:
        """Extract tree data from HistGradientBoosting model.

        Tree data is pulled from _predictors attributes in
        HistGradientBoostingClassifier or HistGradientBoostingRegressor object.

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

        starting_value = (
            ScikitLearnHistTabularTrees._get_starting_value_hist_gradient_booster(model)
        )
        tree_data.loc[tree_data["tree"] == 0, "value"] = (
            tree_data.loc[tree_data["tree"] == 0, "value"] + starting_value
        )

        return tree_data

    @staticmethod
    def _get_starting_value_hist_gradient_booster(  # type: ignore[no-any-unimported]
        model: Union[HistGradientBoostingClassifier, HistGradientBoostingRegressor],
    ) -> Union[int, float]:
        """Extract the initial prediction for the ensemble."""
        return model._baseline_prediction[0][0]  # type: ignore[no-any-return]


@export_tree_data.register(HistGradientBoostingClassifier)
@export_tree_data.register(HistGradientBoostingRegressor)
def _export_tree_data__hist_gradient_boosting_model(  # type: ignore[no-any-unimported]
    model: Union[HistGradientBoostingClassifier, HistGradientBoostingRegressor],
) -> ScikitLearnHistTabularTrees:
    """Export tree data from HistGradientBoostingRegressor or Classifier object.

    Parameters
    ----------
    model : Union[HistGradientBoostingClassifier, HistGradientBoostingRegressor]
        Model to export tree data from.

    """
    return ScikitLearnHistTabularTrees.from_hist_gradient_booster(model)
