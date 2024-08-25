"""Scikit-learn GBM trees in tabular format."""

from dataclasses import dataclass, field
from typing import Union

import numpy as np
import pandas as pd
from numpy.typing import NDArray

try:
    from sklearn.ensemble import (  # type: ignore[import-not-found]
        GradientBoostingClassifier,
        GradientBoostingRegressor,
    )
    from sklearn.tree._tree import Tree  # type: ignore[import-not-found]
except ModuleNotFoundError as err:
    raise ImportError(
        "scikit-learn must be installed to use functionality in sklearn module"
    ) from err

from .. import checks
from ..trees import BaseModelTabularTrees, export_tree_data


@dataclass
class ScikitLearnTabularTrees(BaseModelTabularTrees):
    """Scikit-Learn GradientBoosting trees in tabular format."""

    data: pd.DataFrame
    """Tree data."""

    tree: NDArray[np.int_] = field(init=False, repr=False)
    """Tree index."""

    node: NDArray[np.int_] = field(init=False, repr=False)
    """Node index."""

    children_left: NDArray[np.int_] = field(init=False, repr=False)
    """Left child node index."""

    children_right: NDArray[np.int_] = field(init=False, repr=False)
    """Right child node index."""

    feature: NDArray[np.int_] = field(init=False, repr=False)
    """Split feature index."""

    impurity: NDArray[np.float64] = field(init=False, repr=False)
    """Impurity at node."""

    n_node_samples: NDArray[np.int_] = field(init=False, repr=False)
    """Number of records at node."""

    threshold: NDArray[np.float64] = field(init=False, repr=False)
    """Split threshold."""

    value: NDArray[np.float64] = field(init=False, repr=False)
    """Split value."""

    weighted_n_node_samples: NDArray[np.float64] = field(init=False, repr=False)
    """Weight at node."""

    @classmethod
    def from_gradient_booster(  # type: ignore[no-any-unimported]
        cls, model: Union[GradientBoostingClassifier, GradientBoostingRegressor]
    ) -> "ScikitLearnTabularTrees":
        """Export from a GradientBoostingClassifier or GradientBoostingRegressor.

        Parameters
        ----------
        model : Union[GradientBoostingClassifier, GradientBoostingRegressor]
            GradientBoostingRegressor or Classifier tree data extracted from
            the .estimators_ attribute.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> from tabular_trees import ScikitLearnTabularTrees
        >>> # load data
        >>> diabetes = load_diabetes()
        >>> # build model
        >>> model = GradientBoostingRegressor(max_depth=3, n_estimators=10)
        >>> model.fit(diabetes["data"], diabetes["target"])
        GradientBoostingRegressor(n_estimators=10)
        >>> # export to ScikitLearnTabularTrees
        >>> sklearn_tabular_trees = ScikitLearnTabularTrees.from_gradient_booster(model)
        >>> type(sklearn_tabular_trees)
        <class 'tabular_trees.sklearn.sklearn_tabular_trees.ScikitLearnTabularTrees'>

        """  # noqa: E501
        checks.check_type(
            model, (GradientBoostingClassifier, GradientBoostingRegressor), "model"
        )

        if not hasattr(model, "estimators_"):
            raise ValueError("model is not fitted, cannot export trees")

        if len(model.estimators_[0]) > 1:
            raise NotImplementedError("model with multiple responses not supported")

        tree_data = ScikitLearnTabularTrees._extract_gbm_tree_data(model)

        return ScikitLearnTabularTrees(tree_data)

    @staticmethod
    def _extract_gbm_tree_data(  # type: ignore[no-any-unimported]
        model: Union[GradientBoostingClassifier, GradientBoostingRegressor],
    ) -> pd.DataFrame:
        """Extract tree data from GradientBoosting model.

        Tree data is extracted from estimators_ objects on either
        GradientBoostingClassifier or GradientBoostingRegressor.

        Parameters
        ----------
        model : Union[GradientBoostingClassifier, GradientBoostingRegressor]
            Model to extract tree data from.

        """
        tree_data_list = []

        for tree_no in range(model.n_estimators_):
            tree_df = ScikitLearnTabularTrees._extract_tree_data(
                model.estimators_[tree_no][0].tree_
            )

            tree_df["tree"] = tree_no

            tree_data_list.append(tree_df)

        tree_data = pd.concat(tree_data_list, axis=0)

        return tree_data

    @staticmethod
    def _extract_tree_data(tree: Tree) -> pd.DataFrame:  # type: ignore[no-any-unimported]
        """Extract node data from a sklearn.tree._tree.Tree object.

        Parameters
        ----------
        tree : Tree
            Tree object to extract data from.

        """
        tree_data = pd.DataFrame(
            {
                "children_left": tree.children_left,
                "children_right": tree.children_right,
                "feature": tree.feature,
                "impurity": tree.impurity,
                "n_node_samples": tree.n_node_samples,
                "threshold": tree.threshold,
                "value": tree.value.ravel(),
                "weighted_n_node_samples": tree.weighted_n_node_samples,
            }
        )

        tree_data = tree_data.reset_index().rename(columns={"index": "node"})

        return tree_data


@export_tree_data.register(GradientBoostingClassifier)
@export_tree_data.register(GradientBoostingRegressor)
def _export_tree_data__gradient_boosting_model(  # type: ignore[no-any-unimported]
    model: Union[GradientBoostingClassifier, GradientBoostingRegressor],
) -> ScikitLearnTabularTrees:
    """Export tree data from GradientBoostingRegressor or Classifier object.

    Parameters
    ----------
    model : Union[GradientBoostingClassifier, GradientBoostingRegressor]
        Model to export tree data from.

    """
    return ScikitLearnTabularTrees.from_gradient_booster(model)
