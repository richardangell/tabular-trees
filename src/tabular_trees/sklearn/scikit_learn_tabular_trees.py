"""Scikit-learn GBM trees in tabular format."""

from dataclasses import dataclass
from typing import Union

import pandas as pd

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

    trees: pd.DataFrame
    """Tree data."""

    REQUIRED_COLUMNS = [
        "tree",
        "node",
        "children_left",
        "children_right",
        "feature",
        "impurity",
        "n_node_samples",
        "threshold",
        "value",
        "weighted_n_node_samples",
    ]
    """List of columns required in tree data."""

    SORT_BY_COLUMNS = ["tree", "node"]
    """List of columns to sort tree data by."""

    def __init__(self, trees: pd.DataFrame):
        """Initialise the ScikitLearnTabularTrees object.

        Parameters
        ----------
        trees : pd.DataFrame
            GradientBoostingRegressor or Classifier tree data extracted from
            the .estimators_ attribute.

        Examples
        --------
        >>> from sklearn.datasets import load_diabetes
        >>> from sklearn.ensemble import GradientBoostingRegressor
        >>> from tabular_trees import export_tree_data
        >>> # load data
        >>> diabetes = load_diabetes()
        >>> # build model
        >>> model = GradientBoostingRegressor(max_depth=3, n_estimators=10)
        >>> model.fit(diabetes["data"], diabetes["target"])
        GradientBoostingRegressor(n_estimators=10)
        >>> # export to ScikitLearnTabularTrees
        >>> sklearn_tabular_trees = export_tree_data(model)
        >>> type(sklearn_tabular_trees)
        <class 'tabular_trees.sklearn.scikit_learn_tabular_trees.ScikitLearnTabularTrees'>

        """  # noqa: E501
        self.trees = trees

        self.__post_init__()


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
    checks.check_type(
        model, (GradientBoostingClassifier, GradientBoostingRegressor), "model"
    )

    if not hasattr(model, "estimators_"):
        raise ValueError("model is not fitted, cannot export trees")

    if len(model.estimators_[0]) > 1:
        raise NotImplementedError("model with multiple responses not supported")

    tree_data = _extract_gbm_tree_data(model)

    return ScikitLearnTabularTrees(tree_data)


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
        tree_df = _extract_tree_data(model.estimators_[tree_no][0].tree_)

        tree_df["tree"] = tree_no

        tree_data_list.append(tree_df)

    tree_data = pd.concat(tree_data_list, axis=0)

    return tree_data


def _get_starting_value_gradient_booster(  # type: ignore[no-any-unimported]
    model: Union[GradientBoostingClassifier, GradientBoostingRegressor],
) -> Union[int, float]:
    """Extract the initial prediction for the ensemble."""
    return model.init_.constant_[0][0]  # type: ignore[no-any-return]


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
