"""Fixtures for Scikit-Learn specific data and models."""

import pytest
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier

from tabular_trees.sklearn.trees import (
    _extract_gbm_tree_data,
    _extract_hist_gbm_tree_data,
)


@pytest.fixture(scope="session")
def sklearn_iris_hist_gbc(iris_data):
    """Build an sklearn HistGradientBoostingClassifier on the iris dataset.

    Model has 10 trees and depth 3.

    """
    model = HistGradientBoostingClassifier(max_iter=10, max_depth=3)

    model.fit(iris_data["data"], iris_data["target"])

    return model


@pytest.fixture(scope="session")
def sklearn_iris_gbc(iris_data):
    """Build an sklearn GradientBoostingClassifier on the iris dataset.

    Model has 10 trees and depth 3.

    """
    model = GradientBoostingClassifier(n_estimators=10, max_depth=3)

    model.fit(iris_data["data"], iris_data["target"])

    return model


@pytest.fixture(scope="session")
def sklearn_hist_gbm_trees_dataframe(sklearn_diabetes_hist_gbr):
    """Return the trees from a HistGradientBoostingRegressor in DataFrame structure."""
    return _extract_hist_gbm_tree_data(sklearn_diabetes_hist_gbr)


@pytest.fixture(scope="session")
def sklearn_gbm_trees_dataframe(sklearn_diabetes_gbr):
    """Return the trees from a GradientBoostingRegressor in DataFrame structure."""
    return _extract_gbm_tree_data(sklearn_diabetes_gbr)
