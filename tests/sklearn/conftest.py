"""Fixtures for Scikit-Learn specific data and models."""

import pandas as pd
import pytest
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)

from tabular_trees.sklearn import _extract_gbm_tree_data, _extract_hist_gbm_tree_data


@pytest.fixture(scope="session")
def sklearn_diabetes_hist_gbm_regressor(diabetes_data) -> HistGradientBoostingRegressor:
    """Sklearn HistGradientBoostingRegressor built on the diabetes dataset.

    Model has 10 trees and depth 3.

    """
    model = HistGradientBoostingRegressor(max_iter=10, max_depth=3)

    model.fit(diabetes_data["data"], diabetes_data["target"])

    return model


@pytest.fixture(scope="session")
def sklearn_diabetes_gbm_regressor(diabetes_data) -> GradientBoostingRegressor:
    """Sklearn GradientBoostingRegressor on the diabetes dataset.

    Model has 10 trees and depth 3.

    """
    model = GradientBoostingRegressor(n_estimators=10, max_depth=3)

    model.fit(diabetes_data["data"], diabetes_data["target"])

    return model


@pytest.fixture(scope="session")
def sklearn_breast_cancer_hist_gbm_classifier(
    breast_cancer_data,
) -> HistGradientBoostingClassifier:
    """Sklearn HistGradientBoostingClassifier on the breast cancer dataset.

    Model has 10 trees and depth 3.

    """
    model = HistGradientBoostingClassifier(max_iter=10, max_depth=3)

    model.fit(breast_cancer_data["data"], breast_cancer_data["target"])

    return model


@pytest.fixture(scope="session")
def sklearn_breast_cancer_gbm_classifier(
    breast_cancer_data,
) -> GradientBoostingClassifier:
    """Sklearn GradientBoostingClassifier on the breast cancer dataset.

    Model has 10 trees and depth 3.

    """
    model = GradientBoostingClassifier(n_estimators=1, max_depth=1, learning_rate=0.1)

    model.fit(breast_cancer_data["data"], breast_cancer_data["target"])

    return model


@pytest.fixture(scope="session")
def sklearn_iris_hist_gbm_classifier(iris_data) -> HistGradientBoostingClassifier:
    """Build an sklearn HistGradientBoostingClassifier on the iris dataset.

    Model has 10 trees and depth 3.

    """
    model = HistGradientBoostingClassifier(max_iter=10, max_depth=3)

    model.fit(iris_data["data"], iris_data["target"])

    return model


@pytest.fixture(scope="session")
def sklearn_iris_gbm_classifier(iris_data) -> GradientBoostingClassifier:
    """Build an sklearn GradientBoostingClassifier on the iris dataset.

    Model has 10 trees and depth 3.

    """
    model = GradientBoostingClassifier(n_estimators=10, max_depth=3)

    model.fit(iris_data["data"], iris_data["target"])

    return model


@pytest.fixture(scope="session")
def sklearn_hist_gbm_trees_dataframe(
    sklearn_diabetes_hist_gbm_regressor,
) -> pd.DataFrame:
    """Return the trees from a HistGradientBoostingRegressor in DataFrame structure."""
    return _extract_hist_gbm_tree_data(sklearn_diabetes_hist_gbm_regressor)


@pytest.fixture(scope="session")
def sklearn_gbm_trees_dataframe(sklearn_diabetes_gbm_regressor) -> pd.DataFrame:
    """Return the trees from a GradientBoostingRegressor in DataFrame structure."""
    return _extract_gbm_tree_data(sklearn_diabetes_gbm_regressor)


@pytest.fixture
def handcrafted_data():
    """Handcraft dataset that can be expressed with a simple depth 2 tree.

    Data is as follows;
     a   b      response
     1   1      100
     1   1      100
     1  -1      150
     1  -1      150
    -1   1      200
    -1   1      200
    -1  -1      250
    -1  -1      250

    """
    data = pd.DataFrame(
        {
            "a": [1, 1, 1, 1, -1, -1, -1, -1],
            "b": [1, 1, -1, -1, 1, 1, -1, -1],
            "response": [100, 100, 150, 150, 200, 200, 250, 250],
        }
    )

    return data
