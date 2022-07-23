from sklearn.ensemble import HistGradientBoostingClassifier
from tabular_trees.sklearn.trees import _extract_hist_gbm_tree_data

import pytest


@pytest.fixture(scope="session")
def sklearn_diabetes_model_trees_dataframe(sklearn_diabetes_model):
    """Return the trees from sklearn_diabetes_model in DataFrame structure."""

    return _extract_hist_gbm_tree_data(sklearn_diabetes_model)


@pytest.fixture(scope="session")
def sklearn_iris_model(iris_data):
    """Build an sklearn HistGradientBoostingClassifier with 10 trees and depth
    3 on the iris dataset."""

    model = HistGradientBoostingClassifier(max_iter=10, max_depth=3)

    model.fit(iris_data["data"], iris_data["target"])

    return model
