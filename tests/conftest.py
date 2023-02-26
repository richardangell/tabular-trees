"""Fixtures for Scikit-Learn  specific data and (GBM) models."""

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)


@pytest.fixture(scope="session")
def diabetes_data() -> pd.DataFrame:
    """Load the sklearn diabetes dataset."""
    data = load_diabetes()

    return data


@pytest.fixture(scope="session")
def iris_data() -> pd.DataFrame:
    """Load the sklearn iris dataset."""
    data = load_iris()

    return data


@pytest.fixture(scope="session")
def breast_cancer_data() -> pd.DataFrame:
    """Load the sklearn breast cancer dataset."""
    data = load_breast_cancer()

    return data


@pytest.fixture(scope="session")
def sklearn_diabetes_hist_gbr(diabetes_data):
    """Sklearn HistGradientBoostingRegressor built on the diabetes dataset.

    Model has 10 trees and depth 3.

    """
    model = HistGradientBoostingRegressor(max_iter=10, max_depth=3)

    model.fit(diabetes_data["data"], diabetes_data["target"])

    return model


@pytest.fixture(scope="session")
def sklearn_diabetes_gbr(diabetes_data):
    """Sklearn GradientBoostingRegressor on the diabetes dataset.

    Model has 10 trees and depth 3.

    """
    model = GradientBoostingRegressor(n_estimators=10, max_depth=3)

    model.fit(diabetes_data["data"], diabetes_data["target"])

    return model


@pytest.fixture(scope="session")
def sklearn_breast_cancer_hist_gbc(breast_cancer_data):
    """Sklearn HistGradientBoostingClassifier on the breast cancer dataset.

    Model has 10 trees and depth 3.

    """
    model = HistGradientBoostingClassifier(max_iter=10, max_depth=3)

    model.fit(breast_cancer_data["data"], breast_cancer_data["target"])

    return model


@pytest.fixture(scope="session")
def sklearn_breast_cancer_gbc(breast_cancer_data):
    """Sklearn GradientBoostingClassifier on the breast cancer dataset.

    Model has 10 trees and depth 3.

    """
    model = GradientBoostingClassifier(n_estimators=10, max_depth=3)

    model.fit(breast_cancer_data["data"], breast_cancer_data["target"])

    return model
