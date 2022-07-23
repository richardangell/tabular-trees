import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris, load_breast_cancer
from sklearn.ensemble import (
    HistGradientBoostingRegressor,
    HistGradientBoostingClassifier,
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
def sklearn_diabetes_model(diabetes_data):
    """Build an sklearn HistGradientBoostingRegressor with 10 trees and depth
    3 on the diabetes dataset."""

    model = HistGradientBoostingRegressor(max_iter=10, max_depth=3)

    model.fit(diabetes_data["data"], diabetes_data["target"])

    return model


@pytest.fixture(scope="session")
def sklearn_breast_cancer_model(breast_cancer_data):
    """Build an sklearn HistGradientBoostingClassifier with 10 trees and depth
    3 on the breast cancer dataset."""

    model = HistGradientBoostingClassifier(max_iter=10, max_depth=3)

    model.fit(breast_cancer_data["data"], breast_cancer_data["target"])

    return model
