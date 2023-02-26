"""Fixtures for providing Scikit-Learn toy datasets."""

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris


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
