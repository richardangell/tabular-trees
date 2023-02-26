"""Fixtures for providing Scikit-Learn and other toy datasets."""

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


@pytest.fixture(scope="session")
def dummy_model_tree_data() -> pd.DataFrame:
    """Small dummy DataFrame with 3 columns and 4 rows."""
    dummy_model_tree_data = pd.DataFrame(
        {
            "column1": [4, 3, 2, 1],
            "column2": [5, 6, 7, 8],
            "column3": ["a", "b", "c", "d"],
        }
    )
    return dummy_model_tree_data
