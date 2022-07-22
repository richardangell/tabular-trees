import pandas as pd
import pytest
from sklearn.datasets import load_diabetes, load_iris


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
