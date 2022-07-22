import pandas as pd
import pytest
from sklearn.datasets import load_diabetes


@pytest.fixture(scope="session")
def diabetes_data() -> pd.DataFrame:
    """Load the sklearn diabetes dataset."""

    data = load_diabetes()

    return data
