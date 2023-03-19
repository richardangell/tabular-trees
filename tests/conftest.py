"""Fixtures for providing Scikit-Learn and other toy datasets."""

import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.utils import Bunch


@pytest.fixture(scope="session")
def diabetes_data() -> Bunch:
    """Load the sklearn diabetes dataset."""
    data = load_diabetes()
    return data


@pytest.fixture(scope="session")
def diabetes_data_subset_cols(diabetes_data) -> Bunch:
    """First 4 column from the sklearn diabetes dataset."""
    subset_cols = 4

    subset_cols_bunch = Bunch()
    subset_cols_bunch.update(diabetes_data)

    reduced_data = {
        "data": diabetes_data["data"][:, :subset_cols],
        "feature_names": diabetes_data["feature_names"][:subset_cols],
    }

    subset_cols_bunch.update(reduced_data)

    return subset_cols_bunch


@pytest.fixture(scope="session")
def iris_data() -> Bunch:
    """Load the sklearn iris dataset."""
    data = load_iris()
    return data


@pytest.fixture(scope="session")
def breast_cancer_data() -> Bunch:
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


@pytest.fixture(scope="session")
def two_way_monotonic_increase_x2() -> pd.DataFrame:
    """2 way interaction that is monotonically increasing across both factors.

    Data is as follows:
    a	b	response
    -1	-1	130
    -1	-1	130
    -1	1	180
    -1	1	180
    1	-1	380
    1	-1	380
    1	1	420
    1	1	420

    """
    data = pd.DataFrame(
        {
            "a": [-1, -1, -1, -1, 1, 1, 1, 1],
            "b": [-1, -1, 1, 1, -1, -1, 1, 1],
            "response": [130, 130, 180, 180, 380, 380, 420, 420],
        }
    )

    return data


@pytest.fixture(scope="session")
def two_way_monotonic_increase_decrease() -> pd.DataFrame:
    """2 way interaction where factors are monotonically increasing then decreasing.

    Data is as follows:
    a	b	response
    -1	-1	180
    -1	-1	180
    -1	1	130
    -1	1	130
    1	-1	420
    1	-1	420
    1	1	380
    1	1	380

    """
    data = pd.DataFrame(
        {
            "a": [-1, -1, -1, -1, 1, 1, 1, 1],
            "b": [-1, -1, 1, 1, -1, -1, 1, 1],
            "response": [180, 180, 130, 130, 420, 420, 380, 380],
        }
    )

    return data


@pytest.fixture(scope="session")
def two_way_monotonic_decrease_increase() -> pd.DataFrame:
    """2 way interaction where factors are monotonically decreasing then increasing.

    Data is as follows:
    a	b	response
    -1	-1	380
    -1	-1	380
    -1	1	420
    -1	1	420
    1	-1	130
    1	-1	130
    1	1	180
    1	1	180

    """
    data = pd.DataFrame(
        {
            "a": [-1, -1, -1, -1, 1, 1, 1, 1],
            "b": [-1, -1, 1, 1, -1, -1, 1, 1],
            "response": [380, 380, 420, 420, 130, 130, 180, 180],
        }
    )

    return data


@pytest.fixture(scope="session")
def two_way_monotonic_decrease_x2() -> pd.DataFrame:
    """2 way interaction that is monotonically decreasing for boths factors.

    Data is as follows:
    a	b	response
    -1	-1	420
    -1	-1	420
    -1	1	380
    -1	1	380
    1	-1	180
    1	-1	180
    1	1	130
    1	1	130

    """
    data = pd.DataFrame(
        {
            "a": [-1, -1, -1, -1, 1, 1, 1, 1],
            "b": [-1, -1, 1, 1, -1, -1, 1, 1],
            "response": [420, 420, 380, 380, 180, 180, 130, 130],
        }
    )

    return data
