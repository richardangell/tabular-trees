import pytest
from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.utils import Bunch

from .lightgbm_fixtures import (  # noqa: F401
    lgb_diabetes_dataset,
    lgb_diabetes_model,
    lgb_diabetes_model_trees_dataframe,
)
from .xgboost_fixtures import (  # noqa: F401
    xgb_diabetes_dmatrix,
    xgb_diabetes_dmatrix_subset_cols,
    xgb_diabetes_model,
    xgb_diabetes_model_non_zero_alpha,
    xgb_diabetes_model_subset_cols,
)


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
