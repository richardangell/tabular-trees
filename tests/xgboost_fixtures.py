"""Fixtures for XGBoost specific data and models."""

import pytest
import xgboost as xgb


@pytest.fixture(scope="session")
def xgb_diabetes_dmatrix(diabetes_data) -> xgb.DMatrix:
    """Return the diabetes dataset in a single xgb.DMatrix."""
    xgb_data = xgb.DMatrix(
        diabetes_data["data"],
        label=diabetes_data["target"],
        feature_names=diabetes_data["feature_names"],
    )

    return xgb_data


@pytest.fixture(scope="session")
def xgb_diabetes_model(xgb_diabetes_dmatrix) -> xgb.Booster:
    """Xgboost model with 10 trees and depth 3 on the diabetes dataset."""
    model = xgb.train(
        params={"verbosity": 0, "max_depth": 3},
        dtrain=xgb_diabetes_dmatrix,
        num_boost_round=10,
    )

    return model


@pytest.fixture(scope="session")
def xgb_diabetes_model_non_zero_alpha(xgb_diabetes_dmatrix) -> xgb.Booster:
    """Xgboost model with alpha 1 on the diabetes dataset.

    Model has with 2 trees at depth 3.

    """
    model = xgb.train(
        params={"verbosity": 0, "max_depth": 3, "alpha": 1},
        dtrain=xgb_diabetes_dmatrix,
        num_boost_round=2,
    )

    return model


@pytest.fixture(scope="session")
def xgb_diabetes_dmatrix_subset_cols(diabetes_data_subset_cols) -> xgb.DMatrix:
    """Return a subset (4 columns) of the diabetes dataset in a single xgb.DMatrix.

    Note, the base_margin is set to 0 for this DMatrix.

    """
    xgb_data = xgb.DMatrix(
        diabetes_data_subset_cols["data"],
        label=diabetes_data_subset_cols["target"],
        base_margin=[0] * diabetes_data_subset_cols["data"].shape[0],
        feature_names=diabetes_data_subset_cols["feature_names"],
    )

    return xgb_data


@pytest.fixture(scope="session")
def xgb_diabetes_model_subset_cols(xgb_diabetes_dmatrix_subset_cols) -> xgb.Booster:
    """Xgboost model with 10 trees and depth 3 on 4 columns of the diabetes dataset."""
    model = xgb.train(
        params={"verbosity": 0, "max_depth": 3, "tree_method": "exact"},
        dtrain=xgb_diabetes_dmatrix_subset_cols,
        num_boost_round=10,
    )

    return model
