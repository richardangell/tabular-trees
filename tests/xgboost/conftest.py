import xgboost as xgb
from sklearn.datasets import load_diabetes

import pytest


@pytest.fixture(scope="session")
def xgb_diabetes_model():
    """Build an xgboost model with 10 trees and depth 3 on the diabetes dataset."""

    data = load_diabetes()

    xgb_data = xgb.DMatrix(
        data["data"], label=data["target"], feature_names=data["feature_names"]
    )

    model = xgb.train(
        params={"silent": 1, "max_depth": 3}, dtrain=xgb_data, num_boost_round=10
    )

    return model


@pytest.fixture(scope="session")
def xgb_diabetes_model_trees_dataframe(xgb_diabetes_model):
    """Return the trees from xgb_diabetes_model in DataFrame structure."""

    return xgb_diabetes_model.trees_to_dataframe()
