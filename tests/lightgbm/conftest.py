import lightgbm as lgb
from sklearn.datasets import load_diabetes

import pytest


@pytest.fixture(scope="session")
def lgb_diabetes_dataset():
    """Return the diabetes dataset in a single lgb.Dataset."""

    data = load_diabetes()

    lgb_data = lgb.Dataset(
        data["data"], label=data["target"], feature_name=data["feature_names"]
    )

    return lgb_data


@pytest.fixture(scope="session")
def lgb_diabetes_model(lgb_diabetes_dataset):
    """Build an lightgbm model with 10 trees and depth 3 on the diabetes
    dataset."""

    model = lgb.train(
        params={"verbosity": -1, "max_depth": 3},
        train_set=lgb_diabetes_dataset,
        num_boost_round=10,
    )

    return model


@pytest.fixture(scope="session")
def lgb_diabetes_model_trees_dataframe(lgb_diabetes_model):
    """Return the trees from lgb_diabetes_model in DataFrame structure."""

    return lgb_diabetes_model.trees_to_dataframe()
