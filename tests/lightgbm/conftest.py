"""Fixtures for LightGBM specific data and models."""

import lightgbm as lgb
import pandas as pd
import pytest
from sklearn.datasets import load_diabetes


@pytest.fixture(scope="session")
def lgb_diabetes_dataset() -> lgb.Dataset:
    """Return the diabetes dataset in a single lgb.Dataset."""
    data = load_diabetes()

    lgb_data = lgb.Dataset(
        data["data"], label=data["target"], feature_name=data["feature_names"]
    )

    return lgb_data


@pytest.fixture(scope="session")
def lgb_diabetes_model(lgb_diabetes_dataset) -> lgb.Booster:
    """Return a lightgbm model built on the diabetes dataset.

    Note, model has 10 trees and depth 3.

    """
    model = lgb.train(
        params={"verbosity": -1, "max_depth": 3},
        train_set=lgb_diabetes_dataset,
        num_boost_round=10,
    )

    return model


@pytest.fixture(scope="session")
def lgb_diabetes_model_trees_dataframe(lgb_diabetes_model) -> pd.DataFrame:
    """Return the trees from lgb_diabetes_model in DataFrame structure."""
    return lgb_diabetes_model.trees_to_dataframe()
