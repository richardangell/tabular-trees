import lightgbm as lgb
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def lgb_diabetes_dataset(diabetes_data) -> lgb.Dataset:
    """Return the diabetes dataset in a single lgb.Dataset."""
    lgb_data = lgb.Dataset(
        diabetes_data["data"],
        label=diabetes_data["target"],
        feature_name=diabetes_data["feature_names"],
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
