"""Fixtures for XGBoost specific data and models."""

import pandas as pd
import pytest
import xgboost as xgb

from tabular_trees.xgboost import JsonDumpReader


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
def xgb_diabetes_model(xgb_diabetes_dmatrix) -> xgb.Booster:
    """Xgboost model with 10 trees and depth 3 on the diabetes dataset."""
    model = xgb.train(
        params={"verbosity": 0, "max_depth": 3},
        dtrain=xgb_diabetes_dmatrix,
        num_boost_round=10,
    )

    return model


@pytest.fixture(scope="session")
def xgb_diabetes_model_subset_cols(xgb_diabetes_dmatrix_subset_cols) -> xgb.Booster:
    """Xgboost model with 10 trees and depth 3 on 4 columns of the diabetes dataset."""
    model = xgb.train(
        params={"verbosity": 0, "max_depth": 3},
        dtrain=xgb_diabetes_dmatrix_subset_cols,
        num_boost_round=10,
    )

    return model


@pytest.fixture(scope="session")
def xgb_diabetes_model_lambda_0(xgb_diabetes_dmatrix) -> xgb.Booster:
    """Xgboost model with 10 trees and depth 3 on the diabetes dataset.

    Other parameters; lambda = 0.

    """
    model = xgb.train(
        params={"verbosity": 0, "max_depth": 3, "lambda": 0},
        dtrain=xgb_diabetes_dmatrix,
        num_boost_round=10,
    )

    return model


@pytest.fixture(scope="session")
def xgb_diabetes_model_monotonic(xgb_diabetes_dmatrix) -> tuple[xgb.Booster, dict]:
    """Xgboost model with 10 trees and depth 3 on the diabetes dataset.

    Other parameters;
    - increasing monotonic constraint on bp and age.
    - decreasing monotonic constraint on bmi and s5.

    """
    feature_names = xgb_diabetes_dmatrix.feature_names

    monotonic_constraints = pd.Series([0] * len(feature_names), index=feature_names)
    monotonic_constraints.loc[monotonic_constraints.index.isin(["bmi", "s5"])] = -1
    monotonic_constraints.loc[monotonic_constraints.index.isin(["bp", "age"])] = 1

    monotonic_constraints_dict = monotonic_constraints.loc[
        monotonic_constraints != 0
    ].to_dict()

    model = xgb.train(
        params={
            "verbosity": 0,
            "max_depth": 3,
            "monotone_constraints": tuple(monotonic_constraints),
        },
        dtrain=xgb_diabetes_dmatrix,
        num_boost_round=10,
    )

    return model, monotonic_constraints_dict


@pytest.fixture(scope="session")
def xgb_diabetes_model_trees_dataframe(xgb_diabetes_model) -> pd.DataFrame:
    """Return the trees from xgb_diabetes_model in DataFrame structure."""
    return xgb_diabetes_model.trees_to_dataframe()


@pytest.fixture()
def xgb_diabetes_model_parsed_trees_dataframe(
    tmp_path, xgb_diabetes_model
) -> pd.DataFrame:
    """Return the trees from xgb_diabetes_model in DataFrame structure.

    Trees are dumped to json and then loaded from there so the data
    has different column names than if trees_to_dataframe was used.

    """
    json_file = str(tmp_path / "model.json")

    xgb_diabetes_model.dump_model(json_file, with_stats=True, dump_format="json")

    trees_df = JsonDumpReader().read_dump(json_file)

    return trees_df


@pytest.fixture()
def xgb_diabetes_model_parsed_trees_dataframe_no_stats(
    tmp_path, xgb_diabetes_model
) -> pd.DataFrame:
    """Trees from xgb_diabetes_model in DataFrame structure without gain and cover.

    Trees are dumped to json and then loaded from there so the data has different
    column names than if trees_to_dataframe was used.

    """
    json_file = str(tmp_path / "model.json")

    xgb_diabetes_model.dump_model(json_file, with_stats=False, dump_format="json")

    trees_df = JsonDumpReader().read_dump(json_file)

    return trees_df
