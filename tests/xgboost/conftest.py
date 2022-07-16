import xgboost as xgb
from sklearn.datasets import load_diabetes

from tabular_trees.xgboost.parser import JsonDumpReader

import pytest


@pytest.fixture(scope="session")
def xgb_diabetes_dmatrix():
    """Return the diabetes dataset in a single xgb.DMatrix."""

    data = load_diabetes()

    xgb_data = xgb.DMatrix(
        data["data"], label=data["target"], feature_names=data["feature_names"]
    )

    return xgb_data


@pytest.fixture(scope="session")
def xgb_diabetes_model(xgb_diabetes_dmatrix):
    """Build an xgboost model with 10 trees and depth 3 on the diabetes dataset."""

    model = xgb.train(
        params={"silent": 1, "max_depth": 3, "lambda": 0},
        dtrain=xgb_diabetes_dmatrix,
        num_boost_round=10,
    )

    return model


@pytest.fixture(scope="session")
def xgb_diabetes_model_trees_dataframe(xgb_diabetes_model):
    """Return the trees from xgb_diabetes_model in DataFrame structure."""

    return xgb_diabetes_model.trees_to_dataframe()


@pytest.fixture()
def xgb_diabetes_model_parsed_trees_dataframe(tmp_path, xgb_diabetes_model):
    """Return the trees from xgb_diabetes_model in DataFrame structure.

    Trees are dumped to json and then loaded from there so the data
    has different column names than if trees_to_dataframe was used.
    """

    json_file = str(tmp_path / "model.json")

    xgb_diabetes_model.dump_model(json_file, with_stats=True, dump_format="json")

    trees_df = JsonDumpReader().read_dump(json_file)

    return trees_df


@pytest.fixture()
def xgb_diabetes_model_parsed_trees_dataframe_no_stats(tmp_path, xgb_diabetes_model):
    """Return the trees from xgb_diabetes_model in DataFrame structure
    without gain and cover stats.

    Trees are dumped to json and then loaded from there so the data
    has different column names than if trees_to_dataframe was used.
    """

    json_file = str(tmp_path / "model.json")

    xgb_diabetes_model.dump_model(json_file, with_stats=False, dump_format="json")

    trees_df = JsonDumpReader().read_dump(json_file)

    return trees_df
