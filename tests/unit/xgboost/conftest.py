"""Fixtures for XGBoost specific data and models."""

import pandas as pd
import pytest

from tabular_trees.xgboost.dump_reader import JsonDumpReader


@pytest.fixture(scope="session")
def xgb_diabetes_model_trees_dataframe(xgb_diabetes_model) -> pd.DataFrame:
    """Return the trees from xgb_diabetes_model in DataFrame structure."""
    return xgb_diabetes_model.trees_to_dataframe()


@pytest.fixture()
def xgb_diabetes_model_parsed_trees_dataframe(
    tmp_path, xgb_diabetes_model
) -> pd.DataFrame:
    """Return the trees from xgb_diabetes_model in DataFrame structure.

    Trees are dumped to json and then loaded from there so the data has different column
    names than if trees_to_dataframe was used.

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

    Trees are dumped to json and then loaded from there so the data has different column
    names than if trees_to_dataframe was used.

    """
    json_file = str(tmp_path / "model.json")

    xgb_diabetes_model.dump_model(json_file, with_stats=False, dump_format="json")

    trees_df = JsonDumpReader().read_dump(json_file)

    return trees_df
