import re

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from tabular_trees.explain.prediction_decomposition import (
    PredictionDecomposition,
    decompose_prediction,
)
from tabular_trees.trees import export_tree_data


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


def test_output_type(diabetes_data, xgb_diabetes_model_lambda_0):
    """Test the output of decompose_prediction is an PredictionDecomposition type."""
    diabetes_data_df = pd.DataFrame(
        diabetes_data["data"], columns=diabetes_data["feature_names"]
    )
    row_data = diabetes_data_df.iloc[[0]]

    xgboost_tabular_trees = export_tree_data(xgb_diabetes_model_lambda_0)
    tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()

    result = decompose_prediction(
        tabular_trees=tabular_trees,
        row=row_data,
    )

    assert (
        type(result) is PredictionDecomposition
    ), "incorrect type returned from decompose_prediction"


def test_tabular_trees_not_tabular_trees_exception(diabetes_data):
    """Test an error is raised if tabular_trees is not a TabularTrees object."""
    diabetes_data_df = pd.DataFrame(
        diabetes_data["data"], columns=diabetes_data["feature_names"]
    )
    row_data = diabetes_data_df.iloc[[0]]

    with pytest.raises(
        TypeError,
        match=(
            "tabular_trees is not in expected types "
            "<class 'tabular_trees.trees.TabularTrees'>"
        ),
    ):
        decompose_prediction(
            tabular_trees="abcde",
            row=row_data,
        )


def test_row_not_dataframe_exception(xgb_diabetes_model_lambda_0):
    """Test an exception is raised if row argument is not a pd.DataFrame."""
    xgboost_tabular_trees = export_tree_data(xgb_diabetes_model_lambda_0)
    tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()

    with pytest.raises(
        TypeError,
        match="row is not in expected types <class 'pandas.core.frame.DataFrame'>",
    ):
        decompose_prediction(
            tabular_trees=tabular_trees,
            row=123,
        )


@pytest.mark.parametrize("nrows", [0, 2])
def test_not_single_row_exception(nrows, diabetes_data, xgb_diabetes_model_lambda_0):
    """Test an exception is raised if row argument is not a single row."""
    diabetes_data_df = pd.DataFrame(
        diabetes_data["data"], columns=diabetes_data["feature_names"]
    )
    row_data = diabetes_data_df.iloc[0:nrows]

    xgboost_tabular_trees = export_tree_data(xgb_diabetes_model_lambda_0)
    tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()

    with pytest.raises(
        ValueError,
        match=re.escape("condition: [row is a single pd.DataFrame row] not met"),
    ):
        decompose_prediction(
            tabular_trees=tabular_trees,
            row=row_data,
        )


@pytest.mark.xfail(
    reason=(
        "explain_prediction_df from eli5 no longer used as it is "
        "incompatible with sklearn 1.5.0"
    )
)
@pytest.mark.parametrize("row_number_to_score", [(0), (1), (2), (20), (100), (222)])
def test_prediction_decomposition_eli5_equality(
    row_number_to_score, diabetes_data, xgb_diabetes_model_lambda_0
):
    """Test decompose_prediction and eli5.explain_prediction_df give same results."""
    # get row of diabetes data to score
    diabetes_data_df = pd.DataFrame(
        diabetes_data["data"], columns=diabetes_data["feature_names"]
    )
    row_data = diabetes_data_df.iloc[[row_number_to_score]]

    # get decomposition with eli5
    eli5_decomposition = explain_prediction_df(xgb_diabetes_model_lambda_0, row_data)  # noqa: F821

    # create mapping because eli5 output will have feature names x0, x1 etc..
    column_mapping = {"<BIAS>": "base"}
    for i, x in enumerate(diabetes_data["feature_names"]):
        column_mapping[f"x{i}"] = x

    eli5_decomposition["feature_mapped"] = eli5_decomposition["feature"].map(
        column_mapping
    )

    # get decomposition with tabular_trees
    xgboost_tabular_trees = export_tree_data(xgb_diabetes_model_lambda_0)
    tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()
    tabular_trees_decomposition = decompose_prediction(
        tabular_trees=tabular_trees,
        row=row_data,
    )

    # merge eli5 and tabular_trees values
    decomposition_compare_df = tabular_trees_decomposition.summary.merge(
        eli5_decomposition[["feature_mapped", "weight"]],
        how="left",
        left_on="contributing_feature",
        right_on="feature_mapped",
        indicator=True,
    )

    # check merge is 1:1 i.e. both have same variables
    if (
        decomposition_compare_df["_merge"] == "both"
    ).sum() < decomposition_compare_df.shape[0]:
        pytest.fail(
            "different features in eli5 and tabular_trees (merge not 1:1)\n\n"
            f"{decomposition_compare_df}"
        )

    # check equality between prediction decomposition values
    np.testing.assert_almost_equal(
        decomposition_compare_df["weight"].values,
        decomposition_compare_df["contribution"].values,
        decimal=9,
    )
