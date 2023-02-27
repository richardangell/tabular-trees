import pandas as pd
import pytest
from eli5 import explain_prediction_df

from tabular_trees.trees import export_tree_data
from tabular_trees.xgboost.explainer import decompose_prediction


def test_prediction_decomposition_eli5_equality(
    diabetes_data, xgb_diabetes_model_lambda_0
):
    """Test xgboost.explainer.decompose_prediction and eli5.explain_prediction_df give
    the same results."""

    # get row of diabetes data to score
    diabetes_data_df = pd.DataFrame(
        diabetes_data["data"], columns=diabetes_data["feature_names"]
    )
    row_data = diabetes_data_df.iloc[[0]]

    # get decomposition with eli5
    eli5_decomposition = explain_prediction_df(xgb_diabetes_model_lambda_0, row_data)

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
    tabular_trees_decomposition = decompose_prediction(tabular_trees.trees, row_data)

    # aggregate tabular_trees output to variable level, by default it is at tree x node level
    tabular_trees_decomposition_agg = pd.DataFrame(
        tabular_trees_decomposition.groupby("contributing_var").contribution.sum()
    ).reset_index()

    # merge eli5 and tabular_trees values
    decomposition_compare_df = tabular_trees_decomposition_agg.merge(
        eli5_decomposition[["feature_mapped", "weight"]],
        how="left",
        left_on="contributing_var",
        right_on="feature_mapped",
        indicator=True,
    )

    # check merge is 1:1 i.e. both have same variables
    if (
        decomposition_compare_df["_merge"] == "both"
    ).sum() < decomposition_compare_df.shape[0]:

        pytest.fail(
            f"different features in eli5 and tabular_trees (merge not 1:1)\n\n{decomposition_compare_df}"
        )

    # check equality between prediction decomposition values
    pd.testing.assert_series_equal(
        left=decomposition_compare_df["weight"],
        right=decomposition_compare_df["contribution"],
        check_names=False,
        check_exact=False,
    )
