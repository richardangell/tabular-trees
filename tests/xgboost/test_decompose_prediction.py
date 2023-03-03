import numpy as np
import pandas as pd
import pytest
from eli5 import explain_prediction_df

from tabular_trees.explain import decompose_prediction
from tabular_trees.trees import export_tree_data


@pytest.mark.parametrize("row_number_to_score", [(0), (1), (2), (20), (100), (222)])
def test_prediction_decomposition_eli5_equality(
    row_number_to_score, diabetes_data, xgb_diabetes_model_lambda_0
):
    """Test xgboost.explainer.decompose_prediction and eli5.explain_prediction_df give
    the same results."""

    # get row of diabetes data to score
    diabetes_data_df = pd.DataFrame(
        diabetes_data["data"], columns=diabetes_data["feature_names"]
    )
    row_data = diabetes_data_df.iloc[[row_number_to_score]]

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
    tabular_trees_decomposition = decompose_prediction(
        tabular_trees=tabular_trees,
        row=row_data,
        calculate_root_node=tabular_trees.get_root_node_given_tree,
    )

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
    np.testing.assert_almost_equal(
        decomposition_compare_df["weight"].values,
        decomposition_compare_df["contribution"].values,
        decimal=9,
    )
