import pandas as pd
import pytest

from tabular_trees.explain.shapley_values import calculate_shapley_values
from tabular_trees.trees import export_tree_data
from tabular_trees.xgboost.xgboost_tabular_trees import XGBoostTabularTrees


@pytest.mark.parametrize("row_number_to_score", [0, 1, 11, 22, 199, 201])
def test_equality_with_xgboost_treeshap(
    row_number_to_score,
    diabetes_data_subset_cols,
    xgb_diabetes_dmatrix_subset_cols,
    xgb_diabetes_model_subset_cols,
):
    """Test equality between treeshap from xgboost and calculate_shapley_values."""
    # xgboost treeshap implementation
    shapley_values_xgboost = xgb_diabetes_model_subset_cols.predict(
        xgb_diabetes_dmatrix_subset_cols, pred_contribs=True
    )

    shapley_values_xgboost_df = pd.DataFrame(
        shapley_values_xgboost,
        columns=xgb_diabetes_dmatrix_subset_cols.feature_names + ["bias"],
    )

    # get row of diabetes data to score
    diabetes_data_df = pd.DataFrame(
        diabetes_data_subset_cols["data"],
        columns=diabetes_data_subset_cols["feature_names"],
    )
    row_data = diabetes_data_df.iloc[row_number_to_score]

    xgboost_tabular_trees: XGBoostTabularTrees = export_tree_data(
        xgb_diabetes_model_subset_cols
    )
    tabular_trees = xgboost_tabular_trees.to_tabular_trees()

    shapley_values_tabular_trees = calculate_shapley_values(tabular_trees, row_data)

    shapley_values_tabular_trees = shapley_values_tabular_trees.summary[
        shapley_values_xgboost_df.columns.values
    ]

    pd.testing.assert_frame_equal(
        shapley_values_xgboost_df.iloc[[row_number_to_score]].reset_index(drop=True),
        shapley_values_tabular_trees.reset_index(drop=True),
        check_dtype=False,
    )
