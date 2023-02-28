import pandas as pd

from tabular_trees.trees import export_tree_data
from tabular_trees.xgboost.explainer import calculate_shapley_values


def test_shapley_values_treeshap_equality(
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
    row_data = diabetes_data_df.iloc[0]

    xgboost_tabular_trees = export_tree_data(xgb_diabetes_model_subset_cols)
    tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()

    shapley_values_tabular_trees = calculate_shapley_values(
        tabular_trees.trees, row_data, False
    )

    shapley_values_tabular_trees = shapley_values_tabular_trees[
        shapley_values_xgboost_df.columns.values
    ]

    pd.testing.assert_frame_equal(
        shapley_values_xgboost_df.iloc[[0]],
        shapley_values_tabular_trees,
        check_dtype=False,
    )
