import pandas as pd

import tabular_trees

import build_model


def test_prediction_decomposition_eqal_eli5():
    """Test that the prediction decomposition outputs from xgb.explainer.decompose_prediction are eqaul to the outputs from eli5."""

    # reduce the number of columns as the non-treeSHAP algorithm will never finish with 13 variables
    columns_to_drop = ["NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"]

    model, xgb_data, data = build_model.build_depth_3_model(
        return_data=True, drop_columns=columns_to_drop
    )

    shapley_values_xgb = model.predict(xgb_data, pred_contribs=True)

    shapley_values_xgb = pd.DataFrame(
        shapley_values_xgb, columns=xgb_data.feature_names + ["bias"]
    )

    row_data = data.iloc[0]

    tree_df = tabular_trees.xgboost.parser.parse_model(model)

    shapley_values = tabular_trees.xgboost.explainer.shapley_values(
        tree_df.tree_data, row_data, False
    )

    shapley_values = shapley_values[shapley_values_xgb.columns.values]

    pd.testing.assert_frame_equal(
        shapley_values_xgb.iloc[[0]], shapley_values, check_dtype=False
    )
