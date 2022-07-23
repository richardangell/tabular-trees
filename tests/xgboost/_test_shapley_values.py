import pandas as pd

import build_model

import tabular_trees


def test_expected_shapley_values():
    """Function to test shapley values output are as expected for simple model."""

    model = build_model.build_example_shap_model()

    tree_df = tabular_trees.xgboost.parser.parse_model(model)

    row_to_explain = pd.Series({"x": 150, "y": 75, "z": 200})

    shapley_values = tabular_trees.xgboost.explainer.shapley_values(
        tree_df.tree_data, row_to_explain, False
    )

    expected_values = {"bias": 23.0, "x": -5.0, "y": 2.0, "z": 0.0}

    expected_shapley_values = pd.DataFrame(expected_values, index=[0])

    pd.testing.assert_frame_equal(shapley_values, expected_shapley_values)
