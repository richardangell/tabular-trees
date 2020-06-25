import xgboost as xgb
import pandas as pd
import numpy as np

import build_model

import pygbmexpl



def test_expected_shapley_values():
    """Function to test shapley values output are as expected for simple model."""

    model = build_model.build_example_shap_model()

    tree_df = pygbmexpl.xgb.parser.extract_model_predictions(model)
    
    row_to_explain = pd.Series({'x': 150, 'y': 75, 'z': 200})

    shapley_values = pygbmexpl.xgb.explainer.shapley_values(tree_df, row_to_explain, False)

    expected_values = {
        'bias': 23.0,
        'x': -5.0,
        'y': 2.0,
        'z': 0.0
    }

    expected_shapley_values = pd.DataFrame(expected_values, index = [0])

    pd.testing.assert_frame_equal(shapley_values, expected_shapley_values)
