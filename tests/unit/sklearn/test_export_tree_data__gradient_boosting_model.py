import re

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from tabular_trees.sklearn.scikit_learn_tabular_trees import (
    ScikitLearnTabularTrees,
    _export_tree_data__gradient_boosting_model,
)
from tabular_trees.trees import export_tree_data


def test_model_not_correct_type_exception():
    """Test that a TypeError is raised if model is not an allowed type."""
    msg = (
        "model is not in expected types (<class "
        "'sklearn.ensemble._gb.GradientBoostingClassifier'>, "
        "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>), "
        "got <class 'list'>"
    )

    with pytest.raises(TypeError, match=re.escape(msg)):
        _export_tree_data__gradient_boosting_model(["a", "b"])


def test_multiple_responses_exception(sklearn_iris_gbm_classifier):
    """Test an eror is raised if the model passed has multiple responses."""
    with pytest.raises(
        NotImplementedError, match="model with multiple responses not supported"
    ):
        _export_tree_data__gradient_boosting_model(sklearn_iris_gbm_classifier)


def test_output(handcrafted_data):
    """Test output from _extract_hist_gbm_tree_data is returned."""
    model = GradientBoostingRegressor(n_estimators=1, max_depth=2, learning_rate=1)

    model.fit(handcrafted_data[["a", "b"]], handcrafted_data["response"])

    actual = export_tree_data(model)

    expected_tree_data = pd.DataFrame(
        {
            "tree": [0, 0, 0, 0, 0, 0, 0],
            "node": [0, 1, 2, 3, 4, 5, 6],
            "children_left": [1, 2, -1, -1, 5, -1, -1],
            "children_right": [4, 3, -1, -1, 6, -1, -1],
            "feature": [0, 1, -2, -2, 1, -2, -2],
            "impurity": [3125.0, 625.0, 0.0, 0.0, 625.0, 0.0, 0.0],
            "n_node_samples": [8, 4, 2, 2, 4, 2, 2],
            "threshold": [0.0, 0.0, -2.0, -2.0, 0.0, -2.0, -2.0],
            "value": [175.0, 225.0, 250.0, 200.0, 125.0, 150.0, 100.0],
            "weighted_n_node_samples": [8.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0],
        }
    )

    # note, the starting value for the tree is the mean of the response
    expected_tree_data["value"] = expected_tree_data["value"] - np.mean(
        handcrafted_data["response"]
    )

    assert type(actual) is ScikitLearnTabularTrees

    pd.testing.assert_frame_equal(actual.trees, expected_tree_data, check_dtype=False)
