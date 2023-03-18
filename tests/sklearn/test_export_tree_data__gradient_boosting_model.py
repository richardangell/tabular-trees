import re

import pandas as pd
import pytest

from tabular_trees import sklearn


def test_successful_call(sklearn_diabetes_gbm_regressor):
    """Test a successful call to _export_tree_data__gradient_boosting_model."""

    sklearn._export_tree_data__gradient_boosting_model(sklearn_diabetes_gbm_regressor)


def test_model_not_correct_type_exception():
    """Test that a TypeError is raised if model is not"""

    msg = (
        "model is not in expected types (<class "
        "'sklearn.ensemble._gb.GradientBoostingClassifier'>, "
        "<class 'sklearn.ensemble._gb.GradientBoostingRegressor'>), "
        "got <class 'list'>"
    )

    with pytest.raises(TypeError, match=re.escape(msg)):

        sklearn._export_tree_data__gradient_boosting_model(["a", "b"])


def test_multiple_responses_exception(sklearn_iris_gbm_classifier):
    """Test a NotImplementedError is raised if the model passed has multiple
    responses e.g. multiclass classification."""

    with pytest.raises(
        NotImplementedError, match="model with multiple responses not supported"
    ):

        sklearn._export_tree_data__gradient_boosting_model(sklearn_iris_gbm_classifier)


def test_output(mocker, sklearn_diabetes_gbm_regressor):
    """Test that the output of the function is a ScikitLearnTabularTrees
    object with the output from _extract_hist_gbm_tree_data."""

    dummy_tree_data = pd.DataFrame(
        {
            "tree": 0,
            "node": 1,
            "children_left": 2,
            "children_right": 3,
            "feature": 4,
            "impurity": 5,
            "n_node_samples": 6,
            "threshold": 7,
            "value": 8,
            "weighted_n_node_samples": 9,
        },
        index=[0],
    )

    mocker.patch.object(sklearn, "_extract_gbm_tree_data", return_value=dummy_tree_data)

    exported_trees = sklearn._export_tree_data__gradient_boosting_model(
        sklearn_diabetes_gbm_regressor
    )

    assert (
        type(exported_trees) is sklearn.ScikitLearnTabularTrees
    ), "output from _export_tree_data__gradient_boosting_model incorrect type"

    pd.testing.assert_frame_equal(exported_trees.trees, dummy_tree_data)
