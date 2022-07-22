import pytest
import re
import pandas as pd

from tabular_trees.sklearn import trees


def test_successful_call(sklearn_diabetes_model):
    """Test a successful call to export_tree_data__hist_gradient_boosting_model."""

    trees.export_tree_data__hist_gradient_boosting_model(sklearn_diabetes_model)


def test_model_not_correct_type_exception():
    """Test that a TypeError is raised if model is not"""

    msg = (
        "model is not in expected types (<class "
        "'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingClassifier'>, "
        "<class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting.HistGradientBoostingRegressor'>), "
        "got <class 'list'>"
    )

    with pytest.raises(TypeError, match=re.escape(msg)):

        trees.export_tree_data__hist_gradient_boosting_model(["a", "b"])


def test_multiple_responses_exception(sklearn_iris_model):
    """Test a NotImplementedError is raised if the model passed has multiple
    responses e.g. multiclass classification."""

    with pytest.raises(
        NotImplementedError, match="model with multiple responses not supported"
    ):

        trees.export_tree_data__hist_gradient_boosting_model(sklearn_iris_model)


def test_output(mocker, sklearn_diabetes_model):
    """Test that the output of the function is a ScikitLearnHistTabularTrees
    object with the output from _extract_hist_gbm_tree_data."""

    dummy_tree_data = pd.DataFrame(
        {
            "tree": 0,
            "node": 1,
            "value": 2,
            "count": 3,
            "feature_idx": 4,
            "num_threshold": 5,
            "missing_go_to_left": 6,
            "left": 7,
            "right": 8,
            "gain": 9,
            "depth": 10,
            "is_leaf": 11,
            "bin_threshold": 12,
            "is_categorical": 13,
            "bitset_idx": 14,
        },
        index=[0],
    )

    mocker.patch.object(
        trees, "_extract_hist_gbm_tree_data", return_value=dummy_tree_data
    )

    exported_trees = trees.export_tree_data__hist_gradient_boosting_model(
        sklearn_diabetes_model
    )

    assert (
        type(exported_trees) is trees.ScikitLearnHistTabularTrees
    ), "output from export_tree_data__hist_gradient_boosting_model incorrect type"

    pd.testing.assert_frame_equal(exported_trees.trees, dummy_tree_data)
