import re

import pandas as pd
import pytest

from tabular_trees.sklearn import scikit_learn_hist_tabular_trees


def test_model_not_correct_type_exception():
    """Test that a TypeError is raised if model is not allowed type."""
    msg = (
        "model is not in expected types (<class "
        "'sklearn.ensemble._hist_gradient_boosting.gradient_boosting."
        "HistGradientBoostingClassifier'>, "
        "<class 'sklearn.ensemble._hist_gradient_boosting.gradient_boosting."
        "HistGradientBoostingRegressor'>), "
        "got <class 'list'>"
    )

    with pytest.raises(TypeError, match=re.escape(msg)):
        scikit_learn_hist_tabular_trees._export_tree_data__hist_gradient_boosting_model(
            ["a", "b"]
        )


def test_multiple_responses_exception(sklearn_iris_hist_gbm_classifier):
    """Test an error is raised if the model passed has multiple responses.

    An example would be a multiclass classification model.

    """
    with pytest.raises(
        NotImplementedError, match="model with multiple responses not supported"
    ):
        scikit_learn_hist_tabular_trees._export_tree_data__hist_gradient_boosting_model(
            sklearn_iris_hist_gbm_classifier
        )


def test_output(mocker, sklearn_diabetes_hist_gbm_regressor):
    """Test output is the output from _extract_hist_gbm_tree_data."""
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
        scikit_learn_hist_tabular_trees,
        "_extract_hist_gbm_tree_data",
        return_value=dummy_tree_data,
    )

    exported_trees = (
        scikit_learn_hist_tabular_trees._export_tree_data__hist_gradient_boosting_model(
            sklearn_diabetes_hist_gbm_regressor
        )
    )

    assert (
        type(exported_trees)
        is scikit_learn_hist_tabular_trees.ScikitLearnHistTabularTrees
    ), "output from _export_tree_data__hist_gradient_boosting_model incorrect type"

    pd.testing.assert_frame_equal(exported_trees.trees, dummy_tree_data)
