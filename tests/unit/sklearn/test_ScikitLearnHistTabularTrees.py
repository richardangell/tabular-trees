import re

import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from tabular_trees.sklearn.sklearn_hist_tabular_trees import (
    ScikitLearnHistTabularTrees,
)


def test_trees_not_same_object(sklearn_hist_gbm_trees_dataframe):
    """Test trees is copied from the input data."""
    tabular_trees = ScikitLearnHistTabularTrees(sklearn_hist_gbm_trees_dataframe)

    assert id(tabular_trees.data) != id(
        sklearn_hist_gbm_trees_dataframe
    ), "trees attribute is the same object as passed into initialisation"


class TestFromHistGradientBooster:
    """Tests for the ScikitLearnHistTabularTrees.from_hist_gradient_booster method."""

    def test_model_not_correct_type_exception(self):
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
            ScikitLearnHistTabularTrees.from_hist_gradient_booster(["a", "b"])

    def test_multiple_responses_exception(self, sklearn_iris_hist_gbm_classifier):
        """Test an error is raised if the model passed has multiple responses.

        An example would be a multiclass classification model.

        """
        with pytest.raises(
            NotImplementedError, match="model with multiple responses not supported"
        ):
            ScikitLearnHistTabularTrees.from_hist_gradient_booster(
                sklearn_iris_hist_gbm_classifier
            )

    def test_output(self, handcrafted_data):
        """Test output is the output is expected."""
        model = HistGradientBoostingRegressor(
            max_iter=1, max_depth=2, learning_rate=1, min_samples_leaf=1
        )

        model.fit(handcrafted_data[["a", "b"]], handcrafted_data["response"])

        actual = ScikitLearnHistTabularTrees.from_hist_gradient_booster(model)

        expected_tree_data = pd.DataFrame(
            {
                "tree": [0, 0, 0, 0, 0, 0, 0],
                "node": [0, 1, 2, 3, 4, 5, 6],
                "value": [175.0, 225.0, 250.0, 200.0, 125.0, 150.0, 100.0],
                "count": [8, 4, 2, 2, 4, 2, 2],
                "feature_idx": [0, 1, 0, 0, 1, 0, 0],
                "num_threshold": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "missing_go_to_left": [0, 0, 0, 0, 0, 0, 0],
                "left": [1, 2, 0, 0, 5, 0, 0],
                "right": [4, 3, 0, 0, 6, 0, 0],
                "gain": [20000.0, 2500.0, -1.0, -1.0, 2500.0, -1.0, -1.0],
                "depth": [0, 1, 2, 2, 1, 2, 2],
                "is_leaf": [0, 0, 1, 1, 0, 1, 1],
                "bin_threshold": [0, 0, 0, 0, 0, 0, 0],
                "is_categorical": [0, 0, 0, 0, 0, 0, 0],
                "bitset_idx": [0, 0, 0, 0, 0, 0, 0],
            }
        )

        assert type(actual) is ScikitLearnHistTabularTrees

        pd.testing.assert_frame_equal(
            actual.data[expected_tree_data.columns],
            expected_tree_data,
            check_dtype=False,
        )
