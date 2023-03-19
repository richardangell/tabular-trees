import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor

from tabular_trees import sklearn


@pytest.fixture
def handcrafted_data():
    """Handcraft dataset that can be expressed with a simple depth 2 tree.

    Data is as follows;
    a	b	response
    1	1	100
    1	1	100
    1	-1	150
    1	-1	150
    -1	1	200
    -1	1	200
    -1	-1	250
    -1	-1	250

    """
    data = pd.DataFrame(
        {
            "a": [1, 1, 1, 1, -1, -1, -1, -1],
            "b": [1, 1, -1, -1, 1, 1, -1, -1],
            "response": [100, 100, 150, 150, 200, 200, 250, 250],
        }
    )

    return data


def test_successful_call(sklearn_diabetes_gbm_regressor):
    """Test a successful call to _extract_gbm_tree_data."""

    sklearn._extract_gbm_tree_data(sklearn_diabetes_gbm_regressor)


def test_required_columns(sklearn_diabetes_gbm_regressor):
    """Test the required columns are in the output."""

    tree_data = sklearn._extract_gbm_tree_data(sklearn_diabetes_gbm_regressor)

    assert sorted(sklearn.ScikitLearnTabularTrees.REQUIRED_COLUMNS) == sorted(
        tree_data.columns.values
    ), "columns in output from _extract_gbm_tree_data not correct"


def test_output_values(handcrafted_data):
    """Test that the values output are expected for a simple, known tree."""

    model = GradientBoostingRegressor(n_estimators=1, max_depth=2, learning_rate=1)

    model.fit(handcrafted_data[["a", "b"]], handcrafted_data["response"])

    extracted_tree_data = sklearn._extract_gbm_tree_data(model)

    expected_tree_data = pd.DataFrame(
        {
            "node": [0, 1, 2, 3, 4, 5, 6],
            "children_left": [1, 2, -1, -1, 5, -1, -1],
            "children_right": [4, 3, -1, -1, 6, -1, -1],
            "feature": [0, 1, -2, -2, 1, -2, -2],
            "impurity": [3125.0, 625.0, 0.0, 0.0, 625.0, 0.0, 0.0],
            "n_node_samples": [8, 4, 2, 2, 4, 2, 2],
            "threshold": [0.0, 0.0, -2.0, -2.0, 0.0, -2.0, -2.0],
            "value": [175.0, 225.0, 250.0, 200.0, 125.0, 150.0, 100.0],
            "weighted_n_node_samples": [8.0, 4.0, 2.0, 2.0, 4.0, 2.0, 2.0],
            "tree": [0, 0, 0, 0, 0, 0, 0],
        }
    )

    pd.testing.assert_frame_equal(extracted_tree_data, expected_tree_data)
