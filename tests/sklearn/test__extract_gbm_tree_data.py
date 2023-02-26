import pytest

from tabular_trees.sklearn import trees


def test_successful_call(sklearn_diabetes_gbm_regressor):
    """Test a successful call to _extract_gbm_tree_data."""

    trees._extract_gbm_tree_data(sklearn_diabetes_gbm_regressor)


def test_required_columns(sklearn_diabetes_gbm_regressor):
    """Test the required columns are in the output."""

    tree_data = trees._extract_gbm_tree_data(sklearn_diabetes_gbm_regressor)

    assert sorted(trees.ScikitLearnTabularTrees.REQUIRED_COLUMNS) == sorted(
        tree_data.columns.values
    ), "columns in output from _extract_gbm_tree_data not correct"


@pytest.mark.skip(reason="not implemented")
def test_output_values():
    """Test that the values output are correct for a simple, known tree."""

    pass
