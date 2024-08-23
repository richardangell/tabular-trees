import pandas as pd

from tabular_trees.sklearn.scikit_learn_hist_tabular_trees import (
    ScikitLearnHistTabularTrees,
)
from tabular_trees.trees import BaseModelTabularTrees


def test_inheritance():
    """Test that ScikitLearnHistTabularTrees inherits from BaseModelTabularTrees."""
    assert (
        ScikitLearnHistTabularTrees.__mro__[1] is BaseModelTabularTrees
    ), "ScikitLearnHistTabularTrees does not inherit from BaseModelTabularTrees"


def test_trees_attribute_set(sklearn_hist_gbm_trees_dataframe):
    """Test the trees attribute is set as the value passed in init."""
    tabular_trees = ScikitLearnHistTabularTrees(sklearn_hist_gbm_trees_dataframe)

    pd.testing.assert_frame_equal(
        tabular_trees.trees,
        sklearn_hist_gbm_trees_dataframe[ScikitLearnHistTabularTrees.REQUIRED_COLUMNS]
        .sort_values(ScikitLearnHistTabularTrees.SORT_BY_COLUMNS)
        .reset_index(drop=True),
    )


def test_trees_not_same_object(sklearn_hist_gbm_trees_dataframe):
    """Test trees is copied from the input data."""
    tabular_trees = ScikitLearnHistTabularTrees(sklearn_hist_gbm_trees_dataframe)

    assert id(tabular_trees.trees) != id(
        sklearn_hist_gbm_trees_dataframe
    ), "trees attribute is the same object as passed into initialisation"


def test_post_init_called(mocker, sklearn_hist_gbm_trees_dataframe):
    """Test that BaseModelTabularTrees.__post_init__ is called."""
    mocker.patch.object(BaseModelTabularTrees, "__post_init__")

    ScikitLearnHistTabularTrees(sklearn_hist_gbm_trees_dataframe)

    assert (
        BaseModelTabularTrees.__post_init__.call_count == 1
    ), "BaseModelTabularTrees.__post_init__ not called once during __init__"


def test_sort_by_columns_subset_required_columns():
    """Test that SORT_BY_COLUMNS is a subset of REQUIRED_COLUMNS."""
    assert all(
        column in ScikitLearnHistTabularTrees.REQUIRED_COLUMNS
        for column in ScikitLearnHistTabularTrees.SORT_BY_COLUMNS
    ), "not all SORT_BY_COLUMNS values are in REQUIRED_COLUMNS"
