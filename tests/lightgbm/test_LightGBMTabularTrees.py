import pytest

from tabular_trees.trees import BaseModelTabularTrees
from tabular_trees.lightgbm.trees import LightGBMTabularTrees


def test_successfull_call(lgb_diabetes_model_trees_dataframe):
    """Test successfull initialisation of the LightGBMTabularTrees class."""

    LightGBMTabularTrees(lgb_diabetes_model_trees_dataframe)


def test_inheritance():
    """Test that LightGBMTabularTrees inherits from BaseModelTabularTrees."""

    assert (
        LightGBMTabularTrees.__mro__[1] is BaseModelTabularTrees
    ), "ScikitLearnHistTabularTrees does not inherit from BaseModelTabularTrees"


@pytest.mark.parametrize(
    "attribute_name,expected_value",
    [
        ("SORT_BY_COLUMNS", ["tree_index", "node_depth", "node_index"]),
        (
            "REQUIRED_COLUMNS",
            [
                "tree_index",
                "node_depth",
                "node_index",
                "left_child",
                "right_child",
                "parent_index",
                "split_feature",
                "split_gain",
                "threshold",
                "decision_type",
                "missing_direction",
                "missing_type",
                "value",
                "weight",
                "count",
            ],
        ),
    ],
)
def test_sort_by_columns(
    attribute_name, expected_value, lgb_diabetes_model_trees_dataframe
):
    """Test column related attributes are set as expected."""

    assert (
        getattr(LightGBMTabularTrees, attribute_name) == expected_value
    ), f"{attribute_name} not expected on LightGBMTabularTrees class"

    tabular_trees = LightGBMTabularTrees(lgb_diabetes_model_trees_dataframe)

    assert (
        getattr(
            tabular_trees,
            attribute_name,
        )
        == expected_value
    ), f"{attribute_name} not expected on LightGBMTabularTrees object after initialisation"


def test_trees_not_same_object(lgb_diabetes_model_trees_dataframe):
    """Test the trees attribute is not the same object as that passed into
    the init method."""

    input_df = lgb_diabetes_model_trees_dataframe.copy()

    tabular_trees = LightGBMTabularTrees(input_df)

    assert id(tabular_trees.trees) != id(
        input_df
    ), "trees attribute is the same object as passed into initialisation"
