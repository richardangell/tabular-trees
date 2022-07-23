import pytest

from tabular_trees.trees import BaseModelTabularTrees
from tabular_trees.sklearn.trees import ScikitLearnTabularTrees


def test_successfull_call(sklearn_gbm_trees_dataframe):
    """Test successfull initialisation of the ScikitLearnTabularTrees class."""

    ScikitLearnTabularTrees(sklearn_gbm_trees_dataframe)


def test_inheritance():
    """Test that ScikitLearnTabularTrees inherits from BaseModelTabularTrees."""

    assert (
        ScikitLearnTabularTrees.__mro__[1] is BaseModelTabularTrees
    ), "ScikitLearnTabularTrees does not inherit from BaseModelTabularTrees"


@pytest.mark.parametrize(
    "attribute_name,expected_value",
    [
        ("SORT_BY_COLUMNS", ["tree", "node"]),
        (
            "REQUIRED_COLUMNS",
            [
                "tree",
                "node",
                "children_left",
                "children_right",
                "feature",
                "impurity",
                "n_node_samples",
                "threshold",
                "value",
                "weighted_n_node_samples",
            ],
        ),
    ],
)
def test_column_attributes(attribute_name, expected_value, sklearn_gbm_trees_dataframe):
    """Test column related attributes are set as expected."""

    assert (
        getattr(ScikitLearnTabularTrees, attribute_name) == expected_value
    ), f"{attribute_name} not expected on ScikitLearnTabularTrees class"

    tabular_trees = ScikitLearnTabularTrees(sklearn_gbm_trees_dataframe)

    assert (
        getattr(
            tabular_trees,
            attribute_name,
        )
        == expected_value
    ), f"{attribute_name} not expected on ScikitLearnTabularTrees object after initialisation"


def test_trees_not_same_object(sklearn_gbm_trees_dataframe):
    """Test the trees attribute is not the same object as that passed into
    the init method."""

    input_df = sklearn_gbm_trees_dataframe.copy()

    tabular_trees = ScikitLearnTabularTrees(input_df)

    assert id(tabular_trees.trees) != id(
        input_df
    ), "trees attribute is the same object as passed into initialisation"
