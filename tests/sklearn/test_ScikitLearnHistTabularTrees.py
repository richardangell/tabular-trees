import pytest

from tabular_trees.trees import BaseModelTabularTrees
from tabular_trees.sklearn.trees import ScikitLearnHistTabularTrees


class TestScikitLearnHistTabularTreesInit:
    """Tests for the ScikitLearnHistTabularTrees class."""

    def test_successfull_call(self, sklearn_diabetes_model_trees_dataframe):
        """Test successfull initialisation of the ScikitLearnHistTabularTrees class."""

        ScikitLearnHistTabularTrees(sklearn_diabetes_model_trees_dataframe)

    def test_inheritance(self):
        """Test that ScikitLearnHistTabularTrees inherits from BaseModelTabularTrees."""

        assert (
            ScikitLearnHistTabularTrees.__mro__[1] is BaseModelTabularTrees
        ), "ScikitLearnHistTabularTrees does not inherit from BaseModelTabularTrees"

    @pytest.mark.parametrize(
        "attribute_name,expected_value",
        [
            ("SORT_BY_COLUMNS", ["tree", "node"]),
            (
                "REQUIRED_COLUMNS",
                [
                    "tree",
                    "node",
                    "value",
                    "count",
                    "feature_idx",
                    "num_threshold",
                    "missing_go_to_left",
                    "left",
                    "right",
                    "gain",
                    "depth",
                    "is_leaf",
                    "bin_threshold",
                    "is_categorical",
                    "bitset_idx",
                ],
            ),
        ],
    )
    def test_sort_by_columns(
        self, attribute_name, expected_value, sklearn_diabetes_model_trees_dataframe
    ):
        """Test column related attributes are set as expected."""

        assert (
            getattr(ScikitLearnHistTabularTrees, attribute_name) == expected_value
        ), f"{attribute_name} not expected on ScikitLearnHistTabularTrees class"

        tabular_trees = ScikitLearnHistTabularTrees(
            sklearn_diabetes_model_trees_dataframe
        )

        assert (
            getattr(
                tabular_trees,
                attribute_name,
            )
            == expected_value
        ), f"{attribute_name} not expected on ScikitLearnHistTabularTrees object after initialisation"

    def test_trees_not_same_object(self, sklearn_diabetes_model_trees_dataframe):
        """Test the trees attribute is not the same object as that passed into
        the init method."""

        input_df = sklearn_diabetes_model_trees_dataframe.copy()

        tabular_trees = ScikitLearnHistTabularTrees(input_df)

        assert id(tabular_trees.trees) != id(
            input_df
        ), "trees attribute is the same object as passed into initialisation"