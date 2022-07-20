import pytest

import tabular_trees.lightgbm.trees as trees
from tabular_trees.trees import BaseModelTabularTrees


class TestLightGBMTabularTreesInit:
    """Tests for the LightGBMTabularTrees class."""

    def test_successfull_call(self, lgb_diabetes_model_trees_dataframe):
        """Test successfull initialisation of the LightGBMTabularTrees class."""

        trees.LightGBMTabularTrees(lgb_diabetes_model_trees_dataframe)

    def test_inheritance(self):
        """Test that LightGBMTabularTrees inherits from BaseModelTabularTrees."""

        assert (
            trees.LightGBMTabularTrees.__mro__[1] is BaseModelTabularTrees
        ), "ScikitLearnHistTabularTrees does not inherit from BaseModelTabularTrees"

    @pytest.mark.parametrize(
        "attribute_name,expected_value",
        [
            ("SORT_BY_COLUMNS", ["tree_index", "node_depth", "node_index"]),
            (
                "EXPECTED_COLUMNS",
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
        self, attribute_name, expected_value, lgb_diabetes_model_trees_dataframe
    ):
        """Test the SORT_BY_COLUMNS attribute is set as expected."""

        assert (
            getattr(trees.LightGBMTabularTrees, attribute_name) == expected_value
        ), f"{attribute_name} not expected on LightGBMTabularTrees class"

        assert (
            getattr(
                trees.LightGBMTabularTrees(lgb_diabetes_model_trees_dataframe),
                attribute_name,
            )
            == expected_value
        ), f"{attribute_name} not expected on LightGBMTabularTrees object after initialisation"
