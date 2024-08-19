import pandas as pd

from tabular_trees.lightgbm import LightGBMTabularTrees
from tabular_trees.trees import BaseModelTabularTrees, TabularTrees


class TestLightGBMTabularTreesInit:
    """Tests for the LightGBMTabularTrees.__init__ method."""

    def test_inheritance(self):
        """Test that LightGBMTabularTrees inherits from BaseModelTabularTrees."""
        assert (
            LightGBMTabularTrees.__mro__[1] is BaseModelTabularTrees
        ), "ScikitLearnHistTabularTrees does not inherit from BaseModelTabularTrees"

    def test_trees_attribute_set(self, lgb_diabetes_model_trees_dataframe):
        """Test the trees attribute is set as the value passed in init."""
        tabular_trees = LightGBMTabularTrees(lgb_diabetes_model_trees_dataframe)

        pd.testing.assert_frame_equal(
            tabular_trees.trees,
            lgb_diabetes_model_trees_dataframe.sort_values(
                LightGBMTabularTrees.SORT_BY_COLUMNS
            ).reset_index(drop=True),
        )

    def test_trees_not_same_object(self, lgb_diabetes_model_trees_dataframe):
        """Test trees attribute is copied from the data passed in init."""
        tabular_trees = LightGBMTabularTrees(lgb_diabetes_model_trees_dataframe)

        assert id(tabular_trees.trees) != id(
            lgb_diabetes_model_trees_dataframe
        ), "trees attribute is the same object as passed into initialisation"

    def test_post_init_called(self, mocker, lgb_diabetes_model_trees_dataframe):
        """Test that BaseModelTabularTrees.__post_init__ is called."""
        mocker.patch.object(BaseModelTabularTrees, "__post_init__")

        LightGBMTabularTrees(lgb_diabetes_model_trees_dataframe)

        assert (
            BaseModelTabularTrees.__post_init__.call_count == 1
        ), "BaseModelTabularTrees.__post_init__ not called once during __init__"

    def test_sort_by_columns_subset_required_columns(self):
        """Test that SORT_BY_COLUMNS is a subset of REQUIRED_COLUMNS."""
        assert all(
            column in LightGBMTabularTrees.REQUIRED_COLUMNS
            for column in LightGBMTabularTrees.SORT_BY_COLUMNS
        ), "not all SORT_BY_COLUMNS values are in REQUIRED_COLUMNS"

    def test_tabular_trees_required_columns_in_column_mapping(self):
        """Test that SORT_BY_COLUMNS is a subset of REQUIRED_COLUMNS."""
        assert all(
            column in TabularTrees.REQUIRED_COLUMNS
            for column in LightGBMTabularTrees.COLUMN_MAPPING.values()
        ), "not all TabularTrees.REQUIRED_COLUMNS values are in COLUMN_MAPPING"


class TestLightGBMTabularTreesConvert:
    """Tests for the LightGBMTabularTrees.convert_to_tabular_trees method."""

    def test_output_type(self, lgb_diabetes_model_trees_dataframe):
        """Test the output from convert_to_tabular_trees is a TabularTrees object."""
        lightgbm_tabular_trees = LightGBMTabularTrees(
            lgb_diabetes_model_trees_dataframe
        )

        output = lightgbm_tabular_trees.convert_to_tabular_trees()

        assert type(output) is TabularTrees, (
            "output from LightGBMTabularTrees.convert_to_tabular_trees "
            "is not TabularTrees type"
        )
