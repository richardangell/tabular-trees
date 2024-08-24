from tabular_trees.lightgbm.lightgbm_tabular_trees import LightGBMTabularTrees
from tabular_trees.trees import BaseModelTabularTrees, TabularTrees


class TestInitialisation:
    """Tests for the LightGBMTabularTrees.__init__ method."""

    def test_inheritance(self):
        """Test that LightGBMTabularTrees inherits from BaseModelTabularTrees."""
        assert (
            LightGBMTabularTrees.__mro__[1] is BaseModelTabularTrees
        ), "ScikitLearnHistTabularTrees does not inherit from BaseModelTabularTrees"

    def test_trees_not_same_object(self, lgb_diabetes_model_trees_dataframe):
        """Test trees attribute is copied from the data passed in init."""
        tabular_trees = LightGBMTabularTrees(lgb_diabetes_model_trees_dataframe)

        assert id(tabular_trees.data) != id(
            lgb_diabetes_model_trees_dataframe
        ), "trees attribute is the same object as passed into initialisation"


class TestConvertToTabularTrees:
    """Tests for the LightGBMTabularTrees.to_tabular_trees method."""

    def test_output_type(self, lgb_diabetes_model_trees_dataframe):
        """Test the output from to_tabular_trees is a TabularTrees object."""
        lightgbm_tabular_trees = LightGBMTabularTrees(
            lgb_diabetes_model_trees_dataframe
        )

        output = lightgbm_tabular_trees.to_tabular_trees()

        assert type(output) is TabularTrees, (
            "output from LightGBMTabularTrees.to_tabular_trees "
            "is not TabularTrees type"
        )
