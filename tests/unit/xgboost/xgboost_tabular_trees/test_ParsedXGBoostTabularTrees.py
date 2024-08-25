import pandas as pd

from tabular_trees.xgboost.dump_parser import ParsedXGBoostTabularTrees
from tabular_trees.xgboost.xgboost_tabular_trees import XGBoostTabularTrees


class TestInitialisation:
    """Tests for the ParsedXGBoostTabularTrees.__init__ method."""

    def test_trees_not_same_object(self, xgb_diabetes_model_parsed_trees_dataframe):
        """Test the trees attribute is copied from what is passed."""
        prased_xgboost_tabular_trees = ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe
        )

        assert id(prased_xgboost_tabular_trees.data) != id(
            xgb_diabetes_model_parsed_trees_dataframe
        ), "trees attribute is the same object as passed into initialisation"


class TestConvertToXgboostTabularTrees:
    """Tests for ParsedXGBoostTabularTrees.to_xgboost_tabular_trees method."""

    def test_output_type(self, xgb_diabetes_model_parsed_trees_dataframe):
        """Test to_xgboost_tabular_trees output is XGBoostTabularTrees type."""
        parsed_tabular_trees = ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe
        )

        output = parsed_tabular_trees.to_xgboost_tabular_trees()

        assert type(output) is XGBoostTabularTrees, (
            "output from ParsedXGBoostTabularTrees.to_xgboost_tabular_trees "
            "is not XGBoostTabularTrees type"
        )

    def test_output_same_format_as_xgboost(
        self,
        xgb_diabetes_model_trees_dataframe,
        xgb_diabetes_model_parsed_trees_dataframe,
    ):
        """Test output matches xgb.Booster.trees_to_dataframe."""
        expected_output = xgb_diabetes_model_trees_dataframe

        parsed_tabular_trees = ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe
        )

        xgboost_tabular_trees = parsed_tabular_trees.to_xgboost_tabular_trees()

        subset_actual = xgboost_tabular_trees.data.drop(columns=["weight", "G", "H"])

        pd.testing.assert_frame_equal(
            subset_actual[expected_output.columns],
            expected_output,
        )
