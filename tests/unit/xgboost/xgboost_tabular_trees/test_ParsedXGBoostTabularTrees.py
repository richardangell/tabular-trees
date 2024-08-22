import pandas as pd
import pytest

from tabular_trees.trees import BaseModelTabularTrees
from tabular_trees.xgboost.dump_parser import ParsedXGBoostTabularTrees
from tabular_trees.xgboost.xgboost_tabular_trees import XGBoostTabularTrees


class TestInitialisation:
    """Tests for the ParsedXGBoostTabularTrees.__init__ method."""

    def test_inheritance(self):
        """Test that ParsedXGBoostTabularTrees inherits from BaseModelTabularTrees."""
        assert (
            ParsedXGBoostTabularTrees.__mro__[1] is BaseModelTabularTrees
        ), "ParsedXGBoostTabularTrees does not inherit from BaseModelTabularTrees"

    def test_trees_attribute_set(self, xgb_diabetes_model_parsed_trees_dataframe):
        """Test the trees attribute is set as the value passed in init."""
        prased_xgboost_tabular_trees = ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe
        )

        pd.testing.assert_frame_equal(
            prased_xgboost_tabular_trees.trees,
            xgb_diabetes_model_parsed_trees_dataframe[
                ParsedXGBoostTabularTrees.REQUIRED_COLUMNS
            ]
            .sort_values(ParsedXGBoostTabularTrees.SORT_BY_COLUMNS)
            .reset_index(drop=True),
        )

    def test_trees_not_same_object(self, xgb_diabetes_model_parsed_trees_dataframe):
        """Test the trees attribute is copied from what is passed."""
        prased_xgboost_tabular_trees = ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe
        )

        assert id(prased_xgboost_tabular_trees.trees) != id(
            xgb_diabetes_model_parsed_trees_dataframe
        ), "trees attribute is the same object as passed into initialisation"

    def test_post_init_called(self, mocker, xgb_diabetes_model_parsed_trees_dataframe):
        """Test that BaseModelTabularTrees.__post_init__ is called."""
        mocker.patch.object(BaseModelTabularTrees, "__post_init__")

        ParsedXGBoostTabularTrees(xgb_diabetes_model_parsed_trees_dataframe)

        assert (
            BaseModelTabularTrees.__post_init__.call_count == 1
        ), "BaseModelTabularTrees.__post_init__ not called once during __init__"

    def test_sort_by_columns_subset_required_columns(self):
        """Test that SORT_BY_COLUMNS is a subset of REQUIRED_COLUMNS."""
        assert all(
            column in ParsedXGBoostTabularTrees.REQUIRED_COLUMNS
            for column in ParsedXGBoostTabularTrees.SORT_BY_COLUMNS
        ), "not all SORT_BY_COLUMNS values are in REQUIRED_COLUMNS"

    def test_xgboost_tabular_trees_required_columns_in_column_mapping(self):
        """Test that SORT_BY_COLUMNS is a subset of REQUIRED_COLUMNS."""
        assert all(
            column in XGBoostTabularTrees.REQUIRED_COLUMNS
            for column in ParsedXGBoostTabularTrees.COLUMN_MAPPING.values()
        ), "not all XGBoostTabularTrees.REQUIRED_COLUMNS values are in COLUMN_MAPPING"


class TestPostInit:
    """Tests for the ParsedXGBoostTabularTrees.__post_init__ method."""

    def test_exception_if_no_stats(
        self, xgb_diabetes_model_parsed_trees_dataframe_no_stats
    ):
        """Test a ValueError is raised if the tree data is missing stats columns.

        Cover and gain columns are required.

        """
        expected_exception_message = (
            "Cannot create ParsedXGBoostTabularTrees object unless statistics "
            "are output. Rerun dump_model with with_stats = True."
        )

        with pytest.raises(ValueError, match=expected_exception_message):
            ParsedXGBoostTabularTrees(
                xgb_diabetes_model_parsed_trees_dataframe_no_stats
            )

    def test_super_post_init_called(
        self, mocker, xgb_diabetes_model_parsed_trees_dataframe
    ):
        """Test that BaseModelTabularTrees.__post_init__ method is called."""
        # initialise object then overwrite trees attribute with data that does
        # not contain the stats columns
        parsed_tabular_trees = ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe
        )

        mocked = mocker.patch.object(BaseModelTabularTrees, "__post_init__")

        parsed_tabular_trees.__post_init__()

        assert mocked.call_count == 1, (
            "BaseModelTabularTrees.__post_init__ not called when "
            "ParsedXGBoostTabularTrees.__post_init__ runs"
        )

        assert (
            mocked.call_args_list[0][0] == ()
        ), "positional args in BaseModelTabularTrees.__post_init__ call not correct"

        assert (
            mocked.call_args_list[0][1] == {}
        ), "keyword args in BaseModelTabularTrees.__post_init__ call not correct"


class TestConvertToXgboostTabularTrees:
    """Tests for ParsedXGBoostTabularTrees.convert_to_xgboost_tabular_trees method."""

    def test_output_type(self, xgb_diabetes_model_parsed_trees_dataframe):
        """Test convert_to_xgboost_tabular_trees output is XGBoostTabularTrees type."""
        parsed_tabular_trees = ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe
        )

        output = parsed_tabular_trees.convert_to_xgboost_tabular_trees()

        assert type(output) is XGBoostTabularTrees, (
            "output from ParsedXGBoostTabularTrees.convert_to_xgboost_tabular_trees "
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

        xgboost_tabular_trees = parsed_tabular_trees.convert_to_xgboost_tabular_trees()

        pd.testing.assert_frame_equal(
            xgboost_tabular_trees.trees.drop(columns=["weight", "G", "H"]),
            expected_output,
        )
