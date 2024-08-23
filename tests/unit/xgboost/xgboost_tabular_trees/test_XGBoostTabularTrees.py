import re

import pandas as pd
import pytest

from tabular_trees.trees import BaseModelTabularTrees, TabularTrees
from tabular_trees.xgboost.xgboost_tabular_trees import XGBoostTabularTrees


class TestInitialisation:
    """Tests for the XGBoostTabularTrees.__init__ method."""

    def test_inheritance(self):
        """Test that XGBoostTabularTrees inherits from BaseModelTabularTrees."""
        assert (
            XGBoostTabularTrees.__mro__[1] is BaseModelTabularTrees
        ), "XGBoostTabularTrees does not inherit from BaseModelTabularTrees"

    def test_trees_attribute_set(self, xgb_diabetes_model_trees_dataframe):
        """Test the trees attribute is set as the value passed in init."""
        tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        pd.testing.assert_frame_equal(
            tabular_trees.trees.drop(columns=["G", "H", "weight"]),
            xgb_diabetes_model_trees_dataframe.sort_values(
                XGBoostTabularTrees.SORT_BY_COLUMNS
            ).reset_index(drop=True),
        )

    def test_trees_not_same_object(self, xgb_diabetes_model_trees_dataframe):
        """Test the trees attribute is copied from what is passed."""
        input_df = xgb_diabetes_model_trees_dataframe.copy()

        tabular_trees = XGBoostTabularTrees(input_df)

        assert id(tabular_trees.trees) != id(
            input_df
        ), "trees attribute is the same object as passed into initialisation"

    def test_post_init_called(self, mocker, xgb_diabetes_model_trees_dataframe):
        """Test that XGBoostTabularTrees.__post_init__ is called."""
        mocker.patch.object(XGBoostTabularTrees, "__post_init__")

        XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        assert (
            XGBoostTabularTrees.__post_init__.call_count == 1
        ), "XGBoostTabularTrees.__post_init__ not called once during __init__"

    def test_sort_by_columns_subset_required_columns(self):
        """Test that SORT_BY_COLUMNS is a subset of REQUIRED_COLUMNS."""
        assert all(
            column in XGBoostTabularTrees.REQUIRED_COLUMNS
            for column in XGBoostTabularTrees.SORT_BY_COLUMNS
        ), "not all SORT_BY_COLUMNS values are in REQUIRED_COLUMNS"

    def test_tabular_trees_required_columns_in_column_mapping(self):
        """Test that SORT_BY_COLUMNS is a subset of REQUIRED_COLUMNS."""
        assert all(
            column in TabularTrees.REQUIRED_COLUMNS
            for column in XGBoostTabularTrees.COLUMN_MAPPING.values()
        ), "not all TabularTrees.REQUIRED_COLUMNS values are in COLUMN_MAPPING"


class TestPostInit:
    """Tests for the XGBoostTabularTrees.__post_init__ method."""

    def test_lambda_not_float_exception(self, xgb_diabetes_model_trees_dataframe):
        """Test an exception is raised if lambda_ is not a float."""
        tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        tabular_trees.lambda_ = "1"

        with pytest.raises(
            TypeError,
            match="lambda_ is not in expected types <class 'float'>, got <class 'str'>",
        ):
            tabular_trees.__post_init__()

    def test_alpha_not_float_exception(self, xgb_diabetes_model_trees_dataframe):
        """Test an exception is raised if alpha is not a float."""
        tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        tabular_trees.alpha = "1"

        with pytest.raises(
            TypeError,
            match="alpha is not in expected types <class 'float'>, got <class 'str'>",
        ):
            tabular_trees.__post_init__()

    def test_alpha_not_zero_exception(self, xgb_diabetes_model_trees_dataframe):
        """Test an exception is raised if trees is not a pd.DataFrame."""
        tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        tabular_trees.alpha = 1.0

        with pytest.raises(
            ValueError,
            match=re.escape("condition: [alpha = 0] not met"),
        ):
            tabular_trees.__post_init__()

    def test_super_post_init_called(self, mocker, xgb_diabetes_model_trees_dataframe):
        """Test that BaseModelTabularTrees.__post_init__ method is called."""
        # initialise object then overwrite trees attribute with data that does
        # not contain the stats columns
        tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        mocked = mocker.patch.object(BaseModelTabularTrees, "__post_init__")

        tabular_trees.__post_init__()

        assert mocked.call_count == 1, (
            "BaseModelTabularTrees.__post_init__ not called when "
            "XGBoostTabularTrees.__post_init__ runs"
        )

        assert (
            mocked.call_args_list[0][0] == ()
        ), "positional args in BaseModelTabularTrees.__post_init__ call not correct"

        assert (
            mocked.call_args_list[0][1] == {}
        ), "keyword args in BaseModelTabularTrees.__post_init__ call not correct"

    def test_trees_attribute_updated(self, mocker, xgb_diabetes_model_trees_dataframe):
        """Test trees attribute is updated with output from derive_predictions."""
        tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        derive_predictions_output = 0

        # mock derive_predictions to set return value
        mocker.patch.object(
            XGBoostTabularTrees,
            "derive_predictions",
            return_value=derive_predictions_output,
        )

        # mock __post_init__ so it does nothing when called
        mocker.patch.object(BaseModelTabularTrees, "__post_init__")

        assert type(tabular_trees.trees) is not type(derive_predictions_output)

        tabular_trees.__post_init__()

        assert (
            tabular_trees.trees == derive_predictions_output
        ), "trees attribute not updated with the output from derive_predictions"


class TestConvertToTabularTrees:
    """Tests for the XGBoostTabularTrees.convert_to_tabular_trees method."""

    def test_output_type(self, xgb_diabetes_model_trees_dataframe):
        """Test the output from convert_to_tabular_trees is a TabularTrees object."""
        xgboost_tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        output = xgboost_tabular_trees.convert_to_tabular_trees()

        assert (
            type(output) is TabularTrees
        ), "output from convert_to_tabular_trees is not a TabularTrees object"
