import re

import pandas as pd
import pytest
import xgboost as xgb

from tabular_trees.trees import BaseModelTabularTrees, TabularTrees
from tabular_trees.xgboost import ParsedXGBoostTabularTrees, XGBoostTabularTrees


class TestXGBoostTabularTreesInit:
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


class TestXGBoostTabularTreesPostInit:
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


class TestXGBoostTabularTreesDerivePredictions:
    """Tests for the XGBoostTabularTrees.derive_predictions method."""

    @pytest.mark.parametrize("lambda_", [(0.0), (2.0)])
    def test_predictions_calculated_correctly(self, lambda_, xgb_diabetes_dmatrix):
        """Test that the derived node prediction values are correct."""

        def derive_depths(df) -> pd.DataFrame:
            """Derive node depth for all nodes.

            Returns
            -------
            pd.DataFrame
                Tree data (trees attribute) with 'Depth' column added.

            """
            if not (df.groupby("Tree")["Node"].first() == 0).all():
                raise ValueError("first node by tree must be the root node (0)")

            df["Depth"] = 0

            for row_number in range(df.shape[0]):
                row = df.iloc[row_number]

                # for non-leaf nodes, increase child node depths by 1
                if row["Feature"] != "Leaf":
                    df.loc[df["ID"] == row["Yes"], "Depth"] = row["Depth"] + 1
                    df.loc[df["ID"] == row["No"], "Depth"] = row["Depth"] + 1

            return df

        model_for_predictions = xgb.train(
            params={"verbosity": 0, "max_depth": 3, "lambda": lambda_},
            dtrain=xgb_diabetes_dmatrix,
            num_boost_round=10,
        )

        trees_data = model_for_predictions.trees_to_dataframe()

        xgboost_tabular_trees = XGBoostTabularTrees(trees_data, lambda_)

        predictions = xgboost_tabular_trees.derive_predictions()

        depths = derive_depths(xgboost_tabular_trees.trees.copy())

        predictions["Depth"] = depths["Depth"]

        # loop through internal nodes, non-root nodes
        for row_number in range(predictions.shape[0]):
            row = predictions.iloc[row_number]

            if (row["Feature"] != "Leaf") and (row["Node"] > 0):
                # build model with required number of trees and depth of the
                # current node, so in this tree the node is a leaf node
                if row["Tree"] == 0:
                    model = xgb.train(
                        params={
                            "verbosity": 0,
                            "max_depth": row["Depth"],
                            "lambda": lambda_,
                        },
                        dtrain=xgb_diabetes_dmatrix,
                        num_boost_round=row["Tree"] + 1,
                    )

                # if the number of trees required is > 1 then build the first n - 1
                # trees at the maximum depth, then build the last tree at the depth
                # of the current node
                else:
                    model_n = xgb.train(
                        params={
                            "verbosity": 0,
                            "max_depth": predictions["Depth"].max(),
                            "lambda": lambda_,
                        },
                        dtrain=xgb_diabetes_dmatrix,
                        num_boost_round=row["Tree"],
                    )

                    model = xgb.train(
                        params={
                            "verbosity": 0,
                            "max_depth": row["Depth"],
                            "lambda": lambda_,
                        },
                        dtrain=xgb_diabetes_dmatrix,
                        num_boost_round=1,
                        xgb_model=model_n,
                    )

                model_trees = model.trees_to_dataframe()

                round_to_digits = 4

                derived_prediction = round(row["weight"], round_to_digits)
                prediction_from_leaf_node = round(
                    model_trees.loc[model_trees["ID"] == row["ID"], "Gain"].item(),
                    round_to_digits,
                )

                assert derived_prediction == prediction_from_leaf_node, (
                    f"""derived internal node prediction for node {row["ID"]} """
                    "incorrect (rounding to 3dp)"
                )


class TestXGBoostTabularTreesConvert:
    """Tests for the XGBoostTabularTrees.convert_to_tabular_trees method."""

    def test_output_type(self, xgb_diabetes_model_trees_dataframe):
        """Test the output from convert_to_tabular_trees is a TabularTrees object."""
        xgboost_tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        output = xgboost_tabular_trees.convert_to_tabular_trees()

        assert (
            type(output) is TabularTrees
        ), "output from convert_to_tabular_trees is not a TabularTrees object"


class TestParsedXGBoostTabularTreesInit:
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


class TestParsedXGBoostTabularTreesPostInit:
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


class TestParsedXGBoostTabularTreesConvert:
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
