import re

import pandas as pd
import pytest
import xgboost as xgb

from tabular_trees.trees import BaseModelTabularTrees, TabularTrees
from tabular_trees.xgboost import ParsedXGBoostTabularTrees, XGBoostTabularTrees


class TestXGBoostTabularTreesInit:
    """Tests for the XGBoostTabularTrees.__init__ method."""

    def test_successfull_call(self, xgb_diabetes_model_trees_dataframe):
        """Test successfull initialisation of the XGBoostTabularTrees class."""

        XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

    def test_inheritance(self):
        """Test that XGBoostTabularTrees inherits from BaseModelTabularTrees."""

        assert (
            XGBoostTabularTrees.__mro__[1] is BaseModelTabularTrees
        ), "XGBoostTabularTrees does not inherit from BaseModelTabularTrees"

    @pytest.mark.parametrize(
        "attribute_name,expected_value",
        [
            ("SORT_BY_COLUMNS", ["Tree", "Node"]),
            (
                "REQUIRED_COLUMNS",
                [
                    "Tree",
                    "Node",
                    "ID",
                    "Feature",
                    "Split",
                    "Yes",
                    "No",
                    "Missing",
                    "Gain",
                    "Cover",
                    "Category",
                    "G",
                    "H",
                    "weight",
                ],
            ),
        ],
    )
    def test_column_attributes(
        self, attribute_name, expected_value, xgb_diabetes_model_trees_dataframe
    ):
        """Test column related attributes are set as expected."""

        assert (
            getattr(XGBoostTabularTrees, attribute_name) == expected_value
        ), f"{attribute_name} not expected on XGBoostTabularTrees class"

        tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        assert (
            getattr(
                tabular_trees,
                attribute_name,
            )
            == expected_value
        ), f"{attribute_name} not expected on XGBoostTabularTrees object after initialisation"

    def test_trees_not_same_object(self, xgb_diabetes_model_trees_dataframe):
        """Test the trees attribute is not the same object as that passed into
        the init method."""

        input_df = xgb_diabetes_model_trees_dataframe.copy()

        tabular_trees = XGBoostTabularTrees(input_df)

        assert id(tabular_trees.trees) != id(
            input_df
        ), "trees attribute is the same object as passed into initialisation"


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

        assert (
            mocked.call_count == 1
        ), "BaseModelTabularTrees.__post_init__ not called when XGBoostTabularTrees.__post_init__ runs"

        assert (
            mocked.call_args_list[0][0] == ()
        ), "positional args in BaseModelTabularTrees.__post_init__ call not correct"

        assert (
            mocked.call_args_list[0][1] == {}
        ), "keyword args in BaseModelTabularTrees.__post_init__ call not correct"

    def test_trees_attribute_updated(self, mocker, xgb_diabetes_model_trees_dataframe):
        """Test the trees attribute is updated with the output from derive_predictions."""

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

        assert type(tabular_trees.trees) != type(derive_predictions_output)

        tabular_trees.__post_init__()

        assert (
            tabular_trees.trees == derive_predictions_output
        ), "trees attribute not updated with the output from derive_predictions"


class TestXGBoostTabularTreesDerivePredictions:
    """Tests for the XGBoostTabularTrees.derive_predictions method."""

    def test_successfull_call(self, xgb_diabetes_model_trees_dataframe):
        """Test successfull call of the derive_predictions method."""

        xgboost_tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        xgboost_tabular_trees.derive_predictions()

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

                round_to_digits = 3

                derived_prediction = round(row["weight"], round_to_digits)
                prediction_from_leaf_node = round(
                    model_trees.loc[model_trees["ID"] == row["ID"], "Gain"].item(),
                    round_to_digits,
                )

                assert (
                    derived_prediction == prediction_from_leaf_node
                ), f"""derived internal node prediction for node {row["ID"]} incorrect (rounding to 3dp)"""


class TestXGBoostTabularTreesConvert:
    """Tests for the XGBoostTabularTrees.convert_to_tabular_trees method."""

    def test_successful_call(self, xgb_diabetes_model_trees_dataframe):
        """Test a successful call to convert_to_tabular_trees."""

        xgboost_tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        output = xgboost_tabular_trees.convert_to_tabular_trees()

        assert (
            type(output) is TabularTrees
        ), "output from XGBoostTabularTrees.convert_to_tabular_trees is not TabularTrees type"


class TestParsedXGBoostTabularTreesInit:
    """Tests for the ParsedXGBoostTabularTrees.__init__ method."""

    def test_successfull_call(self, xgb_diabetes_model_parsed_trees_dataframe):
        """Test successfull initialisation of the ParsedXGBoostTabularTrees class."""

        ParsedXGBoostTabularTrees(xgb_diabetes_model_parsed_trees_dataframe)

    def test_inheritance(self):
        """Test that ParsedXGBoostTabularTrees inherits from BaseModelTabularTrees."""

        assert (
            ParsedXGBoostTabularTrees.__mro__[1] is BaseModelTabularTrees
        ), "ParsedXGBoostTabularTrees does not inherit from BaseModelTabularTrees"

    @pytest.mark.parametrize(
        "attribute_name,expected_value",
        [
            ("SORT_BY_COLUMNS", ["tree", "nodeid"]),
            (
                "REQUIRED_COLUMNS",
                [
                    "tree",
                    "nodeid",
                    "depth",
                    "yes",
                    "no",
                    "missing",
                    "split",
                    "split_condition",
                    "leaf",
                    "gain",
                    "cover",
                ],
            ),
            ("STATS_COLUMNS", ["gain", "cover"]),
            (
                "COLUMNS_MAPPING",
                {
                    "tree": "Tree",
                    "nodeid": "Node",
                    "yes": "Yes",
                    "no": "No",
                    "missing": "Missing",
                    "split": "Feature",
                    "split_condition": "Split",
                    "gain": "Gain",
                    "cover": "Cover",
                },
            ),
        ],
    )
    def test_column_attributes(
        self, attribute_name, expected_value, xgb_diabetes_model_parsed_trees_dataframe
    ):
        """Test column related attributes are set as expected."""

        assert (
            getattr(ParsedXGBoostTabularTrees, attribute_name) == expected_value
        ), f"{attribute_name} not expected on ParsedXGBoostTabularTrees class"

        tabular_trees = ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe
        )

        assert (
            getattr(
                tabular_trees,
                attribute_name,
            )
            == expected_value
        ), f"{attribute_name} not expected on XGBoostTabularTrees object after initialisation"

    def test_trees_not_same_object(self, xgb_diabetes_model_parsed_trees_dataframe):
        """Test the trees attribute is not the same object as that passed into
        the init method."""

        input_df = xgb_diabetes_model_parsed_trees_dataframe.copy()

        tabular_trees = ParsedXGBoostTabularTrees(input_df)

        assert id(tabular_trees.trees) != id(
            input_df
        ), "trees attribute is the same object as passed into initialisation"


class TestParsedXGBoostTabularTreesPostInit:
    """Tests for the ParsedXGBoostTabularTrees.__post_init__ method."""

    def test_exception_if_no_stats(
        self, xgb_diabetes_model_parsed_trees_dataframe_no_stats
    ):
        """Test a ValueError is raised if convert_to_xgboost_tabular_trees is
        called when the tree data does not have cover and gain columns."""

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

        assert (
            mocked.call_count == 1
        ), "BaseModelTabularTrees.__post_init__ not called when ParsedXGBoostTabularTrees.__post_init__ runs"

        assert (
            mocked.call_args_list[0][0] == ()
        ), "positional args in BaseModelTabularTrees.__post_init__ call not correct"

        assert (
            mocked.call_args_list[0][1] == {}
        ), "keyword args in BaseModelTabularTrees.__post_init__ call not correct"


class TestParsedXGBoostTabularTreesConvert:
    """Tests for the ParsedXGBoostTabularTrees.convert_to_xgboost_tabular_trees method."""

    def test_successfull_call(self, xgb_diabetes_model_parsed_trees_dataframe):
        """Test successfull call of the convert_to_xgboost_tabular_trees method."""

        parsed_tabular_trees = ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe
        )

        parsed_tabular_trees.convert_to_xgboost_tabular_trees()

    def test_output_same_format_as_xgboost(
        self,
        xgb_diabetes_model_trees_dataframe,
        xgb_diabetes_model_parsed_trees_dataframe,
    ):
        """Test that the output from the function is the same as the output
        from xgb.Booster.trees_to_dataframe."""

        expected_output = xgb_diabetes_model_trees_dataframe

        parsed_tabular_trees = ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe
        )

        xgboost_tabular_trees = parsed_tabular_trees.convert_to_xgboost_tabular_trees()

        pd.testing.assert_frame_equal(
            xgboost_tabular_trees.trees.drop(columns=["weight", "G", "H"]),
            expected_output,
        )
