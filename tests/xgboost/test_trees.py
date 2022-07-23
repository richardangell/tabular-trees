import pandas as pd
import xgboost as xgb
import pytest
import re

from tabular_trees.trees import BaseModelTabularTrees
from tabular_trees.xgboost.trees import XGBoostTabularTrees, ParsedXGBoostTabularTrees


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


class TestXGBoostTabularTreesPostPostInit:
    """Tests for the XGBoostTabularTrees.__post_post_init__ method."""

    def test_lambda_not_float_exception(self, xgb_diabetes_model_trees_dataframe):
        """Test an exception is raised if lambda_ is not a float."""

        with pytest.raises(
            TypeError,
            match="lambda_ is not in expected types <class 'float'>, got <class 'str'>",
        ):

            XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe, lambda_="1")

    def test_alpha_not_float_exception(self, xgb_diabetes_model_trees_dataframe):
        """Test an exception is raised if alpha is not a float."""

        with pytest.raises(
            TypeError,
            match="alpha is not in expected types <class 'float'>, got <class 'str'>",
        ):

            XGBoostTabularTrees(
                xgb_diabetes_model_trees_dataframe, lambda_=1.0, alpha="1"
            )

    def test_alpha_not_zero_exception(self, xgb_diabetes_model_trees_dataframe):
        """Test an exception is raised if trees is not a pd.DataFrame."""

        with pytest.raises(
            ValueError,
            match=re.escape("condition: [alpha = 0] not met"),
        ):

            XGBoostTabularTrees(
                xgb_diabetes_model_trees_dataframe, lambda_=1.0, alpha=1.0
            )


class TestXGBoostTabularTreesDerivePredictions:
    """Tests for the XGBoostTabularTrees.derive_predictions method."""

    def test_successfull_call(self, xgb_diabetes_model_trees_dataframe):
        """Test successfull call of the derive_predictions method."""

        xgboost_tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        xgboost_tabular_trees.derive_predictions()

    @pytest.mark.parametrize("lambda_", [(0.0), (2.0)])
    def test_predictions_calculated_correctly(self, lambda_, xgb_diabetes_dmatrix):
        """Test that the derived node prediction values are correct."""

        model_for_predictions = xgb.train(
            params={"verbosity": 0, "max_depth": 3, "lambda": lambda_},
            dtrain=xgb_diabetes_dmatrix,
            num_boost_round=10,
        )

        trees_data = model_for_predictions.trees_to_dataframe()

        xgboost_tabular_trees = XGBoostTabularTrees(trees_data, lambda_)

        predictions = xgboost_tabular_trees.derive_predictions()

        depths = xgboost_tabular_trees.derive_depths()

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


class TestXGBoostTabularTreesDeriveDepths:
    """Tests for the XGBoostTabularTrees.derive_depths method."""

    def test_successfull_call(self, xgb_diabetes_model_trees_dataframe):
        """Test successfull call of the derive_depths method."""

        xgboost_tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        xgboost_tabular_trees.derive_depths()

    def test_first_node_by_tree_not_root_exception(
        self, xgb_diabetes_model_trees_dataframe
    ):
        """Test that a ValueError is raised if the first node by tree is not
        a root node."""

        xgboost_tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        tree_structure = pd.DataFrame(
            {
                "Tree": [0, 0, 0, 1, 1, 1],
                "Node": [0, 1, 2, 1, 0, 2],
                "Yes": ["0-1", "0-", "0-", "1-", "1-1", "1-"],
                "No": ["0-2", "0-", "0-", "1-", "1-2", "1-"],
                "Feature": ["", "Leaf", "Leaf", "Leaf", "", "Leaf"],
                "Depth": [0, 1, 1, 1, 0, 1],
            }
        )

        tree_structure["ID"] = (
            tree_structure["Tree"].astype(str)
            + "-"
            + tree_structure["Node"].astype(str)
        )

        xgboost_tabular_trees.trees = tree_structure

        with pytest.raises(
            ValueError, match=re.escape("first node by tree must be the root node (0)")
        ):

            xgboost_tabular_trees.derive_depths()

    def test_depth_calculated_correctly(self, xgb_diabetes_model_trees_dataframe):
        """Test that depth values are calculated correctly for a single tree."""

        xgboost_tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        tree_structure = pd.DataFrame(
            {
                "Tree": [0, 0, 0, 0, 0, 0, 0],
                "Node": [0, 1, 2, 3, 4, 5, 6],
                "Yes": ["0-1", "0-", "0-3", "0-5", "0-", "0-", "0-"],
                "No": ["0-2", "0-", "0-4", "0-6", "0-", "0-", "0-"],
                "Feature": ["", "Leaf", "", "", "Leaf", "Leaf", "Leaf"],
                "Depth": [0, 1, 1, 2, 2, 3, 3],
            }
        )

        tree_structure["ID"] = (
            tree_structure["Tree"].astype(str)
            + "-"
            + tree_structure["Node"].astype(str)
        )

        # overwrite tree data
        xgboost_tabular_trees.trees = tree_structure

        derived_depths = xgboost_tabular_trees.derive_depths()

        pd.testing.assert_frame_equal(derived_depths, tree_structure)

    def test_depth_calculated_correctly_multi_tree(
        self, xgb_diabetes_model_trees_dataframe
    ):
        """Test that depth values are calculated correctly for a multiple
        trees."""

        xgboost_tabular_trees = XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

        tree_structure = pd.DataFrame(
            {
                "Tree": [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                "Node": [0, 1, 2, 3, 4, 5, 6, 0, 2, 1],
                "Yes": ["0-1", "0-", "0-3", "0-5", "0-", "0-", "0-", "1-2", "1-", "1-"],
                "No": ["0-2", "0-", "0-4", "0-6", "0-", "0-", "0-", "1-1", "1-", "1-"],
                "Feature": [
                    "",
                    "Leaf",
                    "",
                    "",
                    "Leaf",
                    "Leaf",
                    "Leaf",
                    "",
                    "Leaf",
                    "Leaf",
                ],
                "Depth": [0, 1, 1, 2, 2, 3, 3, 0, 1, 1],
            }
        )

        tree_structure["ID"] = (
            tree_structure["Tree"].astype(str)
            + "-"
            + tree_structure["Node"].astype(str)
        )

        # overwrite tree data
        xgboost_tabular_trees.trees = tree_structure

        derived_depths = xgboost_tabular_trees.derive_depths()

        pd.testing.assert_frame_equal(derived_depths, tree_structure)


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
            "Cannot create ParsedXGBoostTabularTrees object unless statistics"
            " are output. Rerun dump_model with with_stats = True."
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

        pd.testing.assert_frame_equal(xgboost_tabular_trees.trees, expected_output)
