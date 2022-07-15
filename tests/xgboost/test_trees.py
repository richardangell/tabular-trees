import pandas as pd
import pytest
import re

import tabular_trees.xgboost.trees as trees


class TestXGBoostTabularTreesInit:
    """Tests for the XGBoostTabularTrees.__init__ method."""

    def test_successfull_call(self, xgb_diabetes_model_trees_dataframe):
        """Test successfull initialisation of the XGBoostTabularTrees class."""

        trees.XGBoostTabularTrees(xgb_diabetes_model_trees_dataframe)

    def test_trees_not_dataframe_exception(self):
        """Test an exception is raised if trees is not a pd.DataFrame."""

        with pytest.raises(
            TypeError,
            match="trees is not in expected types <class 'pandas.core.frame.DataFrame'>, got <class 'int'>",
        ):

            trees.XGBoostTabularTrees(12345)

    @pytest.mark.parametrize(
        "drop_columns", [(["Yes"]), (["No"]), (["Yes", "No", "Missing"])]
    )
    def test_missing_columns_exception(
        self, xgb_diabetes_model_trees_dataframe, drop_columns
    ):
        """Test an exception is raised if columns from REQUIRED_COLUMNS are
        missing in trees."""

        dropped_columns = xgb_diabetes_model_trees_dataframe.drop(columns=drop_columns)

        with pytest.raises(
            ValueError,
            match=re.escape(f"expected columns not in df; {sorted(drop_columns)}"),
        ):

            trees.XGBoostTabularTrees(dropped_columns)

    def test_trees_attribute(self, xgb_diabetes_model_trees_dataframe):
        """Test that the trees argument is set as the attribute of the same
        name, with columns sorted."""

        reversed_columns = [
            x for x in reversed(trees.XGBoostTabularTrees.REQUIRED_COLUMNS)
        ]

        # sort the columns into reversed order
        df_reversed = xgb_diabetes_model_trees_dataframe[reversed_columns]

        xgb_tabular_trees = trees.XGBoostTabularTrees(df_reversed)

        pd.testing.assert_frame_equal(
            xgb_tabular_trees.trees,
            xgb_diabetes_model_trees_dataframe[
                trees.XGBoostTabularTrees.REQUIRED_COLUMNS
            ],
        )

    def test_n_trees_set(self, xgb_diabetes_model_trees_dataframe):
        """Test the n_trees attribute is set to the correct value."""

        xgboost_tabular_trees = trees.XGBoostTabularTrees(
            xgb_diabetes_model_trees_dataframe
        )

        assert (
            xgboost_tabular_trees.n_trees
            == xgb_diabetes_model_trees_dataframe["Tree"].max()
        ), "n_trees attribute not set correctly"

    def test_trees_column_order(self, xgb_diabetes_model_trees_dataframe):
        """Test that the columns in the trees attribute are in the order of
        REQUIRED_COLUMNS."""

        wrong_order_columns = [
            "Missing",
            "Node",
            "Split",
            "ID",
            "Feature",
            "Tree",
            "Yes",
            "No",
            "Category",
            "Gain",
            "Cover",
        ]

        xgboost_tabular_trees = trees.XGBoostTabularTrees(
            xgb_diabetes_model_trees_dataframe[wrong_order_columns]
        )

        assert (
            xgboost_tabular_trees.trees.columns.to_list()
            == xgboost_tabular_trees.REQUIRED_COLUMNS
        ), "trees attribute columns in wrong order"

    def test_trees_sorted(self, xgb_diabetes_model_trees_dataframe):
        """Test that the trees attribute is sorted by Tree and Node columns."""

        input_df = xgb_diabetes_model_trees_dataframe.copy().sort_values(
            ["Gain", "Cover"]
        )

        xgboost_tabular_trees = trees.XGBoostTabularTrees(input_df)

        pd.testing.assert_frame_equal(
            input_df.sort_values(["Tree", "Node"]), xgboost_tabular_trees.trees
        )


class TestXGBoostTabularTreesGetTrees:
    """Tests for the XGBoostTabularTrees.get_trees method."""

    def test_successfull_call(self, xgb_diabetes_model_trees_dataframe):
        """Test successfull initialisation of the XGBoostTabularTrees class."""

        xgboost_tabular_trees = trees.XGBoostTabularTrees(
            xgb_diabetes_model_trees_dataframe
        )

        xgboost_tabular_trees.get_trees([0, 1, 2, 5, 9])

    @pytest.mark.parametrize(
        "tree_indexes,exception,text",
        [
            (
                [0, 1, "abc", 9],
                TypeError,
                re.escape(
                    "tree_indexes[2] is not in expected types <class 'int'>, got <class 'str'>"
                ),
            ),
            (
                [-1, 0, 1],
                ValueError,
                re.escape("condition: [tree_indexes[0] >= 0] not met"),
            ),
            (
                [0, 1, 5, 10],
                ValueError,
                re.escape(
                    "condition: [tree_indexes[3] in range for number of trees (9)] not met"
                ),
            ),
        ],
    )
    def test_trees_non_integer_exception(
        self, tree_indexes, exception, text, xgb_diabetes_model_trees_dataframe
    ):
        """Test the correct exception is raised if tree_indexes arg is not in
        the correct format."""

        xgboost_tabular_trees = trees.XGBoostTabularTrees(
            xgb_diabetes_model_trees_dataframe
        )

        with pytest.raises(exception, match=text):

            xgboost_tabular_trees.get_trees(tree_indexes)

    @pytest.mark.parametrize("tree_indexes", [([0]), ([0, 1]), ([2, 3, 9])])
    def test_correct_trees_returned(
        self, tree_indexes, xgb_diabetes_model_trees_dataframe
    ):
        """Test that the correct rows are returned when get_trees is called."""

        xgboost_tabular_trees = trees.XGBoostTabularTrees(
            xgb_diabetes_model_trees_dataframe
        )

        expected = xgb_diabetes_model_trees_dataframe.loc[
            xgb_diabetes_model_trees_dataframe["Tree"].isin(tree_indexes)
        ]

        result = xgboost_tabular_trees.get_trees(tree_indexes)

        pd.testing.assert_frame_equal(result, expected)


class TestParsedXGBoostTabularTreesInit:
    """Tests for the ParsedXGBoostTabularTrees.__init__ method."""

    def test_successfull_call(self, xgb_diabetes_model_parsed_trees_dataframe):
        """Test successfull initialisation of the ParsedXGBoostTabularTrees class."""

        trees.ParsedXGBoostTabularTrees(xgb_diabetes_model_parsed_trees_dataframe)

    def test_has_stats_attribute_true(self, xgb_diabetes_model_parsed_trees_dataframe):
        """Test the has_stats attribute is True if cover and gain are present
        in the tree data."""

        parsed_tabular_trees = trees.ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe
        )

        assert "gain" in xgb_diabetes_model_parsed_trees_dataframe.columns.values
        assert "cover" in xgb_diabetes_model_parsed_trees_dataframe.columns.values

        assert (
            parsed_tabular_trees.has_stats
        ), "has_stats not set to True when gain and cover are in the tree data"

    def test_has_stats_attribute_false(
        self, xgb_diabetes_model_parsed_trees_dataframe_no_stats
    ):
        """Test the has_stats attribute is False if cover and gain are not
        present in the tree data."""

        parsed_tabular_trees = trees.ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe_no_stats
        )

        assert (
            "gain"
            not in xgb_diabetes_model_parsed_trees_dataframe_no_stats.columns.values
        )
        assert (
            "cover"
            not in xgb_diabetes_model_parsed_trees_dataframe_no_stats.columns.values
        )

        assert (
            not parsed_tabular_trees.has_stats
        ), "has_stats not set to False when gain and cover are in the tree data"

    @pytest.mark.parametrize(
        "tree_data_fixture_name",
        [
            ("xgb_diabetes_model_parsed_trees_dataframe_no_stats"),
            ("xgb_diabetes_model_parsed_trees_dataframe"),
        ],
    )
    @pytest.mark.parametrize(
        "missing_column",
        [col for col in trees.ParsedXGBoostTabularTrees.REQUIRED_BASE_COLUMNS],
    )
    def test_required_columns_not_present_exception(
        self, request, missing_column, tree_data_fixture_name
    ):
        """Test that if the REQUIRED_BASE_COLUMNS are not present in tree data
        then an exception is raised."""

        tree_data = request.getfixturevalue(tree_data_fixture_name)

        with pytest.raises(
            ValueError,
            match=re.escape(f"expected columns not in df; ['{missing_column}']"),
        ):

            trees.ParsedXGBoostTabularTrees(tree_data.drop(columns=[missing_column]))

    @pytest.mark.parametrize(
        "tree_data_fixture_name,column_order",
        [
            (
                "xgb_diabetes_model_parsed_trees_dataframe_no_stats",
                trees.ParsedXGBoostTabularTrees.REQUIRED_BASE_COLUMNS,
            ),
            (
                "xgb_diabetes_model_parsed_trees_dataframe",
                trees.ParsedXGBoostTabularTrees.REQUIRED_BASE_COLUMNS
                + trees.ParsedXGBoostTabularTrees.STATS_COLUMNS,
            ),
        ],
    )
    def test_trees_sorted(self, request, tree_data_fixture_name, column_order):
        """Test that the trees attribute is sorted by tree and nodeid columns."""

        tree_data = request.getfixturevalue(tree_data_fixture_name)

        # set the correct column order beforehand, so only the row ordering
        # is varying
        tree_data = tree_data[column_order]

        tree_data = tree_data.copy().sort_values(["split_condition"])

        parsed_tabular_trees = trees.ParsedXGBoostTabularTrees(tree_data)

        pd.testing.assert_frame_equal(
            tree_data.sort_values(["tree", "nodeid"]), parsed_tabular_trees.trees
        )

    @pytest.mark.parametrize(
        "tree_data_fixture_name,column_order",
        [
            (
                "xgb_diabetes_model_parsed_trees_dataframe_no_stats",
                trees.ParsedXGBoostTabularTrees.REQUIRED_BASE_COLUMNS,
            ),
            (
                "xgb_diabetes_model_parsed_trees_dataframe",
                trees.ParsedXGBoostTabularTrees.REQUIRED_BASE_COLUMNS
                + trees.ParsedXGBoostTabularTrees.STATS_COLUMNS,
            ),
        ],
    )
    def test_column_order(self, request, tree_data_fixture_name, column_order):
        """Test that the trees attribute columns are in the correct order."""

        tree_data = request.getfixturevalue(tree_data_fixture_name)

        # set the correct row order beforehand, so only the column ordering
        # is varying
        tree_data = tree_data.sort_values(["tree", "nodeid"])

        assert (
            tree_data.columns.values.tolist() != column_order
        ), "tree_data columns are already in correct order"

        parsed_tabular_trees = trees.ParsedXGBoostTabularTrees(tree_data)

        pd.testing.assert_frame_equal(
            tree_data[column_order], parsed_tabular_trees.trees
        )


class TestParsedXGBoostTabularTreesConvert:
    """Tests for the ParsedXGBoostTabularTrees.convert_to_xgboost_tabular_trees method."""

    def test_successfull_call(self, xgb_diabetes_model_parsed_trees_dataframe):
        """Test successfull call of the convert_to_xgboost_tabular_trees method."""

        parsed_tabular_trees = trees.ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe
        )

        parsed_tabular_trees.convert_to_xgboost_tabular_trees()

    def test_exception_if_no_stats(
        self, xgb_diabetes_model_parsed_trees_dataframe_no_stats
    ):
        """Test a ValueError is raised if convert_to_xgboost_tabular_trees is
        called when the tree data does not have cover and gain columns."""

        parsed_tabular_trees = trees.ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe_no_stats
        )

        expected_exception_message = (
            "Cannot convert to XGBoostTabularTrees class unless statistics"
            " are output. Rerun dump_model with with_stats = True."
        )

        with pytest.raises(ValueError, match=expected_exception_message):

            parsed_tabular_trees.convert_to_xgboost_tabular_trees()

    def test_output_same_format_as_xgboost(
        self,
        xgb_diabetes_model_trees_dataframe,
        xgb_diabetes_model_parsed_trees_dataframe,
    ):
        """Test that the output from the function is the same as the output
        from xgb.Booster.trees_to_dataframe."""

        expected_output = xgb_diabetes_model_trees_dataframe

        parsed_tabular_trees = trees.ParsedXGBoostTabularTrees(
            xgb_diabetes_model_parsed_trees_dataframe
        )

        xgboost_tabular_trees = parsed_tabular_trees.convert_to_xgboost_tabular_trees()

        pd.testing.assert_frame_equal(xgboost_tabular_trees.trees, expected_output)
