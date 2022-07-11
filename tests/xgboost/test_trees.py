import pandas as pd
import pytest
import re

import tabular_trees.xgboost.trees as trees


class TestXGBoostTabularTreesInit:
    """Tests for the XGBoostTabularTrees.__init__ method."""

    def test_successfull_call(self, xgb_diabetes_model_trees_dataframe):
        """Test successful initialisation of the XGBoostTabularTrees class."""

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


class TestXGBoostTabularTreesGetTrees:
    """Tests for the XGBoostTabularTrees.get_trees method."""

    def test_successfull_call(self, xgb_diabetes_model_trees_dataframe):
        """Test successful initialisation of the XGBoostTabularTrees class."""

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
