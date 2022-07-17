import pandas as pd
import numpy as np
import pytest
import re

import tabular_trees.lightgbm.trees as trees


class TestLightGBMTabularTreesInit:
    """Tests for the LightGBMTabularTrees.__init__ method."""

    def test_successfull_call(self, lgb_diabetes_model_trees_dataframe):
        """Test successfull initialisation of the LightGBMTabularTrees class."""

        trees.LightGBMTabularTrees(lgb_diabetes_model_trees_dataframe)

    def test_trees_not_dataframe_exception(self):
        """Test an exception is raised if trees is not a pd.DataFrame."""

        with pytest.raises(
            TypeError,
            match="trees is not in expected types <class 'pandas.core.frame.DataFrame'>, got <class 'int'>",
        ):

            trees.LightGBMTabularTrees(12345)

    @pytest.mark.parametrize(
        "drop_columns",
        [(["node_index"]), (["parent_index"]), (["split_gain", "threshold"])],
    )
    def test_missing_columns_exception(
        self, lgb_diabetes_model_trees_dataframe, drop_columns
    ):
        """Test an exception is raised if columns from REQUIRED_COLUMNS are
        missing in trees."""

        dropped_columns = lgb_diabetes_model_trees_dataframe.drop(columns=drop_columns)

        with pytest.raises(
            ValueError,
            match=re.escape(f"expected columns not in df; {sorted(drop_columns)}"),
        ):

            trees.LightGBMTabularTrees(dropped_columns)

    def test_n_trees_set(self, lgb_diabetes_model_trees_dataframe):
        """Test the n_trees attribute is set to the correct value."""

        lightgbm_tabular_trees = trees.LightGBMTabularTrees(
            lgb_diabetes_model_trees_dataframe
        )

        assert (
            lightgbm_tabular_trees.n_trees
            == lgb_diabetes_model_trees_dataframe["tree_index"].max()
        ), "n_trees attribute not set correctly"

    def test_trees_column_order(self, lgb_diabetes_model_trees_dataframe):
        """Test that the columns in the trees attribute are in the order of
        REQUIRED_COLUMNS."""

        wrong_order_columns = [
            "tree_index",
            "node_depth",
            "weight",
            "left_child",
            "missing_type",
            "right_child",
            "parent_index",
            "split_feature",
            "split_gain",
            "threshold",
            "decision_type",
            "missing_direction",
            "node_index",
            "value",
            "count",
        ]

        lightgbm_tabular_trees = trees.LightGBMTabularTrees(
            lgb_diabetes_model_trees_dataframe[wrong_order_columns]
        )

        assert (
            lightgbm_tabular_trees.trees.columns.to_list()
            == lightgbm_tabular_trees.REQUIRED_COLUMNS
        ), "trees attribute columns in wrong order"

    def test_trees_sorted(self, lgb_diabetes_model_trees_dataframe):
        """Test that the trees attribute is sorted by tree_index, node_depth
        and node_index columns."""

        input_df = lgb_diabetes_model_trees_dataframe.copy()

        # sort into correct order and reset index to that order
        input_df = input_df.sort_values(
            ["tree_index", "node_depth", "node_index"]
        ).reset_index(drop=True)

        # sort into incorrect order and don't reset index
        input_df = input_df.sort_values(["split_gain", "threshold"])

        lightgbm_tabular_trees = trees.LightGBMTabularTrees(input_df)

        pd.testing.assert_frame_equal(
            input_df.sort_values(["tree_index", "node_depth", "node_index"]),
            lightgbm_tabular_trees.trees,
        )

    def test_trees_index_reset(self, lgb_diabetes_model_trees_dataframe):
        """Test that the index on trees attribute is reset."""

        input_df = lgb_diabetes_model_trees_dataframe.copy()

        input_df.index = [0] * input_df.shape[0]

        lightgbm_tabular_trees = trees.LightGBMTabularTrees(input_df)

        np.testing.assert_array_equal(
            lightgbm_tabular_trees.trees.index.values,
            np.array([i for i in range(input_df.shape[0])]),
        )
