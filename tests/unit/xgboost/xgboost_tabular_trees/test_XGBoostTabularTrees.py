import pandas as pd
import pytest

from tabular_trees.trees import TabularTrees
from tabular_trees.xgboost.xgboost_tabular_trees import XGBoostTabularTrees


class TestInitialisation:
    """Tests for the XGBoostTabularTrees.__init__ method."""

    def test_trees_attribute_set(
        self, xgb_diabetes_model, xgb_diabetes_model_trees_dataframe
    ):
        """Test the trees attribute is set as the value passed in init."""
        tabular_trees = XGBoostTabularTrees.from_booster(xgb_diabetes_model)

        pd.testing.assert_frame_equal(
            tabular_trees.data.drop(columns=["G", "H", "weight"]),
            xgb_diabetes_model_trees_dataframe,
        )

    def test_trees_not_same_object(self, xgb_diabetes_model_trees_dataframe):
        """Test the trees attribute is copied from what is passed."""
        trees_with_predictions = XGBoostTabularTrees.derive_predictions(
            df=xgb_diabetes_model_trees_dataframe, lambda_=0
        )

        tabular_trees = XGBoostTabularTrees(trees_with_predictions)

        assert id(tabular_trees.data) != id(
            trees_with_predictions
        ), "trees attribute is the same object as passed into initialisation"


class TestFromBooster:
    """Tests for the XGBoostTabularTrees.from_booster method."""

    def test_alpha_not_zero_exception(self, xgb_diabetes_model_non_zero_alpha):
        """Test an exception is raised if alpha is not zero."""
        with pytest.raises(
            ValueError,
            match="Only Booster objects with alpha = 0 are supported.",
        ):
            XGBoostTabularTrees.from_booster(xgb_diabetes_model_non_zero_alpha)


class TestToTabularTrees:
    """Tests for the XGBoostTabularTrees.to_tabular_trees method."""

    def test_output_type(self, xgb_diabetes_model):
        """Test the output from to_tabular_trees is a TabularTrees object."""
        xgboost_tabular_trees = XGBoostTabularTrees.from_booster(xgb_diabetes_model)

        output = xgboost_tabular_trees.to_tabular_trees()

        assert (
            type(output) is TabularTrees
        ), "output from to_tabular_trees is not a TabularTrees object"
