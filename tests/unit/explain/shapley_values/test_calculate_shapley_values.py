import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from tabular_trees.explain.shapley_values import ShapleyValues, calculate_shapley_values
from tabular_trees.trees import export_tree_data


@pytest.fixture()
def handcrafted_shap_data() -> tuple[pd.DataFrame, pd.Series]:
    """Handcrafted data where shapley values are easy to hand calculate."""
    # fmt: off
    # data below has 10 rows as follows:
    #   x	y	z	response
    #	206	108	114	10
    #	206	380	390	10
    #	206	179	340	10
    #	206	153	380	10
    #	206	166	243	10
    #	194	326	328	20
    #	6	299	158	50
    #	6	299	237	50
    #	6	301	193	30
    #	6	301	186	30
    # fmt: on

    np.random.seed(100)
    x_train = pd.DataFrame(
        {
            "x": [206] * 5 + [194] + [6] * 4,
            "y": list(np.random.randint(100, 400, 6)) + [299, 299, 301, 301],
            "z": list(np.random.randint(100, 400, 10)),
        }
    )

    y_train = pd.Series([10] * 5 + [20] + [50] * 2 + [30] * 2)
    y_train.name = "t"

    return x_train, y_train


@pytest.fixture()
def handcrafted_shap_model(handcrafted_shap_data) -> xgb.Booster:
    """Single decision tree on the handcrafted shap data.

    The model includes no regularisation and learning rate of 1 so that the leaf node
    predictions are exactly the average of the values falling into the leaves. The model
    has depth 2 and with the supplied data is able to fit exaclty to the response.

    """
    x_train, y_train = handcrafted_shap_data

    xgb_data = xgb.DMatrix(data=x_train, label=y_train)
    xgb_data.set_base_margin([0] * x_train.shape[0])

    model = xgb.train(
        params={
            "tree_method": "exact",
            "objective": "reg:squarederror",
            "max_depth": 2,
            "subsample": 1,
            "colsample_bytree": 1,
            "eta": 1,
            "lambda": 0,
            "gamma": 0,
            "alpha": 0,
        },
        dtrain=xgb_data,
        num_boost_round=1,
    )

    return model


def test_tabular_trees_not_tabular_trees_exception():
    """Test an exception is raised if tabular_trees is not a TabularTrees object."""
    row_to_explain = pd.Series({"x": 150, "y": 75, "z": 200})

    with pytest.raises(
        TypeError,
        match=(
            "tabular_trees is not in expected types "
            "<class 'tabular_trees.trees.TabularTrees'>"
        ),
    ):
        calculate_shapley_values(12345, row_to_explain)


def test_row_not_series_exception(handcrafted_shap_model):
    """Test an exception is raised if row is not a pd.Series object."""
    xgboost_tabular_trees = export_tree_data(handcrafted_shap_model)
    tabular_trees = xgboost_tabular_trees.to_tabular_trees()

    with pytest.raises(
        TypeError,
        match="row is not in expected types <class 'pandas.core.series.Series'>",
    ):
        calculate_shapley_values(tabular_trees, 12345)


def test_output_type(handcrafted_shap_model):
    """Test the output from calculate_shapley_values is a ShapleyValues object."""
    xgboost_tabular_trees = export_tree_data(handcrafted_shap_model)
    tabular_trees = xgboost_tabular_trees.to_tabular_trees()

    row_to_explain = pd.Series({"x": 150, "y": 75, "z": 200})

    results = calculate_shapley_values(tabular_trees, row_to_explain)

    assert (
        type(results) is ShapleyValues
    ), "output from calculate_shapley_values is not correct type"


def test_expected_shapley_values(handcrafted_shap_model):
    """Test shapley values output are as expected a known model."""
    xgboost_tabular_trees = export_tree_data(handcrafted_shap_model)
    tabular_trees = xgboost_tabular_trees.to_tabular_trees()

    row_to_explain = pd.Series({"x": 150, "y": 75, "z": 200})

    shapley_values_tabular_trees = calculate_shapley_values(
        tabular_trees, row_to_explain
    )

    expected_values = {"bias": 23.0, "x": -5.0, "y": 2.0, "z": 0.0}
    expected_shapley_values = pd.DataFrame(expected_values, index=[0])

    pd.testing.assert_frame_equal(
        shapley_values_tabular_trees.summary, expected_shapley_values
    )
