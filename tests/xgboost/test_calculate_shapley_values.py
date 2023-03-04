import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from tabular_trees.explain import calculate_shapley_values
from tabular_trees.trees import export_tree_data


@pytest.fixture()
def handcrafted_shap_data() -> tuple[pd.DataFrame, pd.Series]:
    """Handcrafted data where shapley values are easy to hand calculate.

    Data has 10 rows and looks as follows;

        x	y	z	response
    0	206	108	114	10
    1	206	380	390	10
    2	206	179	340	10
    3	206	153	380	10
    4	206	166	243	10
    5	194	326	328	20
    6	6	299	158	50
    7	6	299	237	50
    8	6	301	193	30
    9	6	301	186	30

    """
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
    predictions are exactly the average of the values falling into the leaves. The
    model has depth 2 and with the supplied data is able to fit exaclty to the
    response.

    """
    x_train, y_train = handcrafted_shap_data

    xgb_data = xgb.DMatrix(data=x_train, label=y_train)
    xgb_data.set_base_margin([0] * x_train.shape[0])

    model = xgb.train(
        params={
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


@pytest.mark.parametrize("row_number_to_score", [(0), (1), (11), (22), (199), (201)])
def test_shapley_values_treeshap_equality(
    row_number_to_score,
    diabetes_data_subset_cols,
    xgb_diabetes_dmatrix_subset_cols,
    xgb_diabetes_model_subset_cols,
):
    """Test equality between treeshap from xgboost and calculate_shapley_values."""

    # xgboost treeshap implementation
    shapley_values_xgboost = xgb_diabetes_model_subset_cols.predict(
        xgb_diabetes_dmatrix_subset_cols, pred_contribs=True
    )

    shapley_values_xgboost_df = pd.DataFrame(
        shapley_values_xgboost,
        columns=xgb_diabetes_dmatrix_subset_cols.feature_names + ["bias"],
    )

    # get row of diabetes data to score
    diabetes_data_df = pd.DataFrame(
        diabetes_data_subset_cols["data"],
        columns=diabetes_data_subset_cols["feature_names"],
    )
    row_data = diabetes_data_df.iloc[row_number_to_score]

    xgboost_tabular_trees = export_tree_data(xgb_diabetes_model_subset_cols)
    tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()

    shapley_values_tabular_trees = calculate_shapley_values(
        tabular_trees.trees, row_data
    )

    shapley_values_tabular_trees = shapley_values_tabular_trees.summary[
        shapley_values_xgboost_df.columns.values
    ]

    pd.testing.assert_frame_equal(
        shapley_values_xgboost_df.iloc[[row_number_to_score]].reset_index(drop=True),
        shapley_values_tabular_trees.reset_index(drop=True),
        check_dtype=False,
    )


def test_expected_shapley_values(handcrafted_shap_model):
    """Test shapley values output are as expected a known model."""
    xgboost_tabular_trees = export_tree_data(handcrafted_shap_model)
    tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()

    row_to_explain = pd.Series({"x": 150, "y": 75, "z": 200})

    shapley_values_tabular_trees = calculate_shapley_values(
        tabular_trees.trees, row_to_explain
    )

    expected_values = {"bias": 23.0, "x": -5.0, "y": 2.0, "z": 0.0}
    expected_shapley_values = pd.DataFrame(expected_values, index=[0])

    pd.testing.assert_frame_equal(
        shapley_values_tabular_trees.summary, expected_shapley_values
    )
