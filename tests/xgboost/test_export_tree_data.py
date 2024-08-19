import xgboost as xgb

from tabular_trees.trees import export_tree_data
from tabular_trees.xgboost import XGBoostTabularTrees


def test_model_specific_function_dispatch(xgb_diabetes_model):
    """Test export_tree_data returns XGBoostTabularTrees object."""
    tree_data = export_tree_data(xgb_diabetes_model)

    assert type(tree_data) is XGBoostTabularTrees, (
        "incorrect type returned when export_tree_data called with "
        f"{type(xgb_diabetes_model)}"
    )


def test_parameters_passed_to_xgboost_tabular_trees(xgb_diabetes_dmatrix):
    """Test that alpha and lambda are passed to XGBoostTabularTrees output."""
    model = xgb.train(
        params={"verbosity": 0, "max_depth": 3, "alpha": 0.0, "lambda": 1.0},
        dtrain=xgb_diabetes_dmatrix,
        num_boost_round=10,
    )

    tree_data = export_tree_data(model)

    assert (
        tree_data.alpha == 0.0
    ), "alpha value not expected on XGBoostTabularTrees object"

    assert (
        tree_data.lambda_ == 1.0
    ), "lambda value not expected on XGBoostTabularTrees object"
