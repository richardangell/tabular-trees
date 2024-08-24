from tabular_trees.trees import export_tree_data
from tabular_trees.xgboost.xgboost_tabular_trees import XGBoostTabularTrees


def test_model_specific_function_dispatch(xgb_diabetes_model):
    """Test export_tree_data returns XGBoostTabularTrees object."""
    tree_data = export_tree_data(xgb_diabetes_model)

    assert type(tree_data) is XGBoostTabularTrees, (
        "incorrect type returned when export_tree_data called with "
        f"{type(xgb_diabetes_model)}"
    )
