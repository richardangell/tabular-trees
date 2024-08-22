from tabular_trees.lightgbm.lightgbm_tabular_trees import LightGBMTabularTrees
from tabular_trees.trees import export_tree_data


def test_model_specific_function_dispatch(lgb_diabetes_model):
    """Test export_tree_data returns LightGBMTabularTrees object."""
    tree_data = export_tree_data(lgb_diabetes_model)

    assert type(tree_data) is LightGBMTabularTrees, (
        "incorrect type returned when export_tree_data "
        f"called with {type(lgb_diabetes_model)}"
    )
