from tabular_trees.trees import export_tree_data
from tabular_trees.validate import validate_monotonic_constraints_df


def test_successful_run(xgb_diabetes_model_monotonic):
    """Test successful run of validate_monotonic_constraints_df function."""
    model, constraints = xgb_diabetes_model_monotonic

    xgboost_tabular_trees = export_tree_data(model)
    tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()

    validate_monotonic_constraints_df(
        trees_df=tabular_trees.trees, constraints=constraints
    )
