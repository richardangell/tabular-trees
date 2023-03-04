from tabular_trees.trees import export_tree_data
from tabular_trees.validate import validate_monotonic_constraints


def test_successful_run(xgb_diabetes_model_monotonic):
    """Test successful run of validate_monotonic_constraints function."""
    model, constraints = xgb_diabetes_model_monotonic

    xgboost_tabular_trees = export_tree_data(model)
    tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()

    results = validate_monotonic_constraints(
        tabular_trees=tabular_trees, constraints=constraints
    )

    assert results.all_constraints_met, "unexpected violation of monotonic constraints"
