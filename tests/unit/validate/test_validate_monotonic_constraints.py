import pytest

from tabular_trees.trees import export_tree_data
from tabular_trees.validate import (
    validate_monotonic_constraints,
)


@pytest.mark.parametrize(
    "model_fixture_name,constraints,expected_summary_results",
    [
        (
            "two_way_monotonic_increase_x2_xgb_model",
            {"a": 1, "b": 1},
            {"a": True, "b": True},
        ),
        (
            "two_way_monotonic_increase_x2_xgb_model",
            {"a": 1, "b": -1},
            {"a": True, "b": False},
        ),
        (
            "two_way_monotonic_increase_x2_xgb_model",
            {"a": -1, "b": 1},
            {"a": False, "b": True},
        ),
        (
            "two_way_monotonic_increase_x2_xgb_model",
            {"a": -1, "b": -1},
            {"a": False, "b": False},
        ),
        (
            "two_way_monotonic_increase_decrease_xgb_model",
            {"a": 1, "b": 1},
            {"a": True, "b": False},
        ),
        (
            "two_way_monotonic_increase_decrease_xgb_model",
            {"a": 1, "b": -1},
            {"a": True, "b": True},
        ),
        (
            "two_way_monotonic_increase_decrease_xgb_model",
            {"a": -1, "b": 1},
            {"a": False, "b": False},
        ),
        (
            "two_way_monotonic_increase_decrease_xgb_model",
            {"a": -1, "b": -1},
            {"a": False, "b": True},
        ),
        (
            "two_way_monotonic_decrease_increase_xgb_model",
            {"a": 1, "b": 1},
            {"a": False, "b": True},
        ),
        (
            "two_way_monotonic_decrease_increase_xgb_model",
            {"a": 1, "b": -1},
            {"a": False, "b": False},
        ),
        (
            "two_way_monotonic_decrease_increase_xgb_model",
            {"a": -1, "b": 1},
            {"a": True, "b": True},
        ),
        (
            "two_way_monotonic_decrease_increase_xgb_model",
            {"a": -1, "b": -1},
            {"a": True, "b": False},
        ),
        (
            "two_way_monotonic_decrease_x2_xgb_model",
            {"a": 1, "b": 1},
            {"a": False, "b": False},
        ),
        (
            "two_way_monotonic_decrease_x2_xgb_model",
            {"a": 1, "b": -1},
            {"a": False, "b": True},
        ),
        (
            "two_way_monotonic_decrease_x2_xgb_model",
            {"a": -1, "b": 1},
            {"a": True, "b": False},
        ),
        (
            "two_way_monotonic_decrease_x2_xgb_model",
            {"a": -1, "b": -1},
            {"a": True, "b": True},
        ),
    ],
)
def test_expected_output(
    request, model_fixture_name, constraints, expected_summary_results
):
    model = request.getfixturevalue(model_fixture_name)

    xgboost_tabular_trees = export_tree_data(model)
    tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()

    results = validate_monotonic_constraints(
        tabular_trees=tabular_trees, constraints=constraints
    )

    assert results.summary == expected_summary_results
