import pandas as pd
import pytest

from tabular_trees.validate import MonotonicConstraintResults


def test_attributes_set():
    """Test attributes are set during init."""
    summary_value = {"a": True}
    constraints_value = {"a": 1}
    results_df = pd.DataFrame({"a": [1, 2]})

    constraint_results = MonotonicConstraintResults(
        summary=summary_value, constraints=constraints_value, results=results_df
    )

    assert (
        constraint_results.summary == summary_value
    ), "summary attribute not set on MonotonicConstraintResults object"

    assert (
        constraint_results.constraints == constraints_value
    ), "constraints attribute not set on MonotonicConstraintResults object"

    pd.testing.assert_frame_equal(constraint_results.results, results_df)

    assert hasattr(
        constraint_results, "all_constraints_met"
    ), "constraint_results attribute not set on MonotonicConstraintResults object"

    assert (
        type(constraint_results.all_constraints_met) is bool
    ), "all_constraints_met attribute is not bool type"


@pytest.mark.parametrize(
    "columns_to_mark_true,expected_all_constraints_met",
    [(["a", "b"], True), (["a"], False), ([], False)],
)
def test_all_constraints_met_calculated_correctly(
    columns_to_mark_true, expected_all_constraints_met
):
    """Test all_constraints_met attribute is calculated correctly."""
    summary_value = {"a": False, "b": False}

    for column in columns_to_mark_true:
        summary_value[column] = True

    constraints_value = {"a": 1}
    results_df = pd.DataFrame({"a": [1, 2]})

    constraint_results = MonotonicConstraintResults(
        summary=summary_value, constraints=constraints_value, results=results_df
    )

    assert (
        constraint_results.all_constraints_met == expected_all_constraints_met
    ), "all_constraints_met not calcualted as expected"
