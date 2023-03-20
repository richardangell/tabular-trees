import pandas as pd
import pytest
import xgboost as xgb

from tabular_trees.trees import export_tree_data
from tabular_trees.validate import (
    MonotonicConstraintResults,
    validate_monotonic_constraints,
)


def build_depth_2_single_tree_xgb(data: xgb.DMatrix) -> xgb.Booster:
    """Single tree, max depth 2.

    Learnign rate is 1, lambda is 0.

    """
    model = xgb.train(
        params={"verbosity": 0, "max_depth": 2, "eta": 1, "lambda": 0},
        dtrain=data,
        num_boost_round=1,
    )

    return model


@pytest.fixture(scope="session")
def two_way_monotonic_increase_x2_xgb_model(
    two_way_monotonic_increase_x2_dmatrix,
) -> xgb.Booster:
    """Model with single 2 way interaction both features monotonically increasing."""

    return build_depth_2_single_tree_xgb(two_way_monotonic_increase_x2_dmatrix)


@pytest.fixture(scope="session")
def two_way_monotonic_increase_decrease_xgb_model(
    two_way_monotonic_increase_decrease_dmatrix,
) -> xgb.Booster:
    """Model with single 2 way interaction.

    First feature is monotonically increasing, second feature is monotonically
    decreasing with the response.

    """
    return build_depth_2_single_tree_xgb(two_way_monotonic_increase_decrease_dmatrix)


@pytest.fixture(scope="session")
def two_way_monotonic_decrease_increase_xgb_model(
    two_way_monotonic_decrease_increase_dmatrix,
) -> xgb.Booster:
    """Model with single 2 way interaction.

    First feature is monotonically decreasing, second feature is monotonically
    increasing with the response.

    """
    return build_depth_2_single_tree_xgb(two_way_monotonic_decrease_increase_dmatrix)


@pytest.fixture(scope="session")
def two_way_monotonic_decrease_x2_xgb_model(
    two_way_monotonic_decrease_x2_dmatrix,
) -> xgb.Booster:
    """Model with single 2 way interaction both features monotonically decreasing."""

    return build_depth_2_single_tree_xgb(two_way_monotonic_decrease_x2_dmatrix)


class TestMonotonicConstraintResults:
    """Tests for the MonotonicConstraintResults class."""

    def test_attributes_set(self):
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
        self, columns_to_mark_true, expected_all_constraints_met
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


class TestValidateMonotonicConstraints:
    """Tests for the validate_monotonic_constraints function."""

    def test_output_type(self, xgb_diabetes_model_monotonic):
        """Test the output of validate_monotonic_constraints is MonotonicConstraintResults type."""
        model, constraints = xgb_diabetes_model_monotonic

        xgboost_tabular_trees = export_tree_data(model)
        tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()

        results = validate_monotonic_constraints(
            tabular_trees=tabular_trees, constraints=constraints
        )

        assert (
            results.all_constraints_met
        ), "unexpected violation of monotonic constraints"

        assert (
            type(results) is MonotonicConstraintResults
        ), "output from validate_monotonic_constraints is not MonotonicConstraintResults"

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
        self, request, model_fixture_name, constraints, expected_summary_results
    ):

        model = request.getfixturevalue(model_fixture_name)

        xgboost_tabular_trees = export_tree_data(model)
        tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()

        results = validate_monotonic_constraints(
            tabular_trees=tabular_trees, constraints=constraints
        )

        assert results.summary == expected_summary_results
