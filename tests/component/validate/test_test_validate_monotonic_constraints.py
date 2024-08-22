import pandas as pd
import pytest
import xgboost as xgb

from tabular_trees.trees import export_tree_data
from tabular_trees.validate import (
    MonotonicConstraintResults,
    validate_monotonic_constraints,
)


@pytest.fixture(scope="session")
def xgb_diabetes_model_monotonic(xgb_diabetes_dmatrix) -> tuple[xgb.Booster, dict]:
    """Xgboost model with 10 trees and depth 3 on the diabetes dataset.

    Other parameters;
    - increasing monotonic constraint on bp and age.
    - decreasing monotonic constraint on bmi and s5.

    """
    feature_names = xgb_diabetes_dmatrix.feature_names

    monotonic_constraints = pd.Series([0] * len(feature_names), index=feature_names)
    monotonic_constraints.loc[monotonic_constraints.index.isin(["bmi", "s5"])] = -1
    monotonic_constraints.loc[monotonic_constraints.index.isin(["bp", "age"])] = 1

    monotonic_constraints_dict = monotonic_constraints.loc[
        monotonic_constraints != 0
    ].to_dict()

    model = xgb.train(
        params={
            "verbosity": 0,
            "max_depth": 3,
            "monotone_constraints": tuple(monotonic_constraints),
        },
        dtrain=xgb_diabetes_dmatrix,
        num_boost_round=10,
    )

    return model, monotonic_constraints_dict


def test_output_type(xgb_diabetes_model_monotonic):
    """Test validate_monotonic_constraints output is MonotonicConstraintResults."""
    model, constraints = xgb_diabetes_model_monotonic

    xgboost_tabular_trees = export_tree_data(model)
    tabular_trees = xgboost_tabular_trees.convert_to_tabular_trees()

    results = validate_monotonic_constraints(
        tabular_trees=tabular_trees, constraints=constraints
    )

    assert results.all_constraints_met, "unexpected violation of monotonic constraints"

    assert (
        type(results) is MonotonicConstraintResults
    ), "validate_monotonic_constraints output is MonotonicConstraintResults type"
