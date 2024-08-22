import lightgbm as lgb
import numpy as np
import pandas as pd

from tabular_trees.lightgbm.booster_string import (
    BoosterString,
)
from tabular_trees.lightgbm.booster_string_converter import (
    convert_booster_string_to_editable_booster,
)


def test_booster_reproducible_from_booster_string(diabetes_data, lgb_diabetes_model):
    """Test that a Booster can be recovered from the BoosterString version of it."""
    # ARRANGE

    predictions = lgb_diabetes_model.predict(diabetes_data["data"])

    booster_string = BoosterString(lgb_diabetes_model)

    # ACT

    reproduced_booster = booster_string.to_booster()

    # ASSERT

    predictions_reproduced = reproduced_booster.predict(diabetes_data["data"])

    np.testing.assert_array_equal(predictions, predictions_reproduced)


def test_booster_reproducible_from_editable_booster(diabetes_data, lgb_diabetes_model):
    """Test that a Booster can be recovered from the EditableBooster version of it."""
    # ARRANGE

    predictions = lgb_diabetes_model.predict(diabetes_data["data"])

    booster_string = BoosterString(lgb_diabetes_model)

    editable_booster = convert_booster_string_to_editable_booster(booster_string)

    # ACT

    reproduced_booster = editable_booster.to_booster()

    # ASSERT

    predictions_reproduced = reproduced_booster.predict(diabetes_data["data"])

    np.testing.assert_array_equal(predictions, predictions_reproduced)


def test_leaf_redictions_can_be_changed():
    """Test that leaf predictions of an EditableBooster can be chnaged.

    Specifically test that changes are able to propagate through the model and are
    output by the Booster.predict method when the EditableBooster is converted back to a
    Booster.

    """
    # ARRANGE

    data = pd.DataFrame(
        {
            "a": [1, 1, 0, 0],
            "b": [1, 0, 1, 0],
            "target": [25, 19, 9, 14],
        }
    )

    lgb_data = lgb.Dataset(
        data=data[["a", "b"]],
        label=data["target"],
        feature_name=["a", "b"],
    )

    model = lgb.train(
        params={
            "verbosity": -1,
            "max_depth": 3,
            "objective": "regression",
            "learning_rate": 1.0,
            "min_data_in_leaf": 1,
            "bagging_fraction": 1.0,
            "feature_fraction": 1.0,
            "min_gain_to_split": 0.0,
            "min_sum_hessian_in_leaf": 0.0,
            "seed": 0,
        },
        train_set=lgb_data,
        num_boost_round=1,
    )

    original_predictions = model.predict(data[["a", "b"]])

    booster_string = BoosterString(model)

    editable_booster = convert_booster_string_to_editable_booster(booster_string)

    # ACT

    editable_booster.trees[0].leaf_value = [1, 2, 3, 4]

    updated_booster = editable_booster.to_booster()

    new_predictions = updated_booster.predict(data[["a", "b"]])

    # ASSERT

    np.testing.assert_array_almost_equal(
        original_predictions,
        np.array([25, 19, 9, 14]),
        decimal=14,
    )

    np.testing.assert_array_almost_equal(
        new_predictions,
        np.array([3, 2, 4, 1]),
        decimal=14,
    )
