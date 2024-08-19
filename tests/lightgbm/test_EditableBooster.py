import lightgbm as lgb
import numpy as np
import pandas as pd

from tabular_trees.lightgbm import (
    BoosterString,
    convert_booster_string_to_editable_booster,
)


def test_booster_reproducible_from_booster_string(diabetes_data, lgb_diabetes_model):

    predictions = lgb_diabetes_model.predict(diabetes_data["data"])

    booster_string = BoosterString(lgb_diabetes_model)

    reproduced_booster = booster_string.to_booster()

    predictions_reproduced = reproduced_booster.predict(diabetes_data["data"])

    np.testing.assert_array_equal(predictions, predictions_reproduced)


def test_convert_to_editable_booster(lgb_diabetes_model):

    booster_string = BoosterString(lgb_diabetes_model)

    editable_booster = convert_booster_string_to_editable_booster(booster_string)

    assert len(editable_booster.trees) == 10


def test_convert_editable_booster_to_booster_string(lgb_diabetes_model):

    booster_string = BoosterString(lgb_diabetes_model)

    editable_booster = convert_booster_string_to_editable_booster(booster_string)

    booster_string_b = editable_booster._to_booster_string()

    assert type(booster_string_b) is BoosterString


def test_booster_reproducible_from_editable_booster(diabetes_data, lgb_diabetes_model):

    predictions = lgb_diabetes_model.predict(diabetes_data["data"])

    booster_string = BoosterString(lgb_diabetes_model)

    editable_booster = convert_booster_string_to_editable_booster(booster_string)

    reproduced_booster = editable_booster.to_booster()

    predictions_reproduced = reproduced_booster.predict(diabetes_data["data"])

    np.testing.assert_array_equal(predictions, predictions_reproduced)


def test_leaf_redictions_can_be_changed():

    x_train = pd.DataFrame(
        {
            "a": [1, 1, 0, 0],
            "b": [1, 0, 1, 0],
            "target": [25, 19, 9, 14],
        }
    )

    lgb_data = lgb.Dataset(
        data=x_train[["a", "b"]],
        label=x_train["target"],
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

    booster_string = BoosterString(model)

    editable_booster = convert_booster_string_to_editable_booster(booster_string)

    editable_booster.trees[0].leaf_value = [1, 2, 3, 4]

    updated_booster = editable_booster.to_booster()

    original_predictions = model.predict(x_train[["a", "b"]])

    new_predictions = updated_booster.predict(x_train[["a", "b"]])

    np.testing.assert_array_almost_equal(
        x=original_predictions,
        y=np.array([25, 19, 9, 14]),
        decimal=14,
    )

    np.testing.assert_array_almost_equal(
        x=new_predictions,
        y=np.array([3, 2, 4, 1]),
        decimal=14,
    )
