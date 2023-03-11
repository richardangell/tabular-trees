import numpy as np

from tabular_trees.lightgbm import (
    BoosterString,
    convert_booster_string_to_editable_booster,
)


def test_booster_reproducible(diabetes_data, lgb_diabetes_model):

    predictions = lgb_diabetes_model.predict(diabetes_data["data"])

    booster_string = BoosterString(lgb_diabetes_model)

    reproduced_booster = booster_string.to_booster()

    predictions_reproduced = reproduced_booster.predict(diabetes_data["data"])

    np.testing.assert_array_equal(predictions, predictions_reproduced)


def test_convert_to_editable_booster(diabetes_data, lgb_diabetes_model):

    booster_string = BoosterString(lgb_diabetes_model)

    editable_booster = convert_booster_string_to_editable_booster(booster_string)

    assert len(editable_booster.trees) == 10
