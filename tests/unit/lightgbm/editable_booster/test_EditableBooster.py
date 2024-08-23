from tabular_trees.lightgbm.booster_string import (
    BoosterString,
)
from tabular_trees.lightgbm.booster_string_converter import (
    convert_booster_string_to_editable_booster,
)


def test_convert_editable_booster_to_booster_string(lgb_diabetes_model):
    booster_string = BoosterString(lgb_diabetes_model)

    editable_booster = convert_booster_string_to_editable_booster(booster_string)

    booster_string_b = editable_booster._to_booster_string()

    assert type(booster_string_b) is BoosterString


def test_number_of_trees_expected(lgb_diabetes_model):
    """Test the number of trees in an EditableBooster is correct."""
    booster_string = BoosterString(lgb_diabetes_model)

    editable_booster = convert_booster_string_to_editable_booster(booster_string)

    assert len(editable_booster.trees) == 10
