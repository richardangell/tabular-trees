from tabular_trees.lightgbm.booster_string import (
    BoosterString,
)
from tabular_trees.lightgbm.editable_booster import (
    EditableBooster,
)


def test_convert_editable_booster_to_booster_string(lgb_diabetes_model):
    editable_booster = EditableBooster.from_booster(lgb_diabetes_model)

    booster_string_b = editable_booster.to_booster_string()

    assert type(booster_string_b) is BoosterString


def test_number_of_trees_expected(lgb_diabetes_model):
    """Test the number of trees in an EditableBooster is correct."""
    editable_booster = EditableBooster.from_booster(lgb_diabetes_model)

    assert len(editable_booster.trees) == 10
