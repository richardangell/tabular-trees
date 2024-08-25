import numpy as np

from tabular_trees.lightgbm.booster_string import (
    BoosterString,
)
from tabular_trees.lightgbm.editable_booster import (
    EditableBooster,
)


def test_booster_reproducible_from_booster_string(diabetes_data, lgb_diabetes_model):
    """Test that a Booster can be recovered from the BoosterString version of it."""
    # ARRANGE

    predictions = lgb_diabetes_model.predict(diabetes_data["data"])

    booster_string = BoosterString.from_booster(lgb_diabetes_model)

    # ACT

    reproduced_booster = booster_string.to_booster()

    # ASSERT

    predictions_reproduced = reproduced_booster.predict(diabetes_data["data"])

    np.testing.assert_array_equal(predictions, predictions_reproduced)


def test_booster_reproducible_from_editable_booster(diabetes_data, lgb_diabetes_model):
    """Test that a Booster can be recovered from the EditableBooster version of it."""
    # ARRANGE

    predictions = lgb_diabetes_model.predict(diabetes_data["data"])

    editable_booster = EditableBooster.from_booster(lgb_diabetes_model)

    # ACT

    reproduced_booster = editable_booster.to_booster()

    # ASSERT

    predictions_reproduced = reproduced_booster.predict(diabetes_data["data"])

    np.testing.assert_array_equal(predictions, predictions_reproduced)


def test_duplicating_trees_doubles_predictions(diabetes_data, lgb_diabetes_model):
    """Test that duplicating all trees doubles predictions from the model."""
    # ARRANGE

    predictions = lgb_diabetes_model.predict(diabetes_data["data"])

    editable_booster = EditableBooster.from_booster(lgb_diabetes_model)

    # ACT

    editable_booster.trees.extend(editable_booster.trees[:])
    editable_booster.header.tree_sizes.extend(editable_booster.header.tree_sizes)

    doubled_booster = editable_booster.to_booster()

    # ASSERT

    new_predictions = doubled_booster.predict(diabetes_data["data"])

    np.testing.assert_array_almost_equal(2 * predictions, new_predictions, decimal=11)


def test_shrinkage_does_not_change_predictions(diabetes_data, lgb_diabetes_model):
    """Demonstrate that changing the shrinkage does not change Booster predictions."""
    # ARRANGE

    predictions = lgb_diabetes_model.predict(diabetes_data["data"])

    editable_booster = EditableBooster.from_booster(lgb_diabetes_model)

    # ACT

    for tree in editable_booster.trees:
        tree.shrinkage = 0.11

    editable_booster.header.tree_sizes = [
        len(tree.get_booster_sting()) + 1 for tree in editable_booster.trees
    ]

    booster_with_shrinkage = editable_booster.to_booster()

    # ASSERT

    new_predictions = booster_with_shrinkage.predict(diabetes_data["data"])

    np.testing.assert_array_equal(predictions, new_predictions)
