import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest

from tabular_trees.lightgbm.booster_string import (
    BoosterString,
)
from tabular_trees.lightgbm.editable_booster import BoosterTree, EditableBooster


@pytest.fixture(scope="session")
def dummy_booster_to_modify() -> lgb.Booster:
    """Return dummy model using two numeric features.

    Modify this Booster in the tests allows

    """
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
            "max_depth": 1,
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

    return model


def test_convert_editable_booster_to_booster_string(lgb_diabetes_model):
    editable_booster = EditableBooster.from_booster(lgb_diabetes_model)

    booster_string_b = editable_booster.to_booster_string()

    assert type(booster_string_b) is BoosterString


def test_number_of_trees_expected(lgb_diabetes_model):
    """Test the number of trees in an EditableBooster is correct."""
    editable_booster = EditableBooster.from_booster(lgb_diabetes_model)

    assert len(editable_booster.trees) == 10


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

    editable_booster = EditableBooster.from_booster(model)

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


def test_single_split_tree_from_editbale_booster(dummy_booster_to_modify):
    r"""Test a single split tree can be produced from an EditableBooster.

    The tree being built is as follows:

    f0 <= 1
     /   \
    1     2

    """
    # ARRANGE

    editable_booster = EditableBooster.from_booster(dummy_booster_to_modify)

    tree_zero = BoosterTree(
        tree=0,
        num_leaves=2,
        num_cat=0,
        split_feature=[0],
        split_gain=[0],
        threshold=[1.0],
        decision_type=[2],
        left_child=[-1],
        right_child=[-2],
        leaf_value=[1, 2],
        leaf_weight=[1, 1],
        leaf_count=[1, 1],
        internal_value=[0],
        internal_weight=[1],
        internal_count=[1],
        is_linear=0,
        shrinkage=1,
    )

    editable_booster.trees = [tree_zero]

    editable_booster.header.tree_sizes = [
        len(tree.get_booster_sting()) + 1 for tree in editable_booster.trees
    ]

    new_booster = editable_booster.to_booster()

    predict_data = pd.DataFrame(
        {
            "a": [0, 1, 2, 3],
            "b": [0, 0, 1, 1],
        }
    )

    # ACT

    expected_predictions = np.array([1, 1, 2, 2])

    actual_predictions = new_booster.predict(predict_data)

    # ASSERT

    np.testing.assert_array_equal(expected_predictions, actual_predictions)


def test_eight_leaf_tree_from_editable_booster(dummy_booster_to_modify):
    r"""Test a depth 3, 8 leaf tree can be produced from an EditableBooster.

    The tree being built is as follows:

                 f0 <= 3
                  /   \
         f0 <= 1          f0 <= 5
          /   \            /   \
    f0 <= 0  f0 <= 2  f0 <= 4  f0 <= 6
     /   \    /   \    /   \    /   \
    0    1   2    3   4     5  6     7

    """
    # ARRANGE

    editable_booster = EditableBooster.from_booster(dummy_booster_to_modify)

    tree_zero = BoosterTree(
        tree=0,
        num_leaves=8,
        num_cat=0,
        split_feature=[0, 0, 0, 0, 0, 0, 0],
        split_gain=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        threshold=[3, 1, 5, 0, 2, 4, 6],
        decision_type=[2, 2, 2, 2, 2, 2, 2],
        left_child=[1, 3, 5, -1, -3, -5, -7],
        right_child=[2, 4, 6, -2, -4, -6, -8],
        leaf_value=[0, 1, 2, 3, 4, 5, 6, 7],
        leaf_weight=[1, 1, 1, 1, 1, 1, 1, 1],
        leaf_count=[1, 1, 1, 1, 1, 1, 1, 1],
        internal_value=[1, 2, 3, 4, 5, 6, 7],
        internal_weight=[1, 1, 1, 1, 1, 1, 1],
        internal_count=[1, 1, 1, 1, 1, 1, 1],
        is_linear=0,
        shrinkage=1,
    )

    editable_booster.trees = [tree_zero]

    editable_booster.header.tree_sizes = [
        len(tree.get_booster_sting()) + 1 for tree in editable_booster.trees
    ]

    new_booster = editable_booster.to_booster()

    predict_data = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5, 6, 7],
            "b": [0, 0, 0, 0, 0, 0, 0, 0],
        }
    )

    # ACT

    expected_predictions = np.array([0, 1, 2, 3, 4, 5, 6, 7])

    actual_predictions = new_booster.predict(predict_data)

    # ASSERT

    np.testing.assert_array_equal(expected_predictions, actual_predictions)


def test_five_split_tree_from_editable_booster(dummy_booster_to_modify):
    r"""Test a depth 4, 6 leaf tree can be produced from an EditableBooster.

    The tree being built is as follows:

             f0 <= 3
               /   \
     f0 <= 0          f0 <= 4
      /   \            /   \
    0     f0 <= 2     4     5
           /   \
      f0 <= 1   3
       /   \
      1     2

    """
    # ARRANGE

    editable_booster = EditableBooster.from_booster(dummy_booster_to_modify)

    tree_zero = BoosterTree(
        tree=0,
        num_leaves=6,
        num_cat=0,
        split_feature=[0, 0, 0, 0, 0],
        split_gain=[0.0, 0.0, 0.0, 0.0, 0.0],
        threshold=[3, 0, 4, 2, 1],
        decision_type=[2, 2, 2, 2, 2],
        left_child=[1, -1, -2, 4, -5],
        right_child=[2, 3, -3, -4, -6],
        leaf_value=[0, 4, 5, 3, 1, 2],
        leaf_weight=[1, 1, 1, 1, 1, 1],
        leaf_count=[1, 1, 1, 1, 1, 1],
        internal_value=[1, 2, 3, 4, 5],
        internal_weight=[1, 1, 1, 1, 1],
        internal_count=[1, 1, 1, 1, 1],
        is_linear=0,
        shrinkage=1,
    )

    editable_booster.trees = [tree_zero]

    editable_booster.header.tree_sizes = [
        len(tree.get_booster_sting()) + 1 for tree in editable_booster.trees
    ]

    new_booster = editable_booster.to_booster()

    predict_data = pd.DataFrame(
        {
            "a": [0, 1, 2, 3, 4, 5],
            "b": [0, 0, 0, 0, 0, 0],
        }
    )

    # ACT

    expected_predictions = np.array([0, 1, 2, 3, 4, 5])

    actual_predictions = new_booster.predict(predict_data)

    # ASSERT

    np.testing.assert_array_equal(expected_predictions, actual_predictions)


def test_five_leaf_tree_from_editable_booster(dummy_booster_to_modify):
    r"""Test a depth 3, 5 leaf tree can be produced from an EditableBooster.

    The tree being built is as follows:

                 f1 <= 0
                  /   \
         f0 <= 1          f0 <= 0
          /   \            /   \
    f0 <= 0    2          3     4
     /   \
    0     1

    This tree uses two variables to split on.

    """
    # ARRANGE

    editable_booster = EditableBooster.from_booster(dummy_booster_to_modify)

    tree_zero = BoosterTree(
        tree=0,
        num_leaves=5,
        num_cat=0,
        split_feature=[1, 0, 0, 0],
        split_gain=[0.0, 0.0, 0.0, 0.0],
        threshold=[0, 1, 0, 0],
        decision_type=[2, 2, 2, 2],
        left_child=[1, 3, -2, -4],
        right_child=[2, -1, -3, -5],
        leaf_value=[2, 3, 4, 0, 1],
        leaf_weight=[1, 1, 1, 1, 1],
        leaf_count=[1, 1, 1, 1, 1],
        internal_value=[1, 2, 3, 4],
        internal_weight=[1, 1, 1, 1],
        internal_count=[1, 1, 1, 1],
        is_linear=0,
        shrinkage=1,
    )

    editable_booster.trees = [tree_zero]

    editable_booster.header.tree_sizes = [
        len(tree.get_booster_sting()) + 1 for tree in editable_booster.trees
    ]

    new_booster = editable_booster.to_booster()

    predict_data = pd.DataFrame(
        {
            "a": [0, 1, 2, 0, 1],
            "b": [0, 0, 0, 1, 1],
        }
    )

    # ACT

    expected_predictions = np.array([0, 1, 2, 3, 4])

    actual_predictions = new_booster.predict(predict_data)

    # ASSERT

    np.testing.assert_array_equal(expected_predictions, actual_predictions)


def test_two_trees_from_editable_booster(dummy_booster_to_modify):
    r"""Test a 2 trees can be produced from an EditableBooster.

    The first tree being built is as follows:

             f1 <= 0
              /   \
    f0 <= 0          f0 <= 0
     /   \            /   \
    1     2          3     4

    The second tree being built is as follows:

    f0 <= 0
     /   \
    3     6

    """
    # ARRANGE

    editable_booster = EditableBooster.from_booster(dummy_booster_to_modify)

    tree_zero = BoosterTree(
        tree=0,
        num_leaves=4,
        num_cat=0,
        split_feature=[1, 0, 0],
        split_gain=[0.0, 0.0, 0.0],
        threshold=[0, 0, 0],
        decision_type=[2, 2, 2],
        left_child=[1, -1, -3],
        right_child=[2, -2, -4],
        leaf_value=[1, 2, 3, 4],
        leaf_weight=[1, 1, 1, 1],
        leaf_count=[1, 1, 1, 1],
        internal_value=[1, 2, 3],
        internal_weight=[1, 1, 1],
        internal_count=[1, 1, 1],
        is_linear=0,
        shrinkage=1,
    )

    tree_one = BoosterTree(
        tree=1,
        num_leaves=2,
        num_cat=0,
        split_feature=[0],
        split_gain=[0],
        threshold=[0],
        decision_type=[2],
        left_child=[-1],
        right_child=[-2],
        leaf_value=[3, 6],
        leaf_weight=[1, 1],
        leaf_count=[1, 1],
        internal_value=[0],
        internal_weight=[1],
        internal_count=[1],
        is_linear=0,
        shrinkage=1,
    )

    editable_booster.trees = [tree_zero, tree_one]

    editable_booster.header.tree_sizes = [
        len(tree.get_booster_sting()) + 1 for tree in editable_booster.trees
    ]

    new_booster = editable_booster.to_booster()

    predict_data = pd.DataFrame(
        {
            "a": [0, 1, 0, 1],
            "b": [0, 0, 1, 1],
        }
    )

    # ACT

    expected_predictions_tree = {0: np.array([1, 2, 3, 4]), 1: np.array([3, 6, 3, 6])}

    expected_predictions = expected_predictions_tree[0] + expected_predictions_tree[1]

    actual_predictions_tree = {
        i: new_booster.predict(predict_data, start_iteration=i, num_iteration=1)
        for i in range(2)
    }

    actual_predictions = new_booster.predict(predict_data)

    # ASSERT

    for tree_index in range(2):
        np.testing.assert_array_equal(
            expected_predictions_tree[tree_index], actual_predictions_tree[tree_index]
        )

    np.testing.assert_array_equal(expected_predictions, actual_predictions)
