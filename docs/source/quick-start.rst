Quick Start
====================

Welcome to the quick start guide for ``tabular_trees``. 

This package contains various bits of functionality for working with trees from common ML packages.

Installation
--------------------

The easiest way to get ``tabular_trees`` is to install directly with ``pip``;

   .. code::

     pip install tabular_trees

Editable LightGBM Booster
-------------------------

.. warning::
    The ``EditableBooster`` class is experimental and has not been tested with all
    options available within ``LightGBM``.

The ``EditableBooster`` class provides an object that can be converted to and from a ``lgb.Booster``. The ``EditableBooster`` can be modified to change the ``lgb.Booster``, for example defining specific trees.

First build a ``lgb.Booster`` model:

.. code:: python

    import numpy as np
    import lightgbm as lgb

    data = np.random.rand(500, 10)
    label = np.random.randint(2, size=500)
    train_data = lgb.Dataset(data, label=label)
    param = {'num_leaves': 31, 'objective': 'binary'}
    num_round = 10
    bst = lgb.train(param, train_data, num_round)

Then convert to ``EditableBooster``:

.. code:: python

    from tabular_trees import EditableBooster
    
    editable_booster = EditableBooster.from_booster(bst)

Add an extra tree to the ``EditableBooster`` ensemble:

.. code:: python

    from tabular_trees import BoosterTree

    extra_tree = BoosterTree(
        tree=0,
        num_leaves=2,
        num_cat=0,
        split_feature=[0],
        split_gain=[0],
        threshold=[0],
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

    editable_booster.trees.append(extra_tree)

    extra_tree_size = len(extra_tree.get_booster_sting()) + 1
    editable_booster.header.tree_sizes.append(extra_tree_size)

This example adds a simple tree structure with only a single split on the first feature.

Convert back to ``lgb.Booster`` object:

.. code:: python

    new_booster = editable_booster.to_booster()

Now that we have a ``lgb.Booster`` object we can make predictions with the modified model.

Tabular Tree Data
-------------------------

Tree based models (specifically GBMs) from ``xgboost``, ``lightgbm`` or ``scikit-learn`` can be exported to tabular data objects for further analysis.

The following models are supported:

- `LightGBM Booster <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html>`_
- `Scikit-Learn HistGradientBoostingClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html>`_
- `Scikit-Learn HistGradientBoostingRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html>`_
- `Scikit-Learn GradientBoostingClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_
- `Scikit-Learn GradientBoostingRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html>`_
- `XGBoost Booster <https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster>`_

Prediction Explanation
-------------------------

The ``decompose_prediction`` and ``calculate_shapley_values`` functions can be used to explain each feature's contribution to a single prediction.
