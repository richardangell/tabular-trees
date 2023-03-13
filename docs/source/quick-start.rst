Quick Start
====================

Welcome to the quick start guide for ``tabular_trees``. 

Tree based models (specifically GBMs) from ``xgboost``, ``lightgbm`` or ``scikit-learn`` can be exported to ``TabularTrees`` objects for further analysis.

So far the following models are supported:

- `LightGBM Booster <https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.Booster.html>`_
- `Scikit-Learn HistGradientBoostingClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html>`_
- `Scikit-Learn HistGradientBoostingRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html>`_
- `Scikit-Learn GradientBoostingClassifier <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html>`_
- `Scikit-Learn GradientBoostingRegressor <https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html>`_
- `XGBoost Booster <https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster>`_

The ``explain`` and ``validate`` modules contain functions that operate on ``TabularTrees`` objects.

Installation
--------------------

The easiest way to get ``tabular_trees`` is to install directly with ``pip``;

   .. code::

     pip install tabular_trees

Intro
--------------------

Let's say we are working with ``LightGBM`` and have built a Booster. The code below creates a dummy model:

.. code:: python

    import numpy as np
    import lightgbm as lgb

    data = np.random.rand(500, 10)
    label = np.random.randint(2, size=500)
    train_data = lgb.Dataset(data, label=label)
    param = {'num_leaves': 31, 'objective': 'binary'}
    num_round = 10
    bst = lgb.train(param, train_data, num_round)

Then we can export the tree data to a ``TabularTrees`` object with the following:

.. code:: python

    from tabular_trees import export_tree_data
    
    lightgbm_tabular_trees = export_tree_data(bst)
    tabular_trees = lightgbm_tabular_trees.convert_to_tabular_trees()

The tree data can then be inspected and further analysed:

.. code:: python

    tabular_trees.trees.head()
