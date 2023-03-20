api documentation
====================

.. currentmodule:: tabular_trees

Trees API
------------------

.. autosummary::
    :toctree: api/

    export_tree_data
    TabularTrees
    BaseModelTabularTrees
    
LightGBM Trees API
--------------------------

.. autosummary::
    :toctree: api/

    lightgbm.LightGBMTabularTrees

Scikit-Learn Trees API
--------------------------

.. autosummary::
    :toctree: api/

    sklearn.ScikitLearnTabularTrees
    sklearn.ScikitLearnHistTabularTrees

XGBoost Trees API
--------------------------

.. autosummary::
    :toctree: api/

    xgboost.XGBoostTabularTrees
    xgboost.ParsedXGBoostTabularTrees
    xgboost.XGBoostParser
    xgboost.JsonDumpReader
    xgboost.TextDumpReader

.. warning::
    The ``XGBoostDumpParser`` is depreceated, ``Booster.trees_to_dataframe`` can be
    used instead to extract tree data from a ``Booster`` object.

Explain API
------------------

.. autosummary::
    :toctree: api/

    explain.decompose_prediction
    explain.PredictionDecomposition
    explain.calculate_shapley_values
    explain.ShapleyValues

.. warning::
    The ``calculate_shapley_values`` function is very slow and is only implemeneted for
    illustration purposes.

    Both ``xgboost`` and ``lightgbm`` implement the must faster treeSHAP algorithm,
    accessible via the ``Booster.predict`` methods when specifying ``pred_contribs`` or
    ``pred_contrib`` respectively.

Validate API
--------------------------

.. autosummary::
    :toctree: api/

    validate.validate_monotonic_constraints
    validate.MonotonicConstraintResults
