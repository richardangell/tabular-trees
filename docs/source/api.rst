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

Explain API
------------------

.. autosummary::
    :toctree: api/

    explain.decompose_prediction
    explain.PredictionDecomposition
    explain.calculate_shapley_values
    explain.ShapleyValues

Validate API
--------------------------

.. autosummary::
    :toctree: api/

    validate.validate_monotonic_constraints
    validate.MonotonicConstraintResults
