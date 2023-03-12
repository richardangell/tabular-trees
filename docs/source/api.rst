api documentation
====================

.. currentmodule:: tabular_trees

Explain API
------------------

.. autosummary::
    :toctree: api/

    explain.decompose_prediction
    explain.PredictionDecomposition
    explain.calculate_shapley_values
    explain.ShapleyValues
    
LightGBM API
--------------------------

.. autosummary::
    :toctree: api/

    lightgbm.LightGBMTabularTrees

Scikit-Learn API
--------------------------

.. autosummary::
    :toctree: api/

    sklearn.ScikitLearnTabularTrees
    sklearn.ScikitLearnHistTabularTrees

Trees API
------------------

.. autosummary::
    :toctree: api/

    trees.BaseModelTabularTrees
    trees.TabularTrees
    trees.export_tree_data
         
Validate API
--------------------------

.. autosummary::
    :toctree: api/

    validate.validate_monotonic_constraints
    validate.MonotonicConstraintResults

XGBoost API
--------------------------

.. autosummary::
    :toctree: api/

    xgboost.XGBoostTabularTrees
    xgboost.ParsedXGBoostTabularTrees
    xgboost.XGBoostParser
    xgboost.JsonDumpReader
    xgboost.TextDumpReader
