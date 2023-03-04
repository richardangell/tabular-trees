api documentation
====================

.. currentmodule:: tabular_trees

explain module
------------------

.. autosummary::
    :toctree: api/

    explain.decompose_prediction
    explain.PredictionDecomposition
    explain.calculate_shapley_values
    explain.ShapleyValues
    
lightgbm module
--------------------------

.. autosummary::
    :toctree: api/

    lightgbm.LightGBMTabularTrees

sklearn module
--------------------------

.. autosummary::
    :toctree: api/

    sklearn.ScikitLearnTabularTrees
    sklearn.ScikitLearnHistTabularTrees

trees module
------------------

.. autosummary::
    :toctree: api/

    trees.BaseModelTabularTrees
    trees.TabularTrees
    trees.export_tree_data
         
validate module
--------------------------

.. autosummary::
    :toctree: api/

    validate.validate_monotonic_constraints
    validate.MonotonicConstraintResults

xgboost module
--------------------------

.. autosummary::
    :toctree: api/

    xgboost.XGBoostTabularTrees
    xgboost.ParsedXGBoostTabularTrees
    xgboost.XGBoostParser
    xgboost.JsonDumpReader
    xgboost.TextDumpReader
