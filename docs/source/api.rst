api documentation
====================

.. currentmodule:: tabular_trees

Trees API
------------------

.. autosummary::
    :toctree: api/

    export_tree_data
    TabularTrees
    
LightGBM Trees API
--------------------------

.. autosummary::
    :toctree: api/

    lightgbm.lightgbm_tabular_trees.LightGBMTabularTrees

Scikit-Learn Trees API
--------------------------

.. autosummary::
    :toctree: api/

    sklearn.sklearn_tabular_trees.ScikitLearnTabularTrees
    sklearn.sklearn_hist_tabular_trees.ScikitLearnHistTabularTrees

XGBoost Trees API
--------------------------

.. autosummary::
    :toctree: api/

    xgboost.xgboost_tabular_trees.XGBoostTabularTrees
    xgboost.dump_parser.ParsedXGBoostTabularTrees
    xgboost.dump_parser.XGBoostParser
    xgboost.dump_reader.JsonDumpReader
    xgboost.dump_reader.TextDumpReader

.. warning::
    The ``XGBoostDumpParser`` is depreceated, ``Booster.trees_to_dataframe`` can be
    used instead to extract tree data from a ``Booster`` object.

Explain API
------------------

.. autosummary::
    :toctree: api/

    explain.prediction_decomposition.decompose_prediction
    explain.prediction_decomposition.PredictionDecomposition
    explain.shapley_values.calculate_shapley_values
    explain.shapley_values.ShapleyValues

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
