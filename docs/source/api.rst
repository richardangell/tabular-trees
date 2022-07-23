api documentation
====================

.. currentmodule:: tabular_trees

trees module
------------------

.. autosummary::
    :toctree: api/

    trees.BaseModelTabularTrees
    trees.export_tree_data
    trees.TabularTrees
          
sklearn.trees module
--------------------------

.. autosummary::
    :toctree: api/

    sklearn.trees.ScikitLearnTabularTrees
    sklearn.trees.ScikitLearnHistTabularTrees
    sklearn.trees.export_tree_data__gradient_boosting_model
    sklearn.trees.export_tree_data__hist_gradient_boosting_model

lightgbm.trees module
--------------------------

.. autosummary::
    :toctree: api/

    lightgbm.trees.LightGBMTabularTrees

xgboost.explainer module
--------------------------

.. autosummary::
    :toctree: api/

    xgboost.explainer.decompose_prediction
    xgboost.explainer.terminal_node_path
    xgboost.explainer.shapley_values
         
xgboost.parser module
--------------------------

.. autosummary::
    :toctree: api/

    xgboost.parser.DumpReader
    xgboost.parser.JsonDumpReader
    xgboost.parser.TextDumpReader
         
xgboost.trees module
--------------------------

.. autosummary::
    :toctree: api/

    xgboost.trees.XGBoostTabularTrees
    xgboost.trees.ParsedXGBoostTabularTrees

xgboost.validate module
--------------------------

.. autosummary::
    :toctree: api/

    xgboost.validate.validate_monotonic_constraints_df
