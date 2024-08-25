api documentation
====================

.. currentmodule:: tabular_trees

Trees
------------------

.. autosummary::
    :toctree: api/

    export_tree_data
    TabularTrees
    
LightGBM
--------------------------

.. autosummary::
    :toctree: api/

    LightGBMTabularTrees
    EditableBooster
    BoosterHeader
    BoosterTree
    BoosterString

Scikit-Learn
--------------------------

.. autosummary::
    :toctree: api/

    ScikitLearnTabularTrees
    ScikitLearnHistTabularTrees

XGBoost
--------------------------

.. autosummary::
    :toctree: api/

    XGBoostTabularTrees
    XGBoostParser
    ParsedXGBoostTabularTrees

.. warning::
    The ``XGBoostParser`` is depreceated, ``Booster.trees_to_dataframe`` can be
    used instead to extract tree data from a ``Booster`` object.

Explain
------------------

.. autosummary::
    :toctree: api/

    decompose_prediction
    PredictionDecomposition
    calculate_shapley_values
    ShapleyValues

.. warning::
    The ``calculate_shapley_values`` function is very slow and is only implemeneted for
    illustration purposes.

    Both ``xgboost`` and ``lightgbm`` implement the must faster treeSHAP algorithm,
    accessible via the ``Booster.predict`` methods when specifying ``pred_contribs`` or
    ``pred_contrib`` respectively.

Validate
--------------------------

.. autosummary::
    :toctree: api/

    validate_monotonic_constraints
    MonotonicConstraintResults
