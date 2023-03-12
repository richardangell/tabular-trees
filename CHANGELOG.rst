Changelog
=========

This changelog follows the great advice from https://keepachangelog.com/.

Each section will have a title of the format ``X.Y.Z (YYYY-MM-DD)`` giving the version of the package and the date of release of that version. Unreleased changes i.e. those that have been merged into master (e.g. with a .dev suffix) but which are not yet in a new release (on PyPI) are added to the changelog but with the title ``X.Y.Z (unreleased)``. Unreleased sections can be combined when they are released and the date of release added to the title.

Subsections for each version can be one of the following;

- ``Added`` for new features.
- ``Changed`` for changes in existing functionality.
- ``Deprecated`` for soon-to-be removed features.
- ``Removed`` for now removed features.
- ``Fixed`` for any bug fixes.
- ``Security`` in case of vulnerabilities.

Each individual change should have a link to the pull request after the description of the change.

0.2.0.dev6 (unreleased) `#16 <https://github.com/richardangell/tabular-trees/pull/16>`_
---------------------------------------------------------------------------------------

Changed
^^^^^^^
- Change ``version`` github action to look at ``pyproject.toml`` instead of ``_version.py``
- Change ``coverage`` github action to use ``poetry``
- Update dependencies in ``tox`` configuration

0.2.0.dev5 (unreleased) `#15 <https://github.com/richardangell/tabular-trees/pull/15>`_
---------------------------------------------------------------------------------------

Added
^^^^^
- Add ``isort``, ``docformatter`` and ``pydocstyle`` to ``pre-commit`` configuration
- Add ``MonotonicConstraintResults`` class to hold outputs from ``validate_monotonic_constraints``
- Add ``convert_to_tabular_trees`` methods to ``LightGBMTabularTrees`` and ``XGBoostTabularTrees`` classes
- Add ``export_tree_data`` support for ``LightGBM`` and ``XBGoost`` ``Booster`` objects

Changed
^^^^^^^
- Swap to ``poetry`` for project environment management and build tool
- Update notebooks in demo folder
- Update packages to latests version in ``docs/requirements.txt``
- No longer set package version number in ``_version.py``, use ``pyproject.toml`` as the single source for the version number
- Rename ``validate_monotonic_constraints_df`` function to ``validate_monotonic_constraints``
- Change ``validate_monotonic_constraints`` function to accept ``TabularTrees`` objects
- Change ``decompose_prediction function to accept ``TabularTrees`` objects
- Rename ``shapley_values`` function to ``calculate_shapley_values``
- Change ``shapley_values`` function to accept ``TabularTrees`` objects
- Rename ``xgboost.explainer`` module to ``explain``
- Move ``xgboost.explain`` submodule out of ``xgboost`` module
- Move ``xgboost.validate`` submodule out of ``xgboost`` module
- Move contents of ``xgboost.parser`` into ``xgboost.trees`` module
- Rename and move ``xgboost.trees`` module to ``xgboost``
- Rename and move ``sklearn.trees`` module to ``sklearn``
- Rename and move ``lightgbm.trees`` module to ``lightgbm``
- Change ``TabularTrees`` to no longer inherit from ``BaseModelTabularTrees``
- Change ``BaseModelTabularTrees`` and ``DumpReader`` ABCs implementation of abstract properties, @classmethod is now `removed <https://docs.python.org/3.11/whatsnew/3.11.html#language-builtins>`_ which means that classes not implementing the required properties can be defined (whereas they couldn't before), but not initialised

Removed
^^^^^^^
- Remove ``get_trees`` method from ``XGBoostTabularTrees`` class
- Remove ``derive_depths`` method from ``XGBoostTabularTrees`` class
- Remove ``check_tree_data`` method from ``TabularTrees`` class
- Remove ``calculate_number_nodes`` method from ``TabularTrees`` class
- Remove ``calculate_number_leaf_nodes`` method from ``TabularTrees`` class
- Remove ``calculate_max_depth_trees`` method from ``TabularTrees`` class
- Remove ``calculate_number_trees`` method from ``TabularTrees`` class
- Remove ``__post_post_init__`` method from ``BaseModelTabularTrees`` class

0.2.0.dev4 (unreleased) `#14 <https://github.com/richardangell/tabular-trees/pull/14>`_
---------------------------------------------------------------------------------------

Changed
^^^^^^^
- Update ``xgboost.trees.XGBoostTabularTrees`` and ``ParsedXGBoostTabularTrees`` to inherit from ``trees.BaseModelTabularTrees``

0.2.0.dev3 (unreleased) `#13 <https://github.com/richardangell/tabular-trees/pull/13>`_
---------------------------------------------------------------------------------------

Added
^^^^^
- Add ``sklearn`` module
- Add ``sklearn.trees.ScikitLearnTabularTrees`` class to hold tree data from ``GradientBoostingClassifier`` or ``GradientBoostingRegressor`` objects
- Add ``export_tree_data__gradient_boosting_model`` to export tree data from ``GradientBoostingClassifier`` or ``GradientBoostingRegressor`` objects to ``ScikitLearnTabularTrees``
- Add ``sklearn.trees.ScikitLearnHistTabularTrees`` class to hold tree data from ``HistGradientBoostingClassifier`` or ``HistGradientBoostingRegressor`` objects
- Add ``export_tree_data__hist_gradient_boosting_model`` function to export tree data from from ``HistGradientBoostingClassifier`` or ``HistGradientBoostingRegressor`` objects to ``ScikitLearnHistTabularTrees``
- Add new ``trees.BaseModelTabularTrees`` abstract base class for model specific tree data class implementations to inherit from
- Add ``trees.export_tree_data`` as the user interface to export tree data for any supported model, dispatching to the correct model specific function

Changed
^^^^^^^
- Change ``lightgbm.trees.LightGBMTabularTrees`` to inherit from ``trees.BaseModelTabularTrees``

0.2.0.dev2 (unreleased) `#12 <https://github.com/richardangell/tabular-trees/pull/12>`_
---------------------------------------------------------------------------------------

Added
^^^^^
- Add ``lightgbm.trees.LightGBMTabularTrees`` class to hold output from ``lgb.Booster.trees_to_dataframe``

0.2.0.dev1 (unreleased) `#11 <https://github.com/richardangell/tabular-trees/pull/11>`_
---------------------------------------------------------------------------------------

Added
^^^^^

- Add ``XGBoostTabularTrees`` to hold ``xgb.Booster.trees_to_dataframe`` output
- Add ``ParsedXGBoostTabularTrees`` class to hold the outout of parser classes
- Add ``DumpReader``, ``JsonDumpReader``, ``TextDumpReader`` classes in ``xgboost.parser`` module.

Changed
^^^^^^^

- Rename ``xgb`` module to ``xgboost``
- Exclude .tox directory in ``bandit``
- Refactor ``xgboost.parser`` and move some functionality into ``xgboost.trees`` classes

0.2.0.dev0 (unreleased) `#10 <https://github.com/richardangell/tabular-trees/pull/10>`_
---------------------------------------------------------------------------------------

Added
^^^^^

- Separate github action workflows to run test ``coverage``, ``pre-commit``, ``tox`` and check required files have changed
- Docs
- Pipfile for development environment
- ``pre-commit`` with ``black``, ``bandit``, ``mypy`` and ``flake8`

Changed
^^^^^^^

- Project to use ``pyproject.toml`` and ``flit`` as the build tool
- Tox configuration
- Source code moved to src directory
- Package import name to ``tabular_trees`` from ``ttrees``

0.1.4 (2021-02-06)
------------------

- Package before changelog added
