Changelog
=========

This changelog follows the great advice from https://keepachangelog.com/.

Each section will have a title of the format ``X.Y.Z (YYYY-MM-DD)`` giving the version of the package and the date of release of that version. Unreleased changes i.e. those that have been merged into main (e.g. with a .dev suffix) but which are not yet in a new release (on PyPI) are added to the changelog but with the title ``X.Y.Z (unreleased)``. Unreleased sections can be combined when they are released and the date of release added to the title.

Subsections for each version can be one of the following;

- ``Added`` for new features.
- ``Changed`` for changes in existing functionality.
- ``Deprecated`` for soon-to-be removed features.
- ``Removed`` for now removed features.
- ``Fixed`` for any bug fixes.
- ``Security`` in case of vulnerabilities.

Each individual change should have a link to the pull request after the description of the change.

0.3.0 (unreleased)
------------------

Added
^^^^^
- ``EditableBooster`` class as a way to edit ``lgb.Booster`` models
- ``BoosterString`` to help converting back and forth between ``lgb.Booster`` and ``EditableBooster``
- Examples to docstrings for main functions / classes.
- Test documentation examples with ``--doctest-modules`` in ``tox``.

Changed
^^^^^^^
- Restructure modules.
- Bump minimum required dependency versions (``pandas``: ``2.2.0``, ``lightgbm``: ``4.5.0``, ``xgboost``: ``2.1.0``, ``scikit-learn``: ``1.5.0``).
- Swap to use ``ruff`` instead of various other tools.
- Strengthen ``mypy`` settings and improve type hints.
- Improve tests.
- Tests using to ``lightgbm`` or ``xgboost`` are no longer imported if those packages are not installed.
- Update ``decompose_prediction`` to use ``pd.concat`` instead of depreceated ``pd.DataFrame.append``.
- Change ``ScikitLearnHistTabularTrees`` to include the starting value in the predictions of the first tree.

Fixed
^^^^^
- Fix ``validate_monotonic_constraints`` to correctly loop through all trees in the model

0.2.0 (2023-03-13)
------------------

Added
^^^^^

- ``XGBoostTabularTrees`` to hold ``xgb.Booster.trees_to_dataframe`` output.
- ``ParsedXGBoostTabularTrees`` class to hold the outout of xgboost model when parsed from text or json file.
- ``LightGBMTabularTrees`` class to hold output from ``lgb.Booster.trees_to_dataframe``.
- ``ScikitLearnTabularTrees`` class to hold tree data from ``GradientBoostingClassifier`` or ``GradientBoostingRegressor`` objects.
- ``ScikitLearnHistTabularTrees`` class to hold tree data from ``HistGradientBoostingClassifier`` or ``HistGradientBoostingRegressor`` objects.
- ``export_tree_data`` as the user interface to export tree data for any supported model, dispatching to the correct model specific function
- ``MonotonicConstraintResults`` class to hold outputs from ``validate_monotonic_constraints``.
- ``PredictionDecomposition`` class to hold the outputs from ``decompose_prediction``.
- ``ShapleyValues`` class to hold the outputs from ``calculate_shapley_values``.
- github action workflows to run test ``coverage``, ``pre-commit``, ``tox`` and check required files have changed.
- Docs and ``readthedocs`` configuration.
- ``pre-commit`` with ``black``, ``bandit``, ``mypy``, ``flake8``, ``isort``, ``docformatter`` and ``pydocstyle``.
- Type hints.

Changed
^^^^^^^

- Change package import name to ``tabular_trees`` from ``ttrees``.
- Extensive modules restructure and move source code to ``src`` directory.
- Refactor ``xgb.parser`` and move functionality into ``xgboost`` module.
- Rename ``validate_monotonic_constraints_df`` function to ``validate_monotonic_constraints``.
- Refactor ``validate_monotonic_constraints`` function to accept ``TabularTrees`` objects.
- Refactor ``decompose_prediction`` function to accept ``TabularTrees`` objects.
- Rename ``shapley_values`` function to ``calculate_shapley_values``.
- Refactor ``calculate_shapley_values`` function to accept ``TabularTrees`` objects.
- Swap project to use ``pyproject.toml`` and ``poetry`` as the build tool.
- ``LightGBM``, ``Scikit-Learn`` and ``XGBoost`` are now optional extra dependencies.

0.1.4 (2021-02-06)
------------------

Added
^^^^^

- ``TabularTrees`` class to hold tree data in table format.
- ``parse_model`` function to load an ``xgboost.Booster`` and derive predictions for internal nodes.
- ``validate_monotonic_constraints_df`` function to validate monotonic constraints for ``xgboost.Booster``s.
- ``decompose_prediction`` to decompose predictions from an ``xgboost.Booster``.
- ``shapley_values`` to calculate shapley values for predictions from an ``xgboost.Booster``.
