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

0.2.0.dev4 (unreleased) `#14 <https://github.com/richardangell/tabular-trees/pull/14>`_
---------------------------------------------------------------------------------------

Changed
^^^^^^^
- Update xgboost.trees.XGBoostTabularTrees and ParsedXGBoostTabularTrees to inherit from trees.BaseModelTabularTrees

0.2.0.dev3 (unreleased) `#13 <https://github.com/richardangell/tabular-trees/pull/13>`_
---------------------------------------------------------------------------------------

Added
^^^^^
- Add sklearn module
- Add sklearn.trees.ScikitLearnTabularTrees class to hold tree data from GradientBoostingClassifier or Regressor objects
- Add export_tree_data__gradient_boosting_model to export tree data from GradientBoostingClassifier or Regressor objects to ScikitLearnTabularTrees
- Add sklearn.trees.ScikitLearnHistTabularTrees class to hold tree data from HistGradientBoostingClassifier or Regressor objects
- Add export_tree_data__hist_gradient_boosting_model function to export tree data from from HistGradientBoostingClassifier or Regressor objects to ScikitLearnHistTabularTrees
- Add new trees.BaseModelTabularTrees abstract base class for model specific tree data class implementations to inherit from
- Add trees.export_tree_data as the user interface to export tree data for any model, dispatching to the correct model specific function

Changed
^^^^^^^
- Change lightgbm.trees.LightGBMTabularTrees to inherit from trees.BaseModelTabularTrees

0.2.0.dev2 (unreleased) `#12 <https://github.com/richardangell/tabular-trees/pull/12>`_
---------------------------------------------------------------------------------------

Added
^^^^^
- Add lightgbm.trees.LightGBMTabularTrees class to hold output from lgb.Booster.trees_to_dataframe

0.2.0.dev1 (unreleased) `#11 <https://github.com/richardangell/tabular-trees/pull/11>`_
---------------------------------------------------------------------------------------

Added
^^^^^

- Add XGBoostTabularTrees to hold xgb.Booster.trees_to_dataframe output
- Add ParsedXGBoostTabularTrees class to hold the outout of parser classes
- Add DumpReader, JsonDumpReader, TextDumpReader classes in xgboost.parser module.

Changed
^^^^^^^

- Rename xgb module to xgboost
- Exclude .tox directory in bandit
- Refactor xgboost.parser and move some functionality into xgboost.trees classes

0.2.0.dev0 (unreleased) `#10 <https://github.com/richardangell/tabular-trees/pull/10>`_
---------------------------------------------------------------------------------------

Added
^^^^^

- Separate github action workflows to run test coverage, pre-commit, tox and check required files have changed
- Docs
- Pipfile for development environment
- Pre-commit with black, bandit, mypy and flake8

Changed
^^^^^^^

- Project to use pyproject.toml and flit as the build tool
- Tox configuration
- Source code moved to src directory
- Package import name to ``tabular_trees`` from ``ttrees``

0.1.4 (2021-02-06)
------------------

- Package before changelog added
