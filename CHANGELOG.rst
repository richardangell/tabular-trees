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
- Package import name to ``tabular_trees``

0.1.4 (2021-02-06)
------------------

- Package before changelog added
