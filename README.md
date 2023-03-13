# tabular-trees

![PyPI](https://img.shields.io/pypi/v/tabular-trees?color=success&style=flat)
![Read the Docs](https://img.shields.io/readthedocs/tabular-trees)
![GitHub](https://img.shields.io/github/license/richardangell/tabular-trees)
![GitHub last commit](https://img.shields.io/github/last-commit/richardangell/tabular-trees)
![Build](https://github.com/richardangell/tabular-trees/actions/workflows/coverage.yml/badge.svg?branch=main)

## Introduction

`tabular-trees` is a package for making analysis on tree-based models easier. 

Tree based models (specifically GBMs) from `xgboost`, `lightgbm` or `scikit-learn` can be exported to `TabularTrees` objects for further analysis.

The `explain` and `validate` modules contain functions that operate on `TabularTrees` objects.

See the [documentation](http://tabular-trees.readthedocs.io/) for more information.

## Install

The easiest way to get `tabular-trees` is to install directly from [pypi](https://pypi.org/project/tabular-trees/);

```
pip install tabular_trees
```

`tabular-trees` works with GBMs from `xgboost`, `lightgbm` or `scikit-learn`. These packages must be installed to use the relevant functionality from `tabular-trees`, they are not installed as dependencies of `tabular-trees`.

## Build

`tabular-trees` uses [poetry](https://python-poetry.org/) as the environment management and package build tool. Follow the instructions [here](https://python-poetry.org/docs/#installation) to install.

Once installed run 

```
poetry install --with dev
```

to install the development dependencies. Other dependency groups are; `docs`, `lightgbm`, `sklearn` and `xgboost`.
