# tabular-trees

## Introduction

Tabular-trees is a package for making analysis on tree-based models easier. 

Tree based models (specifically GBMs) from `xgboost`, `lightgbm` or `scikit-learn` to be exported to `TabularTrees` objects for further analysis.

The `explain` and `validate` modules contain functions that operate on `TabularTrees`.

## Install

The easiest way to get `tabular-trees` is directly from [pypi](https://pypi.org/project/tabular-trees/);

```
pip install tabular_trees
```

`tabular-trees` works with GBMs from `xgboost`, `lightgbm` or `scikit-learn`. These packages must be installed to use the relevant functionality from `tabular-trees`, they are not installed as dependencies of `tabular-trees`.

## Documentation

Currently documentation is not published for this package.

## Build

`tabular-trees` uses [poetry](https://python-poetry.org/) as the environment management and package build tool. Follow the instructions [here](https://python-poetry.org/docs/#installation) to install.

Once installed run 

```
poetry install --with dev
```

to install the development dependencies. Other dependency groups are; `docs`, `lightgbm`, `scikit-learn` and `xgboost`.
