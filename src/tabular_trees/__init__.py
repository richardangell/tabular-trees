"""Tabular-trees package."""

import contextlib
from importlib import metadata

from .explain.prediction_decomposition import (
    PredictionDecomposition,
    decompose_prediction,
)
from .explain.shapley_values import ShapleyValues, calculate_shapley_values
from .trees import TabularTrees, export_tree_data
from .validate import MonotonicConstraintResults, validate_monotonic_constraints

with contextlib.suppress(ImportError):
    from .lightgbm.booster_string import BoosterString
    from .lightgbm.editable_booster import BoosterHeader, BoosterTree, EditableBooster
    from .lightgbm.lightgbm_tabular_trees import LightGBMTabularTrees

with contextlib.suppress(ImportError):
    from .sklearn.sklearn_hist_tabular_trees import ScikitLearnHistTabularTrees
    from .sklearn.sklearn_tabular_trees import ScikitLearnTabularTrees

with contextlib.suppress(ImportError):
    from .xgboost.dump_parser import ParsedXGBoostTabularTrees, XGBoostParser
    from .xgboost.xgboost_tabular_trees import XGBoostTabularTrees

# single source for version number is in the pyproject.toml
# note, for an editable install, the package version number will not be
# updated until the package is reinstalled with `poetry install`
__version__ = metadata.version(__package__)

# avoids polluting the results of dir(__package__)
del metadata


__all__ = [
    "decompose_prediction",
    "PredictionDecomposition",
    "calculate_shapley_values",
    "ShapleyValues",
    "EditableBooster",
    "BoosterHeader",
    "BoosterTree",
    "LightGBMTabularTrees",
    "BoosterString",
    "ScikitLearnHistTabularTrees",
    "ScikitLearnTabularTrees",
    "XGBoostTabularTrees",
    "XGBoostParser",
    "ParsedXGBoostTabularTrees",
    "TabularTrees",
    "export_tree_data",
    "validate_monotonic_constraints",
    "MonotonicConstraintResults",
]
