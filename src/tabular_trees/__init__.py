"""tabular_trees module."""

import contextlib
from importlib import metadata

from .explain.prediction_decomposition import (
    PredictionDecomposition,
    decompose_prediction,
)
from .explain.shapley_values import ShapleyValues, calculate_shapley_values
from .lightgbm.booster_string import BoosterString
from .lightgbm.editable_booster import EditableBooster
from .lightgbm.lightgbm_tabular_trees import LightGBMTabularTrees
from .sklearn.sklearn_hist_tabular_trees import ScikitLearnHistTabularTrees
from .sklearn.sklearn_tabular_trees import ScikitLearnTabularTrees
from .trees import TabularTrees, export_tree_data
from .validate import MonotonicConstraintResults, validate_monotonic_constraints
from .xgboost.dump_parser import ParsedXGBoostTabularTrees, XGBoostParser
from .xgboost.xgboost_tabular_trees import XGBoostTabularTrees

# single source for version number is in the pyproject.toml
# note, for an editable install, the package version number will not be
# updated until the package is reinstalled with `poetry install`
__version__ = metadata.version(__package__)

# avoids polluting the results of dir(__package__)
del metadata

with contextlib.suppress(ImportError):
    from .lightgbm import lightgbm_tabular_trees

with contextlib.suppress(ImportError):
    from .sklearn import sklearn_tabular_trees

with contextlib.suppress(ImportError):
    from .xgboost import xgboost_tabular_trees

__all__ = [
    "decompose_prediction",
    "PredictionDecomposition",
    "calculate_shapley_values",
    "ShapleyValues",
    "EditableBooster",
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
