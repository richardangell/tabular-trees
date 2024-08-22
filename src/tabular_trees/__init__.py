"""tabular_trees module."""

import contextlib
from importlib import metadata

from . import checks, validate
from .explain import prediction_decomposition, shapley_values
from .trees import BaseModelTabularTrees, TabularTrees, export_tree_data

# single source for version number is in the pyproject.toml
# note, for an editable install, the package version number will not be
# updated until the package is reinstalled with `poetry install`
__version__ = metadata.version(__package__)

# avoids polluting the results of dir(__package__)
del metadata

with contextlib.suppress(ImportError):
    from .lightgbm import lightgbm_tabular_trees

with contextlib.suppress(ImportError):
    from .sklearn import scikit_learn_tabular_trees

with contextlib.suppress(ImportError):
    from .xgboost import xgboost_tabular_trees
