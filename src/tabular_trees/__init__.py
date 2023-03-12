"""tabular_trees module."""

from importlib import metadata

from . import checks, explain, validate
from .trees import BaseModelTabularTrees, TabularTrees, export_tree_data

# single source for version number is in the pyproject.toml
# note, for an editable install, the package version number will not be
# updated until the package is reinstalled with `poetry install`
__version__ = metadata.version(__package__)

# avoids polluting the results of dir(__package__)
del metadata

try:
    from . import lightgbm
except ImportError:
    pass

try:
    from . import sklearn
except ImportError:
    pass

try:
    from . import xgboost
except ImportError:
    pass
