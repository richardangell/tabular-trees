"""tabular_trees"""

from ._version import __version__

from . import checks
from . import trees

try:
    from . import xgboost
except ImportError:
    pass
