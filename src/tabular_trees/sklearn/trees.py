import pandas as pd
from dataclasses import dataclass, field

from .. import checks


@dataclass
class ScikitLearnHistTabularTrees:
    """Class to hold the scikit-learn HistGradientBoosting trees in tabular
    format.

    Parameters
    ----------
    trees : pd.DataFrame
        HistGradientBoostingRegressor or Classifier tree data extracted from
        ._predictors attribute.

    Attributes
    ----------
    n_trees : int
        Number of trees in the model. Indexed from 0.

    """

    trees: pd.DataFrame
    n_trees: int = field(init=False)

    REQUIRED_COLUMNS = [
        "tree",
        "node",
        "value",
        "count",
        "feature_idx",
        "num_threshold",
        "missing_go_to_left",
        "left",
        "right",
        "gain",
        "depth",
        "is_leaf",
        "bin_threshold",
        "is_categorical",
        "bitset_idx",
    ]

    def __post_init__(self):
        """Post init checks and processing.

        Number of trees in the model is calculated and stored in the n_trees
        atttribute.

        Processing on the trees attribute is as follows;
        - Columns are ordered into REQUIRED_COLUMNS order
        - Rows are sorted by tree and node columns
        - The index is reset and original index dropped.

        Raises
        ------
        TypeError
            If self.trees is not a pd.DataFrame.

        ValueError
            If REQUIRED_COLUMNS are not in self.trees.

        """

        checks.check_type(self.trees, pd.DataFrame, "trees")
        checks.check_df_columns(self.trees, self.REQUIRED_COLUMNS)

        self.n_trees = int(self.trees["tree"].max())

        # reorder columns and sort
        self.trees = self.trees[self.REQUIRED_COLUMNS]
        self.trees = self.trees.sort_values(["tree", "node"])

        self.trees = self.trees.reset_index(drop=True)
