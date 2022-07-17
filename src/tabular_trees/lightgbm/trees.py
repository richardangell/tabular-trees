import pandas as pd
from dataclasses import dataclass, field

from .. import checks


@dataclass
class LightGBMTabularTrees:
    """Class to hold the xgboost trees in tabular format.

    Parameters
    ----------
    trees : pd.DataFrame
        LightGBM tree data output from Booster.trees_to_dataframe.

    Attributes
    ----------
    n_trees : int
        Number of trees in the model. Indexed from 0.

    """

    trees: pd.DataFrame
    n_trees: int = field(init=False)

    REQUIRED_COLUMNS = [
        "tree_index",
        "node_depth",
        "node_index",
        "left_child",
        "right_child",
        "parent_index",
        "split_feature",
        "split_gain",
        "threshold",
        "decision_type",
        "missing_direction",
        "missing_type",
        "value",
        "weight",
        "count",
    ]

    def __post_init__(self):
        """Post init checks and processing.

        Number of trees in the model is calculated and stored in the n_trees
        atttribute.

        Processing on the trees attribute is as follows;
        - Columns are ordered into REQUIRED_COLUMNS order
        - Rows are sorted by tree_index, node_depth and node_index columns
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

        self.n_trees = int(self.trees["tree_index"].max())

        # reorder columns and sort
        self.trees = self.trees[self.REQUIRED_COLUMNS]
        self.trees = self.trees.sort_values(["tree_index", "node_depth", "node_index"])

        self.trees = self.trees.reset_index(drop=True)
