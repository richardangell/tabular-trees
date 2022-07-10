import pandas as pd
from dataclasses import dataclass, field

from .. import checks


@dataclass
class XGBoostTabularTrees:
    """Class to hold the xgboost models in tabular format."""

    trees: pd.DataFrame
    n_trees: int = field(init=False)

    REQUIRED_COLUMNS = [
        "Tree",
        "Node",
        "ID",
        "Feature",
        "Split",
        "Yes",
        "No",
        "Missing",
        "Gain",
        "Cover",
        "Category",
    ]

    def __post_init__(self):

        checks.check_type(self.trees, pd.DataFrame, "trees")
        checks.check_df_columns(self.trees, self.REQUIRED_COLUMNS)

        self.n_trees = int(self.trees["Tree"].max())

    def get_trees(self, tree_indexes: list[int]):
        """Return the tabular data for specified tree(s) from model."""

        checks.check_type(tree_indexes, list, "tree_indexes")

        for i, tree_index in enumerate(tree_indexes):

            checks.check_type(tree_index, int, f"tree_indexes[{i}]")
            checks.check_condition(tree_index >= 0, f"tree_indexes[{i}] >= 0")
            checks.check_condition(
                tree_index <= self.n_trees,
                f"tree_indexes[{i}] in range for number of trees ({self.n_trees})",
            )

        return self.trees.loc[self.trees["Tree"].isin(tree_indexes)].copy()
