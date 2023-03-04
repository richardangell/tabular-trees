"""Module for validating monotonic trends in trees."""

from dataclasses import dataclass, field
from typing import Optional, Union

import pandas as pd

from .checks import check_condition, check_type
from .explain import _convert_node_columns_to_integer
from .trees import TabularTrees


@dataclass
class MonotonicConstraintResults:
    """Results of checking monotonic constraints.

    Attributes
    ----------
    summary : dict[str, bool]
        Summary of whether monotonic constraints are met. Keys give variable names and
        values indicate if the constraint is met.

    constraints : dict[str, int]
        Monotonic constraints. Keys give variable names and values define the
        constraints. A value of 1 is a monotonically increasing constraint and a value
        of -1 is a monotonically decreasing constraint.

    results : pd.DataFrame
        Detailed breakdown of whether constraints are met at split level in trees.

    all_constraints_met : bool
        Are all of the monotonic constraints met.

    """

    summary: dict[str, bool]
    constraints: dict[str, int]
    results: pd.DataFrame = field(repr=False)
    all_constraints_met: bool = field(init=False)

    def __post_init__(self):
        """Set constraints_met attribute."""
        self.all_constraints_met = self._all_constraints_met()

    def _all_constraints_met(self) -> bool:
        """Are constraints for all variables conformed to."""
        return all(self.summary.values())


def validate_monotonic_constraints(
    tabular_trees: TabularTrees,
    constraints: dict[str, int],
    return_detailed_results: bool = False,
) -> MonotonicConstraintResults:
    """Validate that trees conform to monotonic constraints.

    Parameters
    ----------
    tabular_trees : TabularTrees
        Trees to check.

    constraints : dict[str, int]
        Monotonic constraints to check. Should be dict where keys give variable names
        and values are either -1 for monotonic decreasing constraint and 1 for
        monotonic increasing constraint.

    return_detailed_results : bool, defualt=False
        Should detailed breakdown of every split be returned?

    """
    check_type(tabular_trees, TabularTrees, "tabular_trees")
    check_type(constraints, dict, "constraints")
    check_type(return_detailed_results, bool, "return_detailed_results")

    constraints_to_check = {}
    for variable, direction in constraints.items():
        check_type(direction, int, f"constraints[{variable}]")
        check_condition(
            direction in [-1, 0, 1], f"constraints[{variable}] not in [-1, 0, 1]"
        )
        if direction != 0:
            constraints_to_check[variable] = direction

    return _validate_monotonic_constraints(
        trees_df=tabular_trees.trees,
        constraints=constraints_to_check,
        return_detailed_results=return_detailed_results,
    )


def _validate_monotonic_constraints(
    trees_df: pd.DataFrame, constraints: dict[str, int], return_detailed_results: bool
) -> MonotonicConstraintResults:
    """Loop through each tree and check monotonic constraints.

    Parameters
    ----------
    trees_df : pd.DataFrame
        Trees to check. Should be the trees attribute of a TabularTrees object.

    constraints : dict[str, int]
        Monotonic constraints to check. Should be dict where keys give variable names
        and values are either -1 for monotonic decreasing constraint and 1 for
        monotonic increasing constraint.

    return_detailed_results : bool, defualt=False
        Should detailed breakdown of every split be returned?

    """
    monotonicity_check_list: list[pd.DataFrame] = []

    # loop through each tree
    for tree_no in range(trees_df["tree"].max()):

        tree_df = trees_df.loc[trees_df["tree"] == tree_no].copy()
        tree_df = _convert_node_columns_to_integer(tree_df)

        # loop throguh each constraint
        for constraint_variable, constraint_direction in constraints.items():

            # if the constraint variable is used in the given tree
            if (tree_df["feature"] == constraint_variable).sum() > 0:

                # get all nodes that are split on the variable of interest
                nodes_split_on_variable = tree_df.loc[
                    tree_df["feature"] == constraint_variable, "node"
                ].tolist()

                # check all nodes below each node which splits on the variable of
                # interest
                # note, this could be made more efficient by not rechecking lower nodes
                # if they have been covered when checking a node above
                for node_to_check in nodes_split_on_variable:

                    child_nodes_left: list = []
                    child_values_left: list = []

                    child_nodes_right: list = []
                    child_values_right: list = []

                    _traverse_tree_down(
                        df=tree_df,
                        node=tree_df.loc[
                            tree_df["node"] == node_to_check, "left_child"
                        ].item(),
                        name=constraint_variable,
                        nodes_list=child_nodes_left,
                        values_list=child_values_left,
                    )

                    _traverse_tree_down(
                        df=tree_df,
                        node=tree_df.loc[
                            tree_df["node"] == node_to_check, "right_child"
                        ].item(),
                        name=constraint_variable,
                        nodes_list=child_nodes_right,
                        values_list=child_values_right,
                    )

                    # check that monotonic constraint {constraint_direction} is applied
                    # on variable {constraint_variable} for tree {tree_no} at node
                    # {node_to_check}
                    check_results = _check_monotonicity_at_split(
                        tree_df=tree_df,
                        tree_no=tree_no,
                        trend=constraint_direction,
                        variable=constraint_variable,
                        node=node_to_check,
                        child_nodes_left=child_nodes_left,
                        child_nodes_right=child_nodes_right,
                    )

                    monotonicity_check_list.append(check_results)

    return _format_constraint_results(monotonicity_check_list, constraints)


def _format_constraint_results(
    monotonicity_check_list: list[pd.DataFrame], constraints: dict[str, int]
) -> MonotonicConstraintResults:
    """Create combined check results across all variables, trees and nodes."""
    constraint_results = (
        pd.concat(monotonicity_check_list, axis=0)
        .sort_values(["variable", "tree", "node"])
        .reset_index(drop=True)
    )

    summarised_constraint_results = (
        constraint_results.groupby("variable")["monotonic"].mean() == 1
    ).to_dict()

    return MonotonicConstraintResults(
        summary=summarised_constraint_results,
        constraints=constraints,
        results=constraint_results,
    )


def _traverse_tree_down(
    df: pd.DataFrame,
    node: int,
    name: str,
    nodes_list: list,
    values_list: list,
    value: Optional[Union[int, float]] = None,
    lower: Optional[Union[int, float]] = None,
    upper: Optional[Union[int, float]] = None,
) -> None:
    """Find a value for variable that would end up in each node in the tree.

    Parameters
    ----------
    df : pd.DataFrame
        Single tree structure output from parser.read_dump_text() or
        parser.read_dump_json().

    node : int
        Node number.

    name : str
        Name of variable of interest, function will determine values for this
        variable that would allow a data point to visit each node.

    nodes_list : list
        List to record the node numbers that are visited.

    values_list : list
        List to record the values of variable {name} which allow each node to be
        visited. In the same order as nodes_list.

    value : int, float, default = None
        Value that has been sent to current node, default value of None is only
        used at the top of the tree.

    lower : int, float, default =  None:
        Lower bound for value of variable {name} that would allow a data point to
        visit current node. Default value of None is used from the top of the tree
        until an lower bound is found i.e. right (no) split is traversed.

    upper : int, float, default =  None:
        Upper bound for value of variable {name} that would allow a data point to
        visit current node. Default value of None is used from the top of the tree
        until an upper bound is found i.e. left (yes) split is traversed.

    Returns
    -------
    None. However nodes_list and values_list which are lists of nodes and corresponding
    data values of variable of interest {name} that would allow each node to be visited
     - are updated as the function descends the tree.

    """
    # add the current node and value to lists to ultimately be returned from function
    nodes_list.append(node)
    values_list.append(value)

    # if we have reached a terminal node
    if df.loc[df["node"] == node, "leaf"].item() == 1:

        pass

    else:

        # if the split for the current node is on the variable of interest update the
        # following; value, lower, upper
        if df.loc[df["node"] == node, "feature"].item() == name:

            # pick a value and update bounds that would send a data point down the left
            # (yes) split

            # record the values for value, lower and upper when the function is called
            # so they can be reset to these values before calling _traverse_tree_down
            # down the no route within this if block
            fcn_call_value = value
            fcn_call_lower = lower
            fcn_call_upper = upper

            # if lower bound is unspecified, this means we have not previously
            # travelled down a right path does not matter if upper bound is specified
            # or not i.e. we have previously travelled down a left path, as going down
            # the left path at the current node updates values in the same way
            if lower is None:

                # choose a value above the split point to go down the left (yes) split
                value = df.loc[df["node"] == node, "split_condition"].item() - 1

                # there is no lower bound on values that will go down this path yet
                lower = None

                # the upper bound that can go down this split is the split condition
                # (as we are now below it)
                upper_split = df.loc[df["node"] == node, "split_condition"].item()

            # if lower bound is specified, this means we have previously travelled down
            # a right path does not matter if upper bound is specified or not i.e. we
            # have previously travelled down a left path, as going down the left path
            # at the current node updates values in the same way
            else:

                # lower bound remains the same
                lower = lower

                # but to send a data point to the left hand split the upper bound is
                # the split condition for this node
                upper_split = df.loc[df["node"] == node, "split_condition"].item()

                # set a value that falls between the bounds
                value = (lower + upper_split) / 2

            # recursively call function down the left child of the current node
            _traverse_tree_down(
                df,
                df.loc[df["node"] == node, "left_child"].item(),
                name,
                nodes_list,
                values_list,
                value,
                lower,
                upper_split,
            )

            # now pick a value and update bounds that would send a data point down the
            # right (no) split

            # reset values as the if else blocks above will have changed them in order
            # to go down the yes route with the right child node
            value = fcn_call_value
            lower = fcn_call_lower
            upper = fcn_call_upper

            # if both upper bound is unspecified, this means we have not previously
            # travelled down a left path
            if upper is None:

                # choose a value above the split point to go down the right (no) split
                value = df.loc[df["node"] == node, "split_condition"].item() + 1

                # the lower bound that can go down this split is the split condition
                # (as we are now above it)
                lower_split = df.loc[df["node"] == node, "split_condition"].item()

                # there is no upper bound on values that will go down this path yet
                upper = None

            else:

                # the lower bound becomes the split condition
                lower_split = df.loc[df["node"] == node, "split_condition"].item()

                # the upper bound remains the same
                upper = upper

                # set a value that falls between the bounds
                value = (lower_split + upper) / 2

            _traverse_tree_down(
                df,
                df.loc[df["node"] == node, "right_child"].item(),
                name,
                nodes_list,
                values_list,
                value,
                lower_split,
                upper,
            )

        # otherwise, if the split for the current node is not on the variable of
        # interest do not update values but continue down the tree
        else:

            _traverse_tree_down(
                df,
                df.loc[df["node"] == node, "left_child"].item(),
                name,
                nodes_list,
                values_list,
                value,
                lower,
                upper,
            )

            _traverse_tree_down(
                df,
                df.loc[df["node"] == node, "right_child"].item(),
                name,
                nodes_list,
                values_list,
                value,
                lower,
                upper,
            )


def _check_monotonicity_at_split(
    tree_df: pd.DataFrame,
    tree_no: int,
    trend: int,
    variable: str,
    node: int,
    child_nodes_left: list[int],
    child_nodes_right: list[int],
) -> pd.DataFrame:
    """Check monotonic trend is in place at a given split in a single tree.

    This involves checking the maximum leaf prediction for the left child nodes against
    the minimum leaf prediction for the right child nodes. E.g. for a monotonically
    increasing trend (trend = 1) all the left child leaf nodes should be less than or
    equal to all of the right child leaf nodes.

    Parameters
    ----------
    tree_df : pd.DataFrame
        Single tree data. Subset of TabularTrees.trees.

    tree_no : int
        Tree number of tree being checked.

    trend : int
        Trend being checked. Either 1 for monotonically increasing and -1 for
        monotonically decreasing trend.

    variable : str
        Name of the variable being checked at the current node.

    node : int
        Node number being checked.

    child_nodes_left : list[int]
        List of {node}'s left child nodes.

    child_nodes_right : list[int]
        List of {node}'s right child nodes.

    """
    tree_nodes = tree_df["node"].tolist()
    all_child_nodes = child_nodes_left + child_nodes_right

    child_nodes_not_in_tree = list(set(all_child_nodes) - set(tree_nodes))
    if len(child_nodes_not_in_tree) > 0:
        raise ValueError(
            f"the following child nodes do not appear in tree; {child_nodes_not_in_tree}"
        )

    left_nodes_max_pred = tree_df.loc[
        tree_df["node"].isin(child_nodes_left), "prediction"
    ].max()

    right_nodes_min_pred = tree_df.loc[
        tree_df["node"].isin(child_nodes_right), "prediction"
    ].min()

    if trend == 1:
        if left_nodes_max_pred <= right_nodes_min_pred:
            monotonic = True
        else:
            monotonic = False
    else:
        if left_nodes_max_pred >= right_nodes_min_pred:
            monotonic = True
        else:
            monotonic = False

    results = {
        "variable": variable,
        "tree": tree_no,
        "node": node,
        "monotonic_trend": trend,
        "monotonic": monotonic,
        "child_nodes_left_max_prediction": left_nodes_max_pred,
        "child_nodes_right_min_prediction": right_nodes_min_pred,
        "child_nodes_left": str(child_nodes_left),
        "child_nodes_right": str(child_nodes_right),
    }

    results_df = pd.DataFrame(results, index=[node])

    return results_df
