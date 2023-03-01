"""Module for validating monotonic trends in trees."""

import pandas as pd

from .checks import check_condition, check_type
from .explain import _convert_node_columns_to_integer
from .trees import TabularTrees


def validate_monotonic_constraints(
    tabular_trees: TabularTrees,
    constraints: dict[str, int],
    return_detailed_results: bool = False,
) -> pd.DataFrame:
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
) -> pd.DataFrame:
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
    monotonicity_check_list = []

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

                # check all nodes below each node which splits on the variable of interest
                # note, this could be made more efficient by
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

    constraint_results = (
        pd.concat(monotonicity_check_list, axis=0)
        .sort_values(["variable", "tree", "node"])
        .reset_index(drop=True)
    )

    if return_detailed_results:

        return constraint_results

    else:

        summarised_constraint_results = (
            constraint_results.groupby("variable")["monotonic"].mean() == 1
        )

        return summarised_constraint_results


def _traverse_tree_down(
    df,
    node,
    name,
    nodes_list,
    values_list,
    value=None,
    lower=None,
    upper=None,
    print_note="start",
    verbose=False,
):
    """Find a value for variable of interest (name) that would end up in each node in the tree.

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
    value : int, float or None
        Value that has been sent to current node, default value of None is only
        used at the top of the tree.
    lower : int, float or None:
        Lower bound for value of variable (name) that would allow a data point to
        visit current node, default value of None is used from the top of the tree
        until an lower bound is found i.e. right (no) split is traversed.
    upper : int, float or None:
        Upper bound for value of variable (name) that would allow a data point to
        visit current node, default value of None is used from the top of the tree
        until an upper bound is found i.e. left (yes) split is traversed.
    print_note : str
        Note to print if verbose = True, only able to set for first call of
        function.
    verbose : bool
        Should notes be printed as function runs.
    nodes_list : list
        List to record the node numbers that are visited.
    values_list : list
        List to record the values of variable (name) which allow each node to be
        visited (same order as nodes_list).

    Returns
    -------
    No returns from function. However nodes_list (list) and values_list (list)
    which are lists of nodes and corresponding data values of variable of
    interest (name) that would allow each node to be visited - are updated as
    the function runs

    """
    if not isinstance(nodes_list, list):

        raise TypeError("nodes_list must be a list")

    if not isinstance(values_list, list):

        raise TypeError("values_list must be a list")

    if verbose:

        print(
            "_traverse_tree_down call;\n"
            + "\tnode: "
            + str(node)
            + "\n"
            + "\tname: "
            + name
            + "\n"
            + "\tvalue: "
            + str(value)
            + "\n"
            + "\tlower: "
            + str(lower)
            + "\n"
            + "\tupper: "
            + str(upper)
            + "\n"
            + "\tprint_note: "
            + print_note
        )

    # add the current node and value to lists to ultimately be returned from function
    nodes_list.append(node)
    values_list.append(value)

    # if we have reached a terminal node
    if df.loc[df["node"] == node, "leaf"].item() == 1:

        if verbose:

            print(
                "node reached;\n"
                + "\tnode: "
                + str(node)
                + "\n"
                + "\tvalue: "
                + str(value)
                + "\n"
                + "\tnode type: terminal node"
            )

    else:

        if verbose:

            print(
                "node reached;\n"
                + "\tnode: "
                + str(node)
                + "\n"
                + "\tvalue: "
                + str(value)
                + "\n"
                + "\tnode type: internal node"
            )

        # if the split for the current node is on the variable of interest update; value, lower, upper
        if df.loc[df["node"] == node, "feature"].iloc[0] == name:

            # pick a value and update bounds that would send a data point down the left (yes) split

            # record the values for value, lower and upper when the function is called so they can be reset
            # to these values before calling _traverse_tree_down down the no route within this if block
            fcn_call_value = value
            fcn_call_lower = lower
            fcn_call_upper = upper

            # if lower bound is unspecified, this means we have not previously travelled down a right path
            # does not matter if upper bound is specified or not i.e. we have previously travelled down a left
            # path, as going down the left path at the current node updates values in the same way
            if lower is None:

                if verbose:

                    print("values update for recursive call; \n\tcase i. lower None")

                # choose a value above the split point to go down the left (yes) split
                value = df.loc[df["node"] == node, "split_condition"].iloc[0] - 1

                # there is no lower bound on values that will go down this path yet
                lower = None

                # the upper bound that can go down this split is the split condition (as we are now below it)
                upper = df.loc[df["node"] == node, "split_condition"].iloc[0]

            # if lower bound is specified, this means we have previously travelled down a right path
            # does not matter if upper bound is specified or not i.e. we have previously travelled down a left
            # path, as going down the left path at the current node updates values in the same way
            else:

                if verbose:

                    print(
                        "values update for recursive call; \n\tcase iii. lower not None and upper None"
                    )

                # lower bound remains the same
                lower = lower

                # but to send a data point to the left hand split the upper bound is the split condition
                # for this node
                upper = df.loc[df["node"] == node, "split_condition"].iloc[0]

                # set a value that falls between the bounds
                value = (lower + upper) / 2

            # recursively call function down the left child of the current node
            _traverse_tree_down(
                df,
                df.loc[df["node"] == node, "left_child"].iloc[0],
                name,
                nodes_list,
                values_list,
                value,
                lower,
                upper,
                "yes child - variable of interest",
                verbose,
            )

            # now pick a value and update bounds that would send a data point down the right (no) split

            # reset values as the if else blocks above will have changed them in order to go down the
            # yes route with the right child node
            value = fcn_call_value
            lower = fcn_call_lower
            upper = fcn_call_upper

            # if both upper bound is unspecified, this means we have not previously travelled down a left path
            if upper is None:

                if verbose:

                    print(
                        "values update for recursive call; \n\tcase v. lower None and upper None"
                    )

                # choose a value above the split point to go down the right (no) split
                value = df.loc[df["node"] == node, "split_condition"].iloc[0] + 1

                # the lower bound that can go down this split is the split condition (as we are now above it)
                lower = df.loc[df["node"] == node, "split_condition"].iloc[0]

                # there is no upper bound on values that will go down this path yet
                upper = None

            else:

                if verbose:

                    print(
                        "values update for recursive call; \n\tcase vi. lower None and upper not None"
                    )

                # the lower bound becomes the split condition
                lower = df.loc[df["node"] == node, "split_condition"].iloc[0]

                # the upper bound remains the same
                upper = upper

                # set a value that falls between the bounds
                value = (lower + upper) / 2

            _traverse_tree_down(
                df,
                df.loc[df["node"] == node, "right_child"].iloc[0],
                name,
                nodes_list,
                values_list,
                value,
                lower,
                upper,
                "no child - variable of interest",
                verbose,
            )

        # otherwise, if the split for the current node is not on the variable of interest do not update values but
        # continue down the tree
        else:

            _traverse_tree_down(
                df,
                df.loc[df["node"] == node, "left_child"].iloc[0],
                name,
                nodes_list,
                values_list,
                value,
                lower,
                upper,
                "yes child - variable not of interest",
                verbose,
            )

            _traverse_tree_down(
                df,
                df.loc[df["node"] == node, "right_child"].iloc[0],
                name,
                nodes_list,
                values_list,
                value,
                lower,
                upper,
                "no child - variable not of interest",
                verbose,
            )


def gather__traverse_tree_down_results(nodes, values, name):
    """Gather results from _traverse_tree_down into pd.DataFrame."""
    if not isinstance(nodes, list):

        raise TypeError("nodes must be a list")

    if not isinstance(values, list):

        raise TypeError("values must be a list")

    if not isinstance(name, str):

        raise TypeError("name must be a str")

    if len(values) != len(nodes):

        raise ValueError("nodes and values must be of the same length")

    df = pd.DataFrame({"node": nodes, name + "_values": values})

    if df["node"].duplicated().sum() > 0:

        raise ValueError("duplicated nodes; " + str(nodes))

    return df


def _check_monotonicity_at_split(
    tree_df, tree_no, trend, variable, node, child_nodes_left, child_nodes_right
):
    """Check monotonic trend is in place at a given split in a single tree."""
    if not isinstance(tree_df, pd.DataFrame):

        raise TypeError("tree_df should be a pd.DataFrame")

    if not isinstance(tree_no, int):

        raise TypeError("tree_no should be an int")

    if not isinstance(trend, int):

        raise TypeError("trend should be an int")

    if not isinstance(node, int):

        raise TypeError("node should be an int")

    if not isinstance(child_nodes_left, list):

        raise TypeError("child_nodes_left should be an list")

    if not isinstance(child_nodes_right, list):

        raise TypeError("child_nodes_right should be an list")

    all_child_nodes = child_nodes_left + child_nodes_right

    tree_nodes = tree_df["node"].tolist()

    child_nodes_not_in_tree = list(set(all_child_nodes) - set(tree_nodes))

    if len(child_nodes_not_in_tree) > 0:

        raise ValueError(
            "the following child nodes do not appear in tree; "
            + str(child_nodes_not_in_tree)
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

    elif trend == -1:

        if left_nodes_max_pred >= right_nodes_min_pred:

            monotonic = True

        else:

            monotonic = False

    else:

        raise ValueError(
            "unexpected value for trend; "
            + str(trend)
            + " variable; "
            + str(variable)
            + " node:"
            + str(node)
        )

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
