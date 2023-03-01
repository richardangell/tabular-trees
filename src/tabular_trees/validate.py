"""Module for validating xgboost trees."""

from copy import deepcopy

import pandas as pd

from .checks import check_df_columns
from .explainer import _convert_node_columns_to_integer


def validate_monotonic_constraints_df(
    trees_df, constraints, return_detailed_results=False
):
    """Check that monotonic constraints are as expected in an xgboost model."""
    expected_columns = [
        "tree",
        "node",  # "nodeid",
        # "depth",
        "left_child",  # "yes",
        "right_child",  # "no",
        "missing",
        "feature",  # "split",
        "split_condition",
        "leaf",
        "prediction",  # "node_prediction",
        # "node_type",
        # "gain",
        "count",  # "cover",
        # "H",
        # "G",
    ]

    check_df_columns(df=trees_df, expected_columns=expected_columns)

    constraints = deepcopy(constraints)

    if not isinstance(constraints, dict):

        raise TypeError("constraints should be a dict")

    # dictionary to hold results of monotonic constraint checks
    constraint_results = {}

    for k, v in constraints.items():

        if not isinstance(v, int):

            raise TypeError('constraints["' + str(k) + '"] is not an int')

        if v < -1 | v > 1:

            raise ValueError('constraints["' + str(k) + '"] is not one of; -1, 0, 1')

        # reduce constraints down to the ones with constraints
        if v == 0:

            del constraints[k]

        constraint_results[k] = {}

    if not isinstance(trees_df, pd.DataFrame):

        raise TypeError("trees_df should be a pd.DataFrame")

    n = trees_df["tree"].max()

    monotonicity_check_list = []

    # loop through each tree
    for i in range(n):

        tree_df = trees_df.loc[trees_df["tree"] == i].copy()
        tree_df = _convert_node_columns_to_integer(tree_df)

        # loop throguh each constraint
        for k, v in constraints.items():

            # if the constraint variable is used in the given tree
            if (tree_df["feature"] == k).sum() > 0:

                # get all nodes that are split on the variable of interest
                nodes_split_on_variable = tree_df.loc[
                    tree_df["feature"] == k, "node"
                ].tolist()

                # check all nodes below each node which splits on the variable of interest
                for n in nodes_split_on_variable:

                    child_nodes_left = []
                    child_values_left = []

                    child_nodes_right = []
                    child_values_right = []

                    traverse_tree_down(
                        df=tree_df,
                        node=tree_df.loc[tree_df["node"] == n, "left_child"].iloc[0],
                        name=k,
                        nodes_list=child_nodes_left,
                        values_list=child_values_left,
                    )

                    traverse_tree_down(
                        df=tree_df,
                        node=tree_df.loc[tree_df["node"] == n, "right_child"].iloc[0],
                        name=k,
                        nodes_list=child_nodes_right,
                        values_list=child_values_right,
                    )

                    # constraint_results[k][i][n]
                    # check that monotonic constraint v is applied on variable k for tree i at node n
                    x = check_monotonicity_at_split(
                        tree_df=tree_df,
                        tree_no=i,
                        trend=v,
                        variable=k,
                        node=n,
                        child_nodes_left=child_nodes_left,
                        child_nodes_right=child_nodes_right,
                    )

                    monotonicity_check_list.append(x)

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


def traverse_tree_down(
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
            "traverse_tree_down call;\n"
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
            # to these values before calling traverse_tree_down down the no route within this if block
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
            traverse_tree_down(
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

            traverse_tree_down(
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

            traverse_tree_down(
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

            traverse_tree_down(
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


def gather_traverse_tree_down_results(nodes, values, name):
    """Gather results from traverse_tree_down into pd.DataFrame."""
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


def check_monotonicity_at_split(
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
