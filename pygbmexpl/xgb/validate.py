import pandas as pd
from copy import deepcopy

from pygbmexpl.helpers import check_df_columns
from pygbmexpl.xgb.parser import EXPECTED_COLUMNS



def validate_monotonic_constraints_df(trees_df, constraints):
    '''Function to check that monotonic constraints are as expected in an xgboost model.'''

    check_df_columns(
        df = trees_df,
        expected_columns = EXPECTED_COLUMNS['tree_df_with_node_predictions']
    )

    constraints = deepcopy(constraints)

    if not isinstance(constraints, dict):

        raise TypeError('constraints should be a dict')

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

        raise TypeError('trees_df should be a pd.DataFrame')

    n = trees_df['tree'].max()

    # loop through each tree
    for i in range(n):

        tree_df = trees_df.loc[trees_df['tree'] == i].copy()

        # loop throguh each constraint
        for k, v in constraints.items():

            # if the constraint variable is used in the given tree 
            if (tree_df['split'] == k).sum() > 0:
                
                all_nodes_checked = []

                # get all nodes that are split on the variable of interest
                nodes_split_on_variable = tree_df.loc[tree_df['split'] == k, 'nodeid'].tolist()

                constraint_results[k][i] = {}

                # check all nodes below each node which splits on the variable of interest
                for n in nodes_split_on_variable:    
                    
                    # if the node has not been checked before, in checking a node higher up the tree
                    if not n in all_nodes_checked:

                        nodes = []
                        values = []

                        traverse_tree_down(
                            df = tree_df, 
                            node = n, 
                            name = k, 
                            nodes_list = nodes, 
                            values_list = values
                        )

                        # append all the nodes checked in this loop to the list of all nodes that have been checked
                        # for this variable k in this tree i
                        all_nodes_checked = all_nodes_checked + nodes

                        nodes_values_df = gather_traverse_tree_down_results(
                            nodes = nodes, 
                            values = values, 
                            name = k
                        )

                        # merge on values of variable k that would allow a data point to visit each node
                        # remove rows from tree_df that were not visited starting from node n going down the tree with inner join
                        tree_df2 = tree_df.merge(
                            right = nodes_values_df,
                            how = 'inner',
                            on = 'nodeid'
                        )

                        # check that monotonic constraint v is applied on variable k for tree i
                        constraint_results[k][i][n] = check_1way_node_trend(
                            df = tree_df2,
                            trend = v,
                            variable = k
                        )

    return constraint_results



def traverse_tree_down(df, 
                       node, 
                       name, 
                       nodes_list, 
                       values_list,
                       value = None, 
                       lower = None, 
                       upper = None, 
                       print_note = 'start', 
                       verbose = False):
    '''Function to find a value for variable of interest (name) that would end up in each node in the tree.
    
    Args:
        df (pd.DataFrame): single tree structure output from parser.read_dump_text() or parser.read_dump_json()
        node (int): node number 
        name (str): name of variable of interest, function will determine values for this variable that would allow a data point
            to visit each node
        value (int or float or None): value that has been sent to current node, default value of None is only
            used at the top of the tree
        lower (int or float or None): lower bound for value of variable (name) that would allow a data point to visit current node,
            default value of None is used from the top of the tree until an lower bound is found i.e. right (no) split is traversed 
        upper (int or float or None): upper bound for value of variable (name) that would allow a data point to visit current node,
            default value of None is used from the top of the tree until an upper bound is found i.e. left (yes) split is traversed 
        print_note (str): note to print if verbose = True, only able to set for first call of function
        verbose (bool): should notes be printed as function runs
        nodes_list (list): list to record the node numbers that are visited
        values_list (list): list to record the values of variable (name) which allow each node to be visited (same order as nodes_list)

    Returns:
        No returns from function. However nodes_list (list) and values_list (list) which are lists of nodes and corresponding data values of 
        variable of interest (name) that would allow each node to be visited - are updated as the function runs

    '''
    
    if not isinstance(nodes_list, list):

        raise TypeError('nodes_list must be a list')

    if not isinstance(values_list, list):

        raise TypeError('values_list must be a list')

    if verbose:
    
        print(
            'traverse_tree_down call;\n' +
            '\tnode: ' + str(node) + '\n' +
            '\tname: ' + name + '\n' +
            '\tvalue: ' + str(value) + '\n' +
            '\tlower: ' + str(lower) + '\n' +
            '\tupper: ' + str(upper) + '\n' + 
            '\tprint_note: ' + print_note
        )
    
    # add the current node and value to lists to ultimately be returned from function
    nodes_list.append(node)
    values_list.append(value)
    
    # if we have reached a terminal node
    if df.loc[df['nodeid'] == node, 'split'].isnull().iloc[0]:
        
        if verbose:
        
            print(
                'node reached;\n' +
                '\tnode: ' + str(node) + '\n' +
                '\tvalue: ' + str(value) + '\n' +
                '\tnode type: terminal node'
            )
            
    else:
    
        if verbose:
        
            print(
                'node reached;\n' +
                '\tnode: ' + str(node) + '\n' +
                '\tvalue: ' + str(value) + '\n' +
                '\tnode type: internal node'
            )   
    
        # if the split for the current node is on the variable of interest update; value, lower, upper 
        if df.loc[df['nodeid'] == node, 'split'].iloc[0] == name:
           
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
                
                    print(
                        'values update for recursive call; \n\tcase i. lower None'
                    )
                
                # choose a value above the split point to go down the left (yes) split
                value = df.loc[df['nodeid'] == node, 'split_condition'].iloc[0] - 1

                # there is no lower bound on values that will go down this path yet
                lower = None

                # the upper bound that can go down this split is the split condition (as we are now below it)
                upper = df.loc[df['nodeid'] == node, 'split_condition'].iloc[0]

            # if lower bound is specified, this means we have previously travelled down a right path
            # does not matter if upper bound is specified or not i.e. we have previously travelled down a left
            # path, as going down the left path at the current node updates values in the same way 
            else:
                
                if verbose:
                
                    print(
                        'values update for recursive call; \n\tcase iii. lower not None and upper None'
                    )                
        
                # lower bound remains the same
                lower = lower
                
                # but to send a data point to the left hand split the upper bound is the split condition 
                # for this node
                upper = df.loc[df['nodeid'] == node, 'split_condition'].iloc[0]
                
                # set a value that falls between the bounds 
                value = (lower + upper) / 2

            # recursively call function down the left child of the current node
            traverse_tree_down(
                df, 
                df.loc[df['nodeid'] == node, 'yes'].iloc[0], 
                name, 
                nodes_list, 
                values_list, 
                value, 
                lower, 
                upper, 
                'yes child - variable of interest', 
                verbose
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
                        'values update for recursive call; \n\tcase v. lower None and upper None'
                    )
                
                # choose a value above the split point to go down the right (no) split
                value = df.loc[df['nodeid'] == node, 'split_condition'].iloc[0] + 1

                # the lower bound that can go down this split is the split condition (as we are now above it)
                lower = df.loc[df['nodeid'] == node, 'split_condition'].iloc[0]

                # there is no upper bound on values that will go down this path yet
                upper = None

            else:

                if verbose:
                
                    print(
                        'values update for recursive call; \n\tcase vi. lower None and upper not None'
                    )                    
                                    
                # the lower bound becomes the split condition
                lower = df.loc[df['nodeid'] == node, 'split_condition'].iloc[0]

                # the upper bound remains the same
                upper = upper
                
                # set a value that falls between the bounds 
                value = (lower + upper) / 2                    

            traverse_tree_down(
                df, 
                df.loc[df['nodeid'] == node, 'no'].iloc[0], 
                name, 
                nodes_list, 
                values_list,
                value, 
                lower, 
                upper, 
                'no child - variable of interest', 
                verbose
            )

        # otherwise, if the split for the current node is not on the variable of interest do not update values but 
        # continue down the tree       
        else:

            traverse_tree_down(
                df, 
                df.loc[df['nodeid'] == node, 'yes'].iloc[0], 
                name, 
                nodes_list, 
                values_list,
                value, 
                lower, 
                upper, 
                'yes child - variable not of interest', 
                verbose
            )

            traverse_tree_down(
                df, 
                df.loc[df['nodeid'] == node, 'no'].iloc[0], 
                name, 
                nodes_list, 
                values_list,
                value, 
                lower, 
                upper, 
                'no child - variable not of interest', 
                verbose
            )



def gather_traverse_tree_down_results(nodes, values, name):
    '''Function to gather results from traverse_tree_down into pd.DataFrame.'''

    if not isinstance(nodes, list):

        raise TypeError('nodes must be a list')

    if not isinstance(values, list):

        raise TypeError('values must be a list')

    if not isinstance(name, str):

        raise TypeError('name must be a str')

    if len(values) != len(nodes):

        raise ValueError('nodes and values must be of the same length')

    df = pd.DataFrame({'nodeid': nodes, name + '_values': values})

    if df['nodeid'].duplicated().sum() > 0:

        raise ValueError('duplicated nodes; ' + str(nodes))

    return df



def check_1way_node_trend(df, trend, variable):
    '''Function to check monotonic trend is in place for single tree.'''

    if not isinstance(df, pd.DataFrame):

        raise TypeError('df should be a pd.DataFrame')

    if not isinstance(trend, int):

        raise TypeError('trend should be an int')

    values_col = str(variable) + '_values'
    monotonic_check_col = str(variable) + '_monotonic_check'
    
    if not values_col in df.columns.values:
        
        raise ValueError(values_col + ' column not in df')
    
    # set sorting order for columns; values_col, weight - True = ascending
    if trend == 1:

        sort_order = [True, True]
    
    # if the monotonic constraint is decreasing (-1) then order the predicted values
    # at each node in decreasing order
    elif trend == -1:

        sort_order = [True, False]

    else:

        raise ValueError('unexpected value for trend; ' + str(trend) + ' variable; ' + str(variable))
    
    select_columns = EXPECTED_COLUMNS['tree_df_with_node_predictions'] + [values_col]
    
    # select the terminal / leaf nodes and nodes where the values column for this variable is not null
    # if the values column is null this means the variable interest does not affect predictions
    # for these nodes
    terminal_nodes = df.loc[(df['node_type'] == 'leaf') & (~df[values_col].isnull()), select_columns] \
        .sort_values(by = [values_col, 'weight'], ascending = sort_order).copy()
    
    weight_group_shift = terminal_nodes.groupby(values_col)['weight'].last().shift(1).reset_index(name = 'weight_shift_group')
    
    terminal_nodes = terminal_nodes.merge(weight_group_shift, how = 'left', on = values_col)
    
    # if the trend to check is increasing then check each node (grouped by values column) has a larger
    # prediction then group max in the previous group
    if trend == 1:
    
        terminal_nodes[monotonic_check_col] = terminal_nodes['weight'] - terminal_nodes['weight_shift_group'] >= 0
    
    # converse for the deacreasing trend case
    else:
        
        terminal_nodes[monotonic_check_col] = terminal_nodes['weight_shift_group'] - terminal_nodes['weight'] >= 0
    
    results = {
        'variable': variable,
        'monotonic_direction': trend,
        'node_info': terminal_nodes
    }
    
    null_rows = terminal_nodes['weight_shift_group'].isnull()
    
    null_row_count = null_rows.sum()
    
    # if all nodes except the ones in the first values group meet the condition the monotonicity trend is preserved
    if terminal_nodes.loc[~null_rows, monotonic_check_col].sum() == (terminal_nodes.shape[0] - null_row_count):
            
        results['monotonic'] = True
            
    else:
        
        results['monotonic'] = False
    
    return results


