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

    for k, v in constraints.items():

        if not isinstance(v, int):

            raise TypeError('constraints["' + str(k) + '"] is not an int')

        if v < -1 | v > 1:

            raise ValueError('constraints["' + str(k) + '"] is not one of; -1, 0, 1')

        # reduce constraints down to the ones with constraints
        if v == 0:

            del constraints[k]

    if not isinstance(trees_df, pd.DataFrame):

        raise TypeError('trees_df should be a pd.DataFrame')

    n = trees_df['tree'].max()

    constraint_results = {}

    # loop throguh each constraint
    for k, v in constraints.items():

        constraint_results[k] = {}

        # loop through each tree
        for i in range(n):

            # if the constraint variable is used in the given tree 
            if (trees_df.loc[trees_df['tree'] == i, 'split'] == k).sum() > 0:
                
                nodes = []
                values = []

                tree_df = trees_df.loc[trees_df['tree'] == i].copy()

                traverse_tree_down(
                    df = tree_df, 
                    node = 0, 
                    name = k, 
                    nodes_list = nodes, 
                    values_list = values
                )

                nodes_values_df = gather_traverse_tree_down_results(
                    nodes = nodes, 
                    values = values, 
                    name = k
                )

                # merge on values of variable k that would allow a data point to visit each node
                tree_df2 = tree_df.merge(
                    right = nodes_values_df,
                    how = 'left',
                    on = 'nodeid',
                    indicator = True
                )

                # check all nodes got a value merged on
                if (tree_df2['_merge'] == 'both').sum() < tree_df2.shape[0]:

                    raise ValueError('not all nodes recieved a value from nodes_values_df; tree: ' + str(i) + ' variable: ' + str(k))

                # check that monotonic constraint v is applied on variable k for tree i
                constraint_results[k][i] = check_1way_node_trend(
                    df = tree_df2,
                    trend = v
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

    if df['nodeid'].duplciated().sum() > 0:

        raise ValueError('duplicated nodes; ' + str(nodes))

    return df


