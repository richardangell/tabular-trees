


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
    if df.loc[df['index'] == node, 'split'].isnull().iloc[0]:
        
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
    
        # if the split for the current node is on the variable of interest update value, lower, upper 
        # otherwise keep them the same
        if df.loc[df['index'] == node, 'split'].iloc[0] == name:
           
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
                value = df.loc[df['index'] == node, 'split_condition'].iloc[0] - 1

                # there is no lower bound on values that will go down this path yet
                lower = None

                # the upper bound that can go down this split is the split condition (as we are now below it)
                upper = df.loc[df['index'] == node, 'split_condition'].iloc[0]

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
                upper = df.loc[df['index'] == node, 'split_condition'].iloc[0]
                
                # set a value that falls between the bounds 
                value = (lower + upper) / 2

            print(type(nodes_list), '\n', nodes_list)

            # recursively call function down the left child of the current node
            traverse_tree_down(
                df, 
                df.loc[df['index'] == node, 'yes'].iloc[0], 
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
            
            if lower is None:

                # if both upper and lower bounds are unspecified
                # in this case, this is the first time that name has been split on
                if upper is None:
                    
                    if verbose:
                    
                        print(
                            'values update for recursive call; \n\tcase v. lower None and upper None'
                        )
                    
                    # choose a value above the split point to go down the right (no) split
                    value = df.loc[df['index'] == node, 'split_condition'].iloc[0] + 1

                    # the lower bound that can go down this split is the split condition (as we are now above it)
                    lower = df.loc[df['index'] == node, 'split_condition'].iloc[0]

                    # there is no upper bound on values that will go down this path yet
                    upper = None

                # if lower bound is unspecified and upper bound is specified
                else:

                    if verbose:
                    
                        print(
                            'values update for recursive call; \n\tcase vi. lower None and upper not None'
                        )                    
                                        
                    # the lower bound becomes the split condition
                    lower = df.loc[df['index'] == node, 'split_condition'].iloc[0]

                    # the upper bound remains the same
                    upper = upper
                    
                    # set a value that falls between the bounds 
                    value = (lower + upper) / 2                    

            else:
                
                # is lower bound is specified and upper bound is not specified
                if upper is None:
            
                    if verbose:
                    
                        print(
                            'values update for recursive call; \n\tcase vii. lower not None and upper None'
                        )                
                        
                    # lower bound remains the same
                    lower = df.loc[df['index'] == node, 'split_condition'].iloc[0]
                    
                    # but to send a data point to the left hand split the upper bound is the split condition 
                    # for this node
                    upper = None
                    
                    # set a value above the split condition
                    value = df.loc[df['index'] == node, 'split_condition'].iloc[0] + 1
                    
                # if both lower and upper bounds are specified
                else:
            
                    if verbose:
                    
                        print(
                            'values update for recursive call; \n\tcase viii. lower not None and upper not None'
                        )                
                        
                    lower = df.loc[df['index'] == node, 'split_condition'].iloc[0]
                    
                    upper = upper 
                    
                    # set a value that falls between the bounds 
                    value = (lower + upper) / 2
                    
            traverse_tree_down(
                df, 
                df.loc[df['index'] == node, 'no'].iloc[0], 
                name, 
                nodes_list, 
                values_list,
                value, 
                lower, 
                upper, 
                'no child - variable of interest', 
                verbose
            )
                        
        else:

            traverse_tree_down(
                df, 
                df.loc[df['index'] == node, 'yes'].iloc[0], 
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
                df.loc[df['index'] == node, 'no'].iloc[0], 
                name, 
                nodes_list, 
                values_list,
                value, 
                lower, 
                upper, 
                'no child - variable not of interest', 
                verbose
            )







