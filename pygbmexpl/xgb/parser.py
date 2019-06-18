import pandas as pd
import json
from copy import deepcopy



def read_dump_json(file, return_raw_lines = True):
    '''Reads an xgboost model dump json file and parses it into a tabular structure.

    Json file to read must be the output from xgboost.Booster.dump_model with dump_format = 'json'. 
    Note this argument was only added in 0.81 and the default prior to this release was dump in 
    text format.

    Args:
        file (str): xgboost model dump json file.
        return_raw_lines (bool): should lines read from the json file be returned in a dict as well?

    Returns: 
        pd.DataFrame: df with columns; tree, nodeid, depth, yes, no, missing, split, split_condition, leaf.
        If the model dump file was output with  with_stats = True then gain and cover columns are also
        in the output.

    '''

    with open(file) as f:

        j = json.load(f)
    
    j_copy = deepcopy(j)

    tree_list = []

    for i in range(len(j)):

        l = []

        recursive_pop_children(n = j[i], l = l, verbose = False)

        tree_df = pd.concat(l, axis = 0, sort = True)

        tree_df['tree'] = i

        tree_list.append(tree_df)

    trees_df = pd.concat(tree_list, axis = 0, sort = True)

    trees_df = reorder_tree_df(trees_df)
    
    if return_raw_lines:

        return j_copy, trees_df

    else:

        return trees_df



def recursive_pop_children(n, l, verbose = False):
    '''Function to recursively extract nodes from nested structure and append to list.
    
    Procedure is as follows;
    - if no children item in dict, append items (in pd.DataFrame) to list
    - or remove children item from dict, then append remaining items (in pd.DataFrame) to list
    - then call function on left and right children.

    '''

    if 'children' in n.keys():

        children = n.pop('children')

        if verbose:

            print(n)

        l.append(pd.DataFrame(n, index = [n['nodeid']]))

        recursive_pop_children(children[0], l, verbose)

        recursive_pop_children(children[1], l, verbose)

    else:

        if verbose:

            print(n)        

        l.append(pd.DataFrame(n, index = [n['nodeid']]))



def read_dump_text(file, return_raw_lines = True):
    '''Reads an xgboost model dump text file and parses it into a tabular structure.

    Text file to read must be the output from xgboost.Booster.dump_model with dump_format = 'text'. 
    Note this argument was only added in 0.81 and this was the default prior to this release.

    Args:
        file (str): xgboost model dump .txt file.
        return_raw_lines (bool): should the raw lines read from the text file be returned?

    Returns: 
        pd.DataFrame: df with columns; tree, nodeid, depth, yes, no, missing, split, split_condition, leaf.
        If the model dump file was output with  with_stats = True then gain and cover columns are also
        in the output.

    '''

    with open(file) as f:

        lines = f.readlines()
    
    tree_no = -1
    
    lines_list = []
    
    for i in range(len(lines)):
        
        # if line is a new tree
        if lines[i][:7] == 'booster':
            
            tree_no += 1
        
        # else if node row
        else:
        
            line_dict = {}
        
            # remove \n from end and any \t from start
            node_str = lines[i][:len(lines[i])-1].replace('\t', '')
            
            line_dict['tree'] = tree_no

            # note this will get tree depth for all nodes, which is not consistent with the json dump output
            # xgb json model dumps only contain depth for the non-terminal nodes
            line_dict['depth'] = lines[i].count('\t')
            
            # split by :
            node_str_split1 = node_str.split(':')
            
            # get the node number before the :
            line_dict['nodeid'] = int(node_str_split1[0])

            # else if leaf node
            if node_str_split1[1][:4]  == 'leaf':
                
                node_str_split2 = node_str_split1[1].split(',')
                
                line_dict['leaf'] = float(node_str_split2[0].split('=')[1])

                # if model is dumped with the arg with_stats = False then cover will not be included 
                # in the dump for terminal nodes
                try:

                    line_dict['cover'] = float(node_str_split2[1].split('=')[1])

                except:

                    pass

            # else non terminal node
            else:
                
                node_str_split2 = node_str_split1[1].split(' ')
                
                node_str_split3 = node_str_split2[0].replace('[', '').replace(']', '').split('<')
                
                # extract split variable name before the <
                line_dict['split'] = node_str_split3[0]

                # extract split point after the <
                line_dict['split_condition'] = float(node_str_split3[1])
  
                node_str_split4 = node_str_split2[1].split(',')
                
                # get the child nodes
                line_dict['yes'] = int(node_str_split4[0].split('=')[1])
                line_dict['no'] = int(node_str_split4[1].split('=')[1])
                line_dict['missing'] = int(node_str_split4[2].split('=')[1])

                # if model is dumped with the arg with_stats = False then gain and cover will not 
                # be included in the dump for non-terminal nodes
                try:

                    # get the child nodes
                    line_dict['gain'] = float(node_str_split4[3].split('=')[1])
                    line_dict['cover'] = float(node_str_split4[4].split('=')[1])

                except:

                    pass                

            lines_list = lines_list + [line_dict]
    
    lines_df = pd.DataFrame.from_dict(lines_list)

    lines_df = reorder_tree_df(lines_df)
    
    if return_raw_lines:

        return lines, lines_df

    else:

        return lines_df



def reorder_tree_df(df):
    '''Function to sort and reorder columns df of trees.'''

    if not isinstance(df, pd.DataFrame):

        raise TypeError('df should be a pd.DataFrame')

    if not df.shape[0] > 0:

        raise ValueError('df has no rows')

    if df.shape[1] == 11:
        
        col_order = [
            'tree', 
            'nodeid', 
            'depth',
            'yes', 
            'no', 
            'missing', 
            'split', 
            'split_condition', 
            'gain', 
            'cover',
            'leaf',             
        ]

    elif df.shape[1] == 9: 

        col_order = [
            'tree', 
            'nodeid', 
            'depth',
            'yes', 
            'no', 
            'missing', 
            'split', 
            'split_condition', 
            'leaf', 
        ]

    else:

        raise ValueError(
            'Unexpected number of columns in parsed model dump. Got ' +
            str(df.shape[1]) + 
            ' expected 11 or 9. Columns; ' +
            str(df.columns.values)
        )
    
    # reorder columns
    df = df.loc[:,col_order]
    
    df.sort_values(['tree', 'nodeid'], inplace = True)

    return df


