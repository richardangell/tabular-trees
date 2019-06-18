import pandas as pd



def read_dump(file):
    '''Reads an xgboost model dump .txt file and parses it into a tabular structure.

    Args:
        file (str): xgboost model dump .txt file.

    Returns: 
        pd.DataFrame: df with columns tree, node, yes, no, missing, split_var, split_point, quality, cover.

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
            
            # split by :
            node_str_split1 = node_str.split(':')
            
            # get the node number before the :
            line_dict['node'] = int(node_str_split1[0])

            # else if leaf node
            if node_str_split1[1][:4]  == 'leaf':
                
                node_str_split2 = node_str_split1[1].split(',')
                
                line_dict['quality'] = float(node_str_split2[0].split('=')[1])

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
                line_dict['split_var'] = node_str_split3[0]

                # extract split point after the <
                line_dict['split_point'] = float(node_str_split3[1])
  
                node_str_split4 = node_str_split2[1].split(',')
                
                # get the child nodes
                line_dict['yes'] = int(node_str_split4[0].split('=')[1])
                line_dict['no'] = int(node_str_split4[1].split('=')[1])
                line_dict['missing'] = int(node_str_split4[2].split('=')[1])

                # if model is dumped with the arg with_stats = False then gain and cover will not 
                # be included in the dump for non-terminal nodes
                try:

                    # get the child nodes
                    # note quality = gain
                    line_dict['quality'] = float(node_str_split4[3].split('=')[1])
                    line_dict['cover'] = float(node_str_split4[4].split('=')[1])

                except:

                    pass                

            lines_list = lines_list + [line_dict]
    
    lines_df = pd.DataFrame.from_dict(lines_list)

    if lines_df.shape[1] == 9:
        
        col_order = [
            'tree', 
            'node', 
            'yes', 
            'no', 
            'missing', 
            'split_var', 
            'split_point', 
            'quality', 
            'cover',
        ]

    elif lines_df.shape[1] == 8: 

        col_order = [
            'tree', 
            'node', 
            'yes', 
            'no', 
            'missing', 
            'split_var', 
            'split_point', 
            'quality', 
        ]

    else:

        raise ValueError(
            'Unexpected number of columns in parsed model dump. Got ' +
            str(lines_df.shape[1]) + 
            ' expected 8 or 9. Columns; ' +
            str(lines_df.columns.values)
        )
    
    # reorder columns
    lines_df = lines_df.loc[:,col_order]
    
    lines_df.sort_values(['tree', 'node'], inplace = True)
    
    return lines, lines_df


