import xgboost as xgb
import pandas as pd
import datetime
import os



def extract_model_predictions(model, file = None):
    """Extract predictions for all nodes in an xgboost model.

    Args:
        model (xgb.core.booster): an xgboost model.
        file (str): xgboost model dump .txt file. Defaults to None.
            Pass a value to keep the model dump .txt file, otherwise it
            is deleted.

    Returns: 
        pd.DataFrame: df with columns tree, node, yes, no, missing, split_var, split_point, quality, cover
            weight, G, H, node_type. Note, weight is the prediction for this node.

    """

    if file == None:

        save_file = str(datetime.datetime.now()) + '-temp-xgb-dump.txt'

        model.dump_model(save_file, with_stats = True)

    else:

        model.dump_model(file, with_stats = True)

    # parse the model dump text file
    lines, trees_df = read_dump(file)

    # if no filename was specified remove the temp model dump
    if file == None:

        os.remove(save_file)

    trees_preds_df = derive_predictions(trees_df)

    return(trees_preds_df)



def read_dump(file):
    """Reads an xgboost model dump .txt file and parses it into a tabular structure.

    Args:
        file (str): xgboost model dump .txt file.

    Returns: 
        pd.DataFrame: df with columns tree, node, yes, no, missing, split_var, split_point, quality, cover.

    """
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
                line_dict['cover'] = float(node_str_split2[1].split('=')[1])

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
                
                # get the child nodes
                # note quality = gain
                line_dict['quality'] = float(node_str_split4[3].split('=')[1])
                line_dict['cover'] = float(node_str_split4[4].split('=')[1])

            lines_list = lines_list + [line_dict]
    
    lines_df = pd.DataFrame.from_dict(lines_list)
    
    col_order = ['tree', 'node', 'yes', 'no', 'missing', 'split_var', 'split_point','quality', 'cover']
    
    # reorder columns
    lines_df = lines_df.loc[:,col_order]
    
    lines_df.sort_values(['tree', 'node'], inplace = True)
    
    return(lines, lines_df)




def derive_predictions(df):
    
    # column to hold predictions
    df['weight'] = 0

    # identify leaf and internal nodes
    df['node_type'] = 'internal'
    df.loc[df.split_point.isnull(), 'node_type'] = 'leaf'

    df.loc[df.node_type == 'leaf', 'weight'] = df.loc[df.node_type == 'leaf', 'quality']

    df['H'] = df['cover']
    df['G'] = 0

    df.loc[df.node_type == 'leaf', 'G'] = \
        - df.loc[df.node_type == 'leaf', 'weight'] * df.loc[df.node_type == 'leaf', 'H']

    df.reset_index(inplace = True)

    n_trees = df.tree.max()

    # propagate G up from the leaf nodes to internal nodes, for each tree
    df_G_list = [derive_internal_node_G(df.loc[df.tree == n]) for n in range(n_trees + 1)]

    # append all updated trees
    df_G = pd.concat(df_G_list, axis = 0)

    assert (df_G.G == 0).sum() == 0, 'G not propagated successfully'

    # update weight values for internal nodes
    df_G.loc[df_G.node_type == 'internal', 'weight'] = \
        - df.loc[df.node_type == 'internal', 'weight'] * df.loc[df.node_type == 'internal', 'H']

    return(df_G)




def derive_internal_node_G(tree_df):
    """Function to derive predictons for internal nodes in a single tree.
    
    This involves starting at each leaf node in the tree and propagating the
    G value back up through the tree, adding this leaf node G to each node
    that is travelled to.

    Args:
        tree_df (pd.DataFrame): rows from corresponding to a single tree (from derive_predictions)

    Returns: 
        pd.DataFrame: updated tree_df with G propagated up the tree s.t. each internal node's G value
            is the sum of G for it's child nodes.

    """

    tree_df = tree_df.copy()

    leaf_df = tree_df.loc[tree_df.node_type == 'leaf']

    # loop through each leaf node
    for i in leaf_df.index:

        #print(i, 'leaf------------------')

        leaf_row = leaf_df.loc[[i]]
        current_node = leaf_row['node'].item()

        leaf_G = leaf_row['G'].item()

        #print('current_node', current_node)
        #print(tree_df)
        #print('----')

        # if the leaf node is not also the first node in the tree
        if current_node > 0:

            # traverse the tree bottom from bottom to top and propagate the G value upwards
            while True:
                
                # find parent node row
                parent = (tree_df.yes == current_node) | (tree_df.no == current_node)

                # get parent node G
                tree_df.loc[parent, 'G'] = tree_df.loc[parent, 'G'] + leaf_G

                # update the current node to be the parent node
                leaf_row = tree_df.loc[parent]
                current_node = leaf_row['node'].item()

                #print('current_node', current_node)
                #print(tree_df)
                #print('----')

                # if we have made it back to the top node in the tree then stop
                if current_node == 0:

                    break

    return(tree_df)












