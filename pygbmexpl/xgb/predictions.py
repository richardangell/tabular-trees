import xgboost as xgb
import pandas as pd
import datetime
import os

import parser as p



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

        file = str(datetime.datetime.now()) + '-temp-xgb-dump.txt'

        delete_file = True

    model.dump_model(file, with_stats = True)

    # parse the model dump text file
    lines, trees_df = p.read_dump(file)

    # if no filename was specified remove the temp model dump
    if delete_file:

        os.remove(file)

    trees_preds_df = derive_predictions(trees_df)

    return(trees_preds_df)







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
        - df_G.loc[df_G.node_type == 'internal', 'G'] / df_G.loc[df_G.node_type == 'internal', 'H']

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












