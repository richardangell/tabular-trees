import pandas as pd
import numpy as np


def decompose_prediction(trees_df, row):

    n_trees = trees_df.tree.max()

    # run terminal_node_path for each tree
    terminal_node_paths = [terminal_node_path(tree_df = trees_df.loc[trees_df.tree == n],
                                              row = row) for n in range(n_trees + 1)]

    # append the paths for each tree
    terminal_node_paths = pd.concat(terminal_node_paths, axis = 0)

    return(terminal_node_paths)



def terminal_node_path(tree_df, row):
    """Traverse tree according to the values in the given row of data.

    Args:
        tree_df (pd.DataFrame): df subset of output from pygbm.expl.xgb.extract_model_predictions 
            for single tree. 
        row (pd.DataFrame): single row df to explain prediction.

    Returns: 
        pd.DataFrame: df where each successive row shows the path of row through the tree.

    """

    # get column headers no rows
    path = tree_df.loc[tree_df.node == -1]

    # get the first node in the tree
    current_node = tree_df.loc[tree_df.node == 0].copy()

    # for internal nodes record the value of the variable that will be used to split
    if current_node['node_type'].item() != 'leaf':

        current_node['value'] = row[current_node['split_var']].values[0][0]

    else:

        current_node['value'] = np.NaN

    path = path.append(current_node)

    # as long as we are not at a leaf node already
    if current_node['node_type'].item() != 'leaf':
        
        # determine if the value of the split variable sends the row left (yes) or right (no)
        if row[current_node['split_var']].values[0] < current_node['split_point'].values[0]:

            next_node = current_node['yes'].item()

        else:

            next_node = current_node['no'].item()

        # (loop) traverse the tree until a leaf node is reached 
        while True:

            current_node = tree_df.loc[tree_df.node == next_node].copy()

            # for internal nodes record the value of the variable that will be used to split
            if current_node['node_type'].item() != 'leaf':

                current_node['value'] = row[current_node['split_var']].values[0][0]
        
            path = path.append(current_node)

            if current_node['node_type'].item() != 'leaf':
                
                # determine if the value of the split variable sends the row left (yes) or right (no)
                if row[current_node['split_var']].values[0] < current_node['split_point'].values[0]:

                    next_node = current_node['yes'].item()

                else:

                    next_node = current_node['no'].item()

            else:

                break

    # shift the split_vars down by 1 to get the variable which is contributing to the change in prediction
    path['contributing_var'] = path['split_var'].shift(1)

    # get change in predicted value due to split i.e. contribution for that variable
    path['contribution'] = path['weight'] - path['weight'].shift(1).fillna(0)

    path.loc[path.contributing_var.isnull(), 'contributing_var'] = 'base'

    cols_order = ['index',
                  'tree',
                  'node',
                  'yes',
                  'no',
                  'missing',
                  'split_var',
                  'split_point',
                  'quality',
                  'cover',
                  'weight',
                  'node_type',
                  'H',
                  'G',
                  'value',
                  'contributing_var',
                  'contribution']
    
    return(path[cols_order])




