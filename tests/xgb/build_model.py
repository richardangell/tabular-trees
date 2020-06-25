import xgboost as xgb
import pandas as pd
import numpy as np
from sklearn.datasets import load_boston



def build_depth_3_model(n_trees = 10, return_data = False, drop_columns = None):
    """Build xgboost model on boston dataset with 10 trees and depth 3."""

    boston = load_boston()

    boston_df = pd.DataFrame(boston['data'], columns = boston['feature_names'])

    if not drop_columns is None:

        boston_df.drop(columns = drop_columns, inplace = True)

    xgb_data = xgb.DMatrix(
        boston_df, 
        label = boston['target']
    )

    xgb_data.set_base_margin([0] * boston_df.shape[0])

    model = xgb.train(
        params = {
            'silent': 1,
            'max_depth': 3
        }, 
        dtrain = xgb_data, 
        num_boost_round = n_trees
    )

    if return_data:

        return model, xgb_data, boston_df

    else:

        return model



def build_example_shap_model():
    """Function to build simple model on 10 row dataset where expected shapley values are known."""

    np.random.seed(100)
    X_train = pd.DataFrame(
        {
            'x':[206]*5 + [194] + [6]*4,
            'y': list(np.random.randint(100, 400, 6)) + [299, 299, 301, 301],
            'z': list(np.random.randint(100, 400, 10))
        }
    )

    Y_train = pd.Series([10]*5 + [20] + [50]*2 + [30]*2)
    Y_train.name = 't'
    
    xgb_data = xgb.DMatrix(
        data = X_train, 
        label = Y_train
    )
    xgb_data.set_base_margin([0] * X_train.shape[0])

    model = xgb.train(
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 2, 
            'subsample': 1,
            'colsample_bytree': 1,
            'eta': 1, 
            'lambda': 0,
            'gamma': 0,
            'alpha': 0
        }, 
        dtrain = xgb_data, 
        num_boost_round = 1
    )

    return model
