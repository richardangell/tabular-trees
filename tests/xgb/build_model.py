import xgboost as xgb
from sklearn.datasets import load_boston



def build_depth_3_model(n_trees = 10):
    """Build xgboost model on boston dataset with 10 trees and depth 3."""

    boston = load_boston()

    xgb_data = xgb.DMatrix(
        data = boston['data'], 
        label = boston['target'], 
        feature_names = boston['feature_names']
    )

    model = xgb.train(
        params = {
            'silent': 1,
            'max_depth': 3
        }, 
        dtrain = xgb_data, 
        num_boost_round = n_trees
    )

    return model
