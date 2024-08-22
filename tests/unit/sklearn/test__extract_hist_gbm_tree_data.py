import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from tabular_trees.sklearn import scikit_learn_hist_tabular_trees


def test_output_values(handcrafted_data):
    """Test that the values output are expected for a simple, known tree."""
    model = HistGradientBoostingRegressor(
        max_iter=1, max_depth=2, learning_rate=1, min_samples_leaf=1
    )

    model.fit(handcrafted_data[["a", "b"]], handcrafted_data["response"])

    extracted_tree_data = scikit_learn_hist_tabular_trees._extract_hist_gbm_tree_data(
        model
    )

    expected_tree_data = pd.DataFrame(
        {
            "node": [0, 1, 2, 3, 4, 5, 6],
            "value": [175.0, 225.0, 250.0, 200.0, 125.0, 150.0, 100.0],
            "count": [8, 4, 2, 2, 4, 2, 2],
            "feature_idx": [0, 1, 0, 0, 1, 0, 0],
            "num_threshold": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "missing_go_to_left": [0, 0, 0, 0, 0, 0, 0],
            "left": [1, 2, 0, 0, 5, 0, 0],
            "right": [4, 3, 0, 0, 6, 0, 0],
            "gain": [20000.0, 2500.0, -1.0, -1.0, 2500.0, -1.0, -1.0],
            "depth": [0, 1, 2, 2, 1, 2, 2],
            "is_leaf": [0, 0, 1, 1, 0, 1, 1],
            "bin_threshold": [0, 0, 0, 0, 0, 0, 0],
            "is_categorical": [0, 0, 0, 0, 0, 0, 0],
            "bitset_idx": [0, 0, 0, 0, 0, 0, 0],
            "tree": [0, 0, 0, 0, 0, 0, 0],
        }
    )

    pd.testing.assert_frame_equal(
        extracted_tree_data, expected_tree_data, check_dtype=False
    )
