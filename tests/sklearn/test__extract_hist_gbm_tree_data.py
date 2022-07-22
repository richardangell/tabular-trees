import pandas as pd
import numpy as np
from collections import namedtuple

from tabular_trees.sklearn import trees


def test_successful_call(sklearn_diabetes_model):
    """Test a successful call to _extract_hist_gbm_tree_data."""

    trees._extract_hist_gbm_tree_data(sklearn_diabetes_model)


def test_output():
    """Test the output from _extract_hist_gbm_tree_data is correct."""

    DummyNodes = namedtuple("DummyNodes", ["nodes"])

    class DummyModel:
        """Dummy class implementing the required attributes that will be
        accessed by  _extract_hist_gbm_tree_data."""

        def __init__(self, n_iter_, predictor_data):

            if n_iter_ != len(predictor_data):
                raise ValueError(
                    "number of items in predictor_data not equal to n_iter_"
                )

            self.n_iter_ = n_iter_

            self._predictors = [[DummyNodes(nodes=data)] for data in predictor_data]

    predictor_1 = np.rec.fromarrays(
        (np.array([1, 2, 3]), np.array([4, 5, 6])), names=("col", "col2")
    )

    predictor_2 = np.rec.fromarrays(
        (np.array([8, 3, 4]), np.array([2, 3, 9])), names=("col", "col2")
    )

    # set the _predictors and n_iter_ attributes to known values
    dummy_model = DummyModel(n_iter_=2, predictor_data=[predictor_1, predictor_2])

    results = trees._extract_hist_gbm_tree_data(dummy_model)

    expected_results = pd.DataFrame(
        {
            "node": [0, 1, 2, 0, 1, 2],
            "col": [1, 2, 3, 8, 3, 4],
            "col2": [4, 5, 6, 2, 3, 9],
            "tree": [0, 0, 0, 1, 1, 1],
        },
        index=[0, 1, 2, 0, 1, 2],
    )

    pd.testing.assert_frame_equal(results, expected_results)
