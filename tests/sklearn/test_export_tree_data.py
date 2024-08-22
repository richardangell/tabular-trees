import pytest

from tabular_trees.sklearn.scikit_learn_hist_tabular_trees import (
    ScikitLearnHistTabularTrees,
)
from tabular_trees.sklearn.scikit_learn_tabular_trees import ScikitLearnTabularTrees
from tabular_trees.trees import export_tree_data


@pytest.mark.parametrize(
    "model_fixture_name,expected_type",
    [
        ("sklearn_diabetes_hist_gbm_regressor", ScikitLearnHistTabularTrees),
        ("sklearn_breast_cancer_hist_gbm_classifier", ScikitLearnHistTabularTrees),
        ("sklearn_diabetes_gbm_regressor", ScikitLearnTabularTrees),
        ("sklearn_breast_cancer_gbm_classifier", ScikitLearnTabularTrees),
    ],
)
def test_model_specific_function_dispatch(request, model_fixture_name, expected_type):
    """Test export_tree_data returns the correct TabularTrees subclass."""
    model = request.getfixturevalue(model_fixture_name)

    tree_data = export_tree_data(model)

    assert (
        type(tree_data) is expected_type
    ), f"incorrect type returned when export_tree_data called with {type(model)}"
