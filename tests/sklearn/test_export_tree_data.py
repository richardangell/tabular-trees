import pytest

from tabular_trees.sklearn.trees import (
    ScikitLearnHistTabularTrees,
    ScikitLearnTabularTrees,
)
from tabular_trees.trees import export_tree_data


@pytest.mark.parametrize(
    "model_fixture_name,expected_type",
    [
        ("sklearn_diabetes_hist_gbr", ScikitLearnHistTabularTrees),
        ("sklearn_breast_cancer_hist_gbc", ScikitLearnHistTabularTrees),
        ("sklearn_diabetes_gbr", ScikitLearnTabularTrees),
        ("sklearn_breast_cancer_gbc", ScikitLearnTabularTrees),
    ],
)
def test_model_specific_function_dispatch(request, model_fixture_name, expected_type):
    """Test export_tree_data calls the correct model-speific export function."""

    model = request.getfixturevalue(model_fixture_name)

    tree_data = export_tree_data(model)

    assert (
        type(tree_data) is expected_type
    ), f"incorrect type returned when export_tree_data called with {type(model)}"
