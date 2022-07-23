import pytest

from tabular_trees.trees import export_tree_data
from tabular_trees.sklearn.trees import ScikitLearnHistTabularTrees


def test_successful_call(sklearn_diabetes_hist_gbr):
    """Test a successful call to export tree data."""

    export_tree_data(sklearn_diabetes_hist_gbr)


def test_non_supported_type_exception():
    """Test an NotImplementedError is raised if a non-supported type is passed
    in the model argument."""

    with pytest.raises(
        NotImplementedError, match="model type not supported; <class 'int'>"
    ):

        export_tree_data(1)


@pytest.mark.parametrize(
    "model_fixture_name,expected_type",
    [
        ("sklearn_diabetes_hist_gbr", ScikitLearnHistTabularTrees),
        ("sklearn_breast_cancer_model", ScikitLearnHistTabularTrees),
    ],
)
def test_model_specific_function_dispatch(request, model_fixture_name, expected_type):
    """Test export_tree_data calls the correct model-speific export function."""

    model = request.getfixturevalue(model_fixture_name)

    tree_data = export_tree_data(model)

    assert (
        type(tree_data) is expected_type
    ), f"incorrect type returned when export_tree_data called with {type(model)}"
