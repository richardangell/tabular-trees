import pytest

from tabular_trees.trees import export_tree_data


def test_non_supported_type_exception():
    """Test error is raised if a non-supported type is passed in the model."""
    with pytest.raises(
        NotImplementedError, match="model type not supported; <class 'int'>"
    ):
        export_tree_data(1)
