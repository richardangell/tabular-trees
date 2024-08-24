import re

import pandas as pd
import pytest

from tabular_trees.trees import BaseModelTabularTrees, TabularTrees


@pytest.fixture(scope="session")
def dummy_trees_data() -> pd.DataFrame:
    """Return DataFrame with columns required by TabularTrees but no rows."""
    return pd.DataFrame(columns=TabularTrees.REQUIRED_COLUMNS)


class TestInitialisation:
    """Tests for the TabularTrees.__init__ method."""

    @pytest.mark.xfail()
    def test_inheritance(self):
        """Test that TabularTrees inherits from BaseModelTabularTrees."""
        assert (
            TabularTrees.__mro__[1] is BaseModelTabularTrees
        ), "TabularTrees does not inherit from BaseModelTabularTrees"

    def test_attributes_set(self, dummy_trees_data):
        """Test trees, get_root_node_given_tree attributes are set to values passed."""

        def dummy_get_root_node_function():
            """Return a constant of 0.

            Dummy function to pass into get_root_node_given_tree argument.

            """
            return 0

        tabular_trees = TabularTrees(
            trees=dummy_trees_data,
            get_root_node_given_tree=dummy_get_root_node_function,
        )

        assert (
            tabular_trees.get_root_node_given_tree == dummy_get_root_node_function
        ), "get_root_node_given_tree not set to passed value"

        pd.testing.assert_frame_equal(
            tabular_trees.trees, dummy_trees_data.reset_index(drop=True)
        )

    def test_exception_raised_non_callable_passed(self, dummy_trees_data):
        """Test an exception is raised if get_root_node_given_tree is not callable."""
        with pytest.raises(
            ValueError,
            match=re.escape(
                "condition: [get_root_node_given_tree is callable] not met"
            ),
        ):
            TabularTrees(trees=dummy_trees_data, get_root_node_given_tree=123)

    @pytest.mark.xfail()
    def test_tree_data_copied(self, dummy_trees_data):
        """Test that trees attribute is a copy of the data passed in init."""

        def dummy_get_root_node_function():
            """Return a constant of 0.

            Dummy function to pass into get_root_node_given_tree argument.

            """
            return 0

        tabular_trees = TabularTrees(
            trees=dummy_trees_data,
            get_root_node_given_tree=dummy_get_root_node_function,
        )

        assert id(tabular_trees.trees) != id(
            dummy_trees_data
        ), "trees attribute has the same id as the passed DataFrame"
