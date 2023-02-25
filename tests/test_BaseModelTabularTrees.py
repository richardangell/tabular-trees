import abc
import re
from dataclasses import dataclass

import pandas as pd
import pytest

from tabular_trees.trees import BaseModelTabularTrees


@pytest.fixture
def dummy_model_tree_data() -> pd.DataFrame:

    dummy_model_tree_data = pd.DataFrame(
        {"column1": [3, 2, 1], "column2": [4, 5, 6], "column3": ["a", "b", "c"]}
    )

    return dummy_model_tree_data


@pytest.fixture
def BaseModelTabularTreesImplementation():
    """Return an object of type DummyModelTabularTrees with dummy tree data."""

    @dataclass
    class DummyModelTabularTrees(BaseModelTabularTrees):
        """Dummy class mimicking a model specific class inheriting from
        BaseModelTabularTrees.

        This implements eevrything required to use the class.
        """

        trees: pd.DataFrame

        REQUIRED_COLUMNS = ["column1", "column2", "column3"]

        SORT_BY_COLUMNS = ["column2"]

        def __post_post_init__(self):
            """No processing after post_init."""

            pass

    return DummyModelTabularTrees


class TestBaseModelTabularTreesInit:
    """Tests for the BaseModelTabularTrees.__init__ method."""

    def test_abstract_base_class(self):
        """Test that BaseModelTabularTrees is an abstract base class."""

        assert (
            BaseModelTabularTrees.__mro__[1] is abc.ABC
        ), "BaseModelTabularTrees is not an abstract base class"

    def test_successful_initialisation(
        self, BaseModelTabularTreesImplementation, dummy_model_tree_data
    ):
        """Test successful initialisation of an implementation of
        BaseModelTabularTrees."""

        BaseModelTabularTreesImplementation(trees=dummy_model_tree_data)

    def test_trees_attribute_set(
        self, BaseModelTabularTreesImplementation, dummy_model_tree_data, mocker
    ):
        """Test the trees attribute is set to the value passed in init."""

        # mock the post init method so no other processing happens to trees arg
        mocker.patch.object(BaseModelTabularTrees, "__post_init__")

        dummy_model_tabular_trees = BaseModelTabularTreesImplementation(
            trees=dummy_model_tree_data
        )

        pd.testing.assert_frame_equal(
            dummy_model_tabular_trees.trees, dummy_model_tree_data
        )

    def test_exception_required_columns_not_defined(self):
        """Test an implementation of BaseModelTabularTrees cannot be
        initialised if it does not set the REQUIRED_COLUMNS attribute.
        """

        with pytest.raises(
            NotImplementedError, match="REQUIRED_COLUMNS attribute not defined"
        ):

            @dataclass
            class DummyModelTabularTrees(BaseModelTabularTrees):
                """Dummy class mimicking a model specific class inheriting from
                BaseModelTabularTrees.

                This impementation does have the REQUIRED_COLUMNS attribute.
                """

                trees: pd.DataFrame

                SORT_BY_COLUMNS = ["column2"]

    def test_exception_sort_by_columns_not_defined(self):
        """Test an implementation of BaseModelTabularTrees cannot be
        initialised if it does not set the SORT_BY_COLUMNS attribute.
        """

        with pytest.raises(
            NotImplementedError, match="SORT_BY_COLUMNS attribute not defined"
        ):

            @dataclass
            class DummyModelTabularTrees(BaseModelTabularTrees):
                """Dummy class mimicking a model specific class inheriting from
                BaseModelTabularTrees.

                This impementation does have the SORT_BY_COLUMNS attribute.
                """

                trees: pd.DataFrame

                REQUIRED_COLUMNS = ["column1", "column2", "column3"]


class TestBaseModelTabularTreesPostInit:
    """Tests for the BaseModelTabularTrees.__post_init__ method."""

    def test_attribute_error_no_trees_attribute(self):
        """Test an exception is raised if the implementation of
        BaseModelTabularTrees does not set the trees attribute.
        """

        @dataclass
        class DummyModelTabularTrees(BaseModelTabularTrees):
            """Dummy class mimicking a model specific class inheriting from
            BaseModelTabularTrees.

            This implementation does will not set a trees attribute.
            """

            REQUIRED_COLUMNS = ["column1", "column2", "column3"]

            SORT_BY_COLUMNS = ["column2"]

        with pytest.raises(AttributeError, match="trees attribute not set"):

            DummyModelTabularTrees()

    def test_trees_not_dataframe_exception(self, BaseModelTabularTreesImplementation):
        """Test a TypeError is raised if trees is not a pd.DataFrame."""

        with pytest.raises(
            TypeError,
            match="trees is not in expected types <class 'pandas.core.frame.DataFrame'>, got <class 'int'>",
        ):

            BaseModelTabularTreesImplementation(trees=12345)

    def test_required_columns_not_list_exception(self, dummy_model_tree_data):
        """Test a TypeError is raised if REQUIRED_COLUMNS is not a list."""

        @dataclass
        class DummyModelTabularTrees(BaseModelTabularTrees):
            """Dummy class mimicking a model specific class inheriting from
            BaseModelTabularTrees.

            This implementation sets REQUIRED_COLUMNS as a str instead of list.
            """

            trees: pd.DataFrame

            REQUIRED_COLUMNS = "a"

            SORT_BY_COLUMNS = ["column2"]

        with pytest.raises(
            TypeError,
            match="REQUIRED_COLUMNS is not in expected types <class 'list'>, got <class 'str'>",
        ):

            DummyModelTabularTrees(trees=dummy_model_tree_data)

    def test_sort_by_columns_not_list_exception(self, dummy_model_tree_data):
        """Test a TypeError is raised if SORT_BY_COLUMNS is not a list."""

        @dataclass
        class DummyModelTabularTrees(BaseModelTabularTrees):
            """Dummy class mimicking a model specific class inheriting from
            BaseModelTabularTrees.

            This implementation sets SORT_BY_COLUMNS as a str instead of list.
            """

            trees: pd.DataFrame

            REQUIRED_COLUMNS = ["column1", "column2", "column3"]

            SORT_BY_COLUMNS = "a"

        with pytest.raises(
            TypeError,
            match="SORT_BY_COLUMNS is not in expected types <class 'list'>, got <class 'str'>",
        ):

            DummyModelTabularTrees(trees=dummy_model_tree_data)

    @pytest.mark.parametrize(
        "drop_columns",
        [
            (["column1"]),
            (["column2"]),
            (["column3"]),
            (["column1", "column2"]),
            (["column3", "column2"]),
            (["column3", "column1", "column2"]),
        ],
    )
    def test_missing_columns_exception(
        self,
        drop_columns,
        BaseModelTabularTreesImplementation,
        dummy_model_tree_data,
    ):
        """Test an exception is raised if columns from REQUIRED_COLUMNS are
        missing in trees."""

        dropped_columns = dummy_model_tree_data.drop(columns=drop_columns)

        with pytest.raises(
            ValueError,
            match=re.escape(f"expected columns not in df; {sorted(drop_columns)}"),
        ):

            BaseModelTabularTreesImplementation(dropped_columns)

    def test_sort_by_not_subset_required_exception(self, dummy_model_tree_data):
        """Test a ValueError is raised if SORT_BY_COLUMNS is not a subset of
        REQUIRED_COLUMNS."""

        @dataclass
        class DummyModelTabularTrees(BaseModelTabularTrees):
            """Dummy class mimicking a model specific class inheriting from
            BaseModelTabularTrees.

            This implementation has SORT_BY_COLUMNS set as NOT a subset of
            REQUIRED_COLUMNS.
            """

            trees: pd.DataFrame

            REQUIRED_COLUMNS = ["column1", "column2", "column3"]

            SORT_BY_COLUMNS = ["a"]

        with pytest.raises(
            ValueError,
            match=re.escape(
                "condition: [SORT_BY_COLUMNS is a subset of REQUIRED_COLUMNS] not met"
            ),
        ):

            DummyModelTabularTrees(trees=dummy_model_tree_data)

    def test_trees_attribute_sorted(
        self, BaseModelTabularTreesImplementation, dummy_model_tree_data
    ):
        """Test the rows of trees are sorted by SORT_BY_COLUMNS."""

        out_of_order_rows = dummy_model_tree_data.sort_values("column1")

        tabular_trees = BaseModelTabularTreesImplementation(out_of_order_rows)

        pd.testing.assert_frame_equal(
            tabular_trees.trees,
            out_of_order_rows.sort_values(
                BaseModelTabularTreesImplementation.SORT_BY_COLUMNS
            ),
        )

    def test_trees_attribute_columns_ordered(
        self, BaseModelTabularTreesImplementation, dummy_model_tree_data
    ):
        """Test the columns of trees are sorted into REQUIRED_COLUMNS order."""

        out_of_order_columns = dummy_model_tree_data[
            ["column2", "column3", "column1"]
        ].copy()

        tabular_trees = BaseModelTabularTreesImplementation(out_of_order_columns)

        pd.testing.assert_frame_equal(
            tabular_trees.trees,
            out_of_order_columns[BaseModelTabularTreesImplementation.REQUIRED_COLUMNS],
        )

    def test_trees_attribute_index_reset(
        self, BaseModelTabularTreesImplementation, dummy_model_tree_data
    ):
        """Test that the index is reset on the trees attribute."""

        new_index = dummy_model_tree_data.copy()
        new_index.index = [9, -1, 3]

        tabular_trees = BaseModelTabularTreesImplementation(new_index)

        pd.testing.assert_frame_equal(
            tabular_trees.trees, new_index.reset_index(drop=True)
        )

    def test_post_post_init_called(
        self, mocker, BaseModelTabularTreesImplementation, dummy_model_tree_data
    ):
        """Test that the __post_post_init__ method is called once."""

        mocked = mocker.patch.object(
            BaseModelTabularTreesImplementation, "__post_post_init__"
        )

        BaseModelTabularTreesImplementation(dummy_model_tree_data)

        assert mocked.call_count == 1, "__post_post_init__ not called once"

        assert (
            mocked.call_args_list[0][0] == ()
        ), "positional args not as expected in __post_post_init__ call"

        assert (
            mocked.call_args_list[0][1] == {}
        ), "keyword args not as expected in __post_post_init__ call"
