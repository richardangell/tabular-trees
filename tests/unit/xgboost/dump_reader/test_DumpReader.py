import re

import pytest

from tabular_trees.xgboost.dump_reader import (
    DumpReader,
)


class DummyDumpReader(DumpReader):
    """Dummy class inheriting from DumpReader, to test DumpReader functionality."""

    dump_type = "dummy"

    def read_dump(self, file: str) -> None:
        """Simply call the DumpReader.read_dump method."""
        return super().read_dump(file)


class TestDumpReaderCheckFileExists:
    """Tests for the DumpReader.check_file_exists abstract method."""

    def test_file_not_str_exception(self):
        """Test that a TypeError is raised if file is not a str."""
        dump_reader = DummyDumpReader()

        with pytest.raises(
            TypeError,
            match="file is not in expected types <class 'str'>, got <class 'list'>",
        ):
            dump_reader.check_file_exists([1, 2, 3])

    def test_file_does_not_exist_exception(self):
        """Test that a ValueError is raised if file does not exist."""
        dump_reader = DummyDumpReader()

        with pytest.raises(
            ValueError,
            match=re.escape("condition: [does_not_exist.txt exists] not met"),
        ):
            dump_reader.check_file_exists("does_not_exist.txt")
