import pandas as pd
import tabular_trees
import re

import pytest


class DummyDumpReader(tabular_trees.xgboost.parser.DumpReader):
    """Dummy class inheriting from DumpReader, to test DumpReader
    functionality."""

    def read_dump(self, file: str) -> None:
        """Method that simply calls the DumpReader.read_dump method."""

        return super().read_dump(file)


class TestDumpReaderReadDump:
    """Tests for the DumpReader.read_dump abstract method."""

    def test_file_not_str_exception(self):
        """Test that a TypeError is raised if file is not a str."""

        dump_reader = DummyDumpReader()

        with pytest.raises(
            TypeError,
            match="file is not in expected types <class 'str'>, got <class 'list'>",
        ):

            dump_reader.read_dump([1, 2, 3])

    def test_file_does_not_exist_exception(self):
        """Test that a ValueError is raised if file does not exist."""

        dump_reader = DummyDumpReader()

        with pytest.raises(
            ValueError,
            match=re.escape("condition: [does_not_exist.txt exists] not met"),
        ):

            dump_reader.read_dump("does_not_exist.txt")


class TestDumpReader_Implementations:
    """Tests for the DumpReader subclasses."""

    @pytest.mark.parametrize("with_stats", [(False), (True)])
    def test_text_json_parsing_equal(self, with_stats, tmp_path, xgb_diabetes_model):
        """Test that parsing an xgboost model dumped to text file and json
        file gives the same output."""

        text_dump = str(tmp_path.joinpath("dump_raw.txt"))
        json_dump = str(tmp_path.joinpath("dump_raw.json"))

        xgb_diabetes_model.dump_model(
            text_dump, with_stats=with_stats, dump_format="text"
        )
        xgb_diabetes_model.dump_model(
            json_dump, with_stats=with_stats, dump_format="json"
        )

        text_parser = tabular_trees.xgboost.parser.TextDumpReader()
        text_output = text_parser.read_dump(text_dump)

        json_parser = tabular_trees.xgboost.parser.JsonDumpReader()
        json_output = json_parser.read_dump(json_dump)

        assert sorted(text_output.columns.values) == sorted(
            json_output.columns.values
        ), "column names are not the same between text and json outputs"

        # reorder columns to be in the same order
        text_output = text_output[json_output.columns.values]

        pd.testing.assert_frame_equal(text_output, json_output)
