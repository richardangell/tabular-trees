import re

import pandas as pd
import pytest
import xgboost as xgb

from tabular_trees.xgboost import (
    DumpReader,
    JsonDumpReader,
    ParsedXGBoostTabularTrees,
    TextDumpReader,
    XGBoostParser,
)


class DummyDumpReader(DumpReader):
    """Dummy class inheriting from DumpReader, to test DumpReader
    functionality."""

    dump_type = "dummy"

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


class TestDumpReaderImplementations:
    """Tests for the DumpReader subclasses."""

    @pytest.mark.parametrize(
        "reader_class,expected_value",
        [(TextDumpReader, "text"), (JsonDumpReader, "json")],
    )
    def test_dump_reader_dump_type(self, reader_class, expected_value):
        """Test DumpReader classes have correct dump_type attribute value."""

        assert (
            reader_class.dump_type == expected_value
        ), f"dump_type attribute of {reader_class} class not expected"

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

        text_parser = TextDumpReader()
        text_output = text_parser.read_dump(text_dump)

        json_parser = JsonDumpReader()
        json_output = json_parser.read_dump(json_dump)

        assert sorted(text_output.columns.values) == sorted(
            json_output.columns.values
        ), "column names are not the same between text and json outputs"

        # reorder columns to be in the same order
        text_output = text_output[json_output.columns.values]

        pd.testing.assert_frame_equal(text_output, json_output)


class TestXGBoostParserInit:
    """Tests for the XGBoostParser.__init__ method."""

    def test_model_not_booster_exception(self):
        """Test that a TypeError is raised if model is not a Booster."""

        with pytest.raises(
            TypeError,
            match=(
                "model is not in expected types <class 'xgboost.core.Booster'>, "
                "got <class 'tuple'>"
            ),
        ):
            XGBoostParser((1, 2, 3))

    @pytest.mark.parametrize("dump_reader", [(JsonDumpReader), (TextDumpReader)])
    def test_attributes_set(self, xgb_diabetes_model, dump_reader):
        """Test model, dump_type and reader attributes are correctly set."""

        xgboost_parser = XGBoostParser(xgb_diabetes_model, dump_reader())

        assert (
            xgboost_parser.model == xgb_diabetes_model
        ), "model attribute not set correctly"

        assert (
            type(xgboost_parser.reader) is dump_reader
        ), "incorrect type for reader attribute"

    def test_depreceation_warning(self, xgb_diabetes_model):
        """Test a FutureWarning is raised when initialising an XGBoostParser object."""

        expected_warning = (
            "XGBoostDumpParser class is depreceated, "
            "Booster.trees_to_dataframe is available instead"
        )

        with pytest.warns(FutureWarning, match=expected_warning):
            XGBoostParser(xgb_diabetes_model)


class TestXGBoostParserParseModel:
    """Tests for the XGBoostParser.parse_model method."""

    def test_model_dumped_then_read(self, mocker, xgb_diabetes_model):
        """Test the booster calls dump_model and the output is then read
        with read_dump."""

        xgboost_parser = XGBoostParser(xgb_diabetes_model, JsonDumpReader())

        dump_model_spy = mocker.spy(xgb.core.Booster, "dump_model")
        read_dump_spy = mocker.spy(JsonDumpReader, "read_dump")

        xgboost_parser.parse_model()

        assert dump_model_spy.call_count == 1, "xgb.Booster.dump_model not called once"

        assert read_dump_spy.call_count == 1, "JsonDumpReader.read_dump not called once"

        dump_model_call_args = dump_model_spy.call_args_list[0]

        assert dump_model_call_args[1][
            "with_stats"
        ], "xgboost model not dumped with with_stats = True"

        assert (
            dump_model_call_args[1]["dump_format"] == JsonDumpReader.dump_type
        ), "xgboost model not dumped with dump_format from JsonDumpReader"

        assert (
            len(dump_model_call_args[0]) == 2
        ), "incorrect number of positional arguments in dump_model call"

        dumped_file = dump_model_call_args[0][1]

        read_dump_call_args = read_dump_spy.call_args_list[0]

        assert read_dump_call_args[0] == (
            xgboost_parser.reader,
            dumped_file,
        ), "read_dump not called on the file output from dump_model"

        assert (
            read_dump_call_args[1] == {}
        ), "keyword args not corrext in read_dump call"

    def test_read_dump_output_returned(self, mocker, xgb_diabetes_model):
        """Test that the output from parse_model is ParsedXGBoostTabularTrees
        with the output of read_dump."""

        xgboost_parser = XGBoostParser(xgb_diabetes_model, JsonDumpReader())

        read_dump_return_value = 12345

        read_dump_mock = mocker.patch.object(
            JsonDumpReader, "read_dump", return_value=read_dump_return_value
        )
        parsed_xgb_tabular_trees_mock = mocker.patch.object(
            ParsedXGBoostTabularTrees, "__init__", return_value=None
        )

        xgboost_parser.parse_model()

        assert (
            read_dump_mock.call_count == 1
        ), "JsonDumpReader.read_dump not called once"

        assert (
            parsed_xgb_tabular_trees_mock.call_count == 1
        ), "ParsedXGBoostTabularTrees.__init__ not called once"

        init_call_args = parsed_xgb_tabular_trees_mock.call_args_list[0]

        assert init_call_args[0] == (
            read_dump_return_value,
        ), "ParsedXGBoostTabularTrees.__init__ not called with output from read_dump"

        assert (
            init_call_args[1] == {}
        ), "keyword args in ParsedXGBoostTabularTrees.__init__ call not as expected"
