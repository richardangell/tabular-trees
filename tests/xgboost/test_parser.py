import re

import pandas as pd
import pytest
import xgboost as xgb

import tabular_trees.xgboost.parser as parser


class DummyDumpReader(parser.DumpReader):
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


class TestDumpReader_Implementations:
    """Tests for the DumpReader subclasses."""

    def test_text_dump_reader_dump_type(self):
        """Test TextDumpReader object has correct dump_type attribute value."""

        text_dump_reader = parser.TextDumpReader()

        assert (
            text_dump_reader.dump_type == "text"
        ), "dump_type attribute incorrect on TextDumpReader"

    def test_json_dump_reader_dump_type(self):
        """Test JsonDumpReader object has correct dump_type attribute value."""

        text_dump_reader = parser.JsonDumpReader()

        assert (
            text_dump_reader.dump_type == "json"
        ), "dump_type attribute incorrect on JsonDumpReader"

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

        text_parser = parser.TextDumpReader()
        text_output = text_parser.read_dump(text_dump)

        json_parser = parser.JsonDumpReader()
        json_output = json_parser.read_dump(json_dump)

        assert sorted(text_output.columns.values) == sorted(
            json_output.columns.values
        ), "column names are not the same between text and json outputs"

        # reorder columns to be in the same order
        text_output = text_output[json_output.columns.values]

        pd.testing.assert_frame_equal(text_output, json_output)


class TestXGBoostParserInit:
    """Tests for the XGBoostParser.__init__ method."""

    def test_successful_initialisation(self, xgb_diabetes_model):
        """Test successful initialisation of the XGBoostParser class."""

        parser.XGBoostParser(xgb_diabetes_model)

    def test_model_not_booster_exception(self):
        """Test that a TypeError is raised if model is not a Booster."""

        with pytest.raises(
            TypeError,
            match="model is not in expected types <class 'xgboost.core.Booster'>, got <class 'tuple'>",
        ):

            parser.XGBoostParser((1, 2, 3))

    @pytest.mark.parametrize(
        "dump_reader", [(parser.JsonDumpReader), (parser.TextDumpReader)]
    )
    def test_attributes_set(self, xgb_diabetes_model, dump_reader):
        """Test model, dump_type and reader attributes are correctly set."""

        xgboost_parser = parser.XGBoostParser(xgb_diabetes_model, dump_reader())

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

            parser.XGBoostParser(xgb_diabetes_model)


class TestXGBoostParserParseModel:
    """Tests for the XGBoostParser.parse_model method."""

    def test_successful_call(self, xgb_diabetes_model):
        """Test successful call of the XGBoostParser.parse_model method."""

        xgboost_parser = parser.XGBoostParser(xgb_diabetes_model)

        results = xgboost_parser.parse_model()

        assert (
            type(results) is parser.ParsedXGBoostTabularTrees
        ), "output from parse_model not ParsedXGBoostTabularTrees type"

    def test_model_dumped_then_read(self, mocker, xgb_diabetes_model):
        """Test the booster calls dump_model and the output is then read
        with read_dump."""

        xgboost_parser = parser.XGBoostParser(
            xgb_diabetes_model, parser.JsonDumpReader()
        )

        dump_model_spy = mocker.spy(xgb.core.Booster, "dump_model")
        read_dump_spy = mocker.spy(parser.JsonDumpReader, "read_dump")

        xgboost_parser.parse_model()

        assert dump_model_spy.call_count == 1, "xgb.Booster.dump_model not called once"

        assert (
            read_dump_spy.call_count == 1
        ), "parser.JsonDumpReader.read_dump not called once"

        dump_model_call_args = dump_model_spy.call_args_list[0]

        assert dump_model_call_args[1][
            "with_stats"
        ], "xgboost model not dumped with with_stats = True"

        assert (
            dump_model_call_args[1]["dump_format"] == parser.JsonDumpReader.dump_type
        ), "xgboost model not dumped with dump_format from parser.JsonDumpReader"

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

        xgboost_parser = parser.XGBoostParser(
            xgb_diabetes_model, parser.JsonDumpReader()
        )

        read_dump_return_value = 12345

        read_dump_mock = mocker.patch.object(
            parser.JsonDumpReader, "read_dump", return_value=read_dump_return_value
        )
        parsed_xgb_tabular_trees_mock = mocker.patch.object(
            parser.ParsedXGBoostTabularTrees, "__init__", return_value=None
        )

        xgboost_parser.parse_model()

        assert (
            read_dump_mock.call_count == 1
        ), "parser.JsonDumpReader.read_dump not called once"

        assert (
            parsed_xgb_tabular_trees_mock.call_count == 1
        ), "parser.ParsedXGBoostTabularTrees.__init__ not called once"

        init_call_args = parsed_xgb_tabular_trees_mock.call_args_list[0]

        assert init_call_args[0] == (
            read_dump_return_value,
        ), "parser.ParsedXGBoostTabularTrees.__init__ not called with the output from read_dump"

        assert (
            init_call_args[1] == {}
        ), "keyword args in parser.ParsedXGBoostTabularTrees.__init__ call not as expected"
