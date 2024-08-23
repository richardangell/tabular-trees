import pandas as pd
import pytest

from tabular_trees.xgboost.dump_reader import JsonDumpReader, TextDumpReader


@pytest.mark.parametrize("with_stats", [(False), (True)])
def test_text_and_json_readers_equal(with_stats, tmp_path, xgb_diabetes_model):
    """Test that reading text and json files give the same output."""
    text_dump = str(tmp_path.joinpath("dump_raw.txt"))
    json_dump = str(tmp_path.joinpath("dump_raw.json"))

    xgb_diabetes_model.dump_model(text_dump, with_stats=with_stats, dump_format="text")
    xgb_diabetes_model.dump_model(json_dump, with_stats=with_stats, dump_format="json")

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
