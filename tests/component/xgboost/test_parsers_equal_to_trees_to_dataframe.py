import pytest

from tabular_trees.xgboost.dump_parser import XGBoostParser
from tabular_trees.xgboost.dump_reader import (
    JsonDumpReader,
    TextDumpReader,
)


@pytest.mark.parametrize("reader", [JsonDumpReader(), TextDumpReader()])
def test_parse_output_equal_to_xgboost_output(reader, xgb_diabetes_model):
    """Test Booster.dump_model output is read with read_dump."""
    expected = xgb_diabetes_model.trees_to_dataframe()

    xgboost_parser = XGBoostParser(xgb_diabetes_model, reader)

    actual = xgboost_parser.parse_model()

    assert actual.data.shape[0] == expected.shape[0]
    assert (actual.data["tree"] == expected["Tree"]).all()
    assert (actual.data["nodeid"] == expected["Node"]).all()
    assert (actual.data["cover"] == expected["Cover"]).all()

    actual_leaves = actual.data["split"].isnull()
    expected_leaves = expected["Feature"] == "Leaf"

    assert (actual_leaves == expected_leaves).all()

    assert (
        actual.data.loc[~actual_leaves, "split"]
        == expected.loc[~expected_leaves, "Feature"]
    ).all()

    assert (
        actual.data.loc[~actual_leaves, "split_condition"]
        == expected.loc[~expected_leaves, "Split"]
    ).all()

    assert (
        actual.data.loc[~actual_leaves, "gain"]
        == expected.loc[~expected_leaves, "Gain"]
    ).all()

    assert (
        actual.data.loc[actual_leaves, "leaf"] == expected.loc[expected_leaves, "Gain"]
    ).all()
