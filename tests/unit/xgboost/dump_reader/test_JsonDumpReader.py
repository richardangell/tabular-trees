from tabular_trees.xgboost.dump_reader import DumpType, JsonDumpReader


def test_dump_type_attribute():
    assert JsonDumpReader.dump_type == DumpType.JSON
