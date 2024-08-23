from tabular_trees.xgboost.dump_reader import DumpType, TextDumpReader


def test_dump_type_attribute():
    assert TextDumpReader.dump_type == DumpType.TEXT
