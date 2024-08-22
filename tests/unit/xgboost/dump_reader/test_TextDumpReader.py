from tabular_trees.xgboost.dump_reader import TextDumpReader


def test_dump_type_attribute():
    assert TextDumpReader.dump_type == "text"
