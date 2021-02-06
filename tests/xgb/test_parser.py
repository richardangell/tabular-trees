import pandas as pd
import pygbmexpl

import build_model


def test_text_json_parsing_equal(tmp_path):
    """Test that parsing an xgboost model dumped to text file and json file gives the same output."""

    model = build_model.build_depth_3_model()

    # filepaths to dump model to in different combinations of format (text/json) and with/without stats
    text_dump = str(tmp_path.joinpath("dump_raw.txt"))
    text_dump_no_stats = str(tmp_path.joinpath("dump_raw_no_stats.txt"))
    json_dump = str(tmp_path.joinpath("dump_raw.json"))
    json_dump_no_stats = str(tmp_path.joinpath("dump_raw_no_stats.json"))

    # dump model to files above
    model.dump_model(text_dump, with_stats=True, dump_format="text")
    model.dump_model(text_dump_no_stats, with_stats=False, dump_format="text")
    model.dump_model(json_dump, with_stats=True, dump_format="json")
    model.dump_model(json_dump_no_stats, with_stats=False, dump_format="json")

    # parse dumped files
    tree_df1 = pygbmexpl.xgb.parser._read_dump_text(text_dump, return_raw_lines=False)
    tree_df2 = pygbmexpl.xgb.parser._read_dump_text(
        text_dump_no_stats, return_raw_lines=False
    )
    tree_df3 = pygbmexpl.xgb.parser._read_dump_json(json_dump, return_raw_lines=False)
    tree_df4 = pygbmexpl.xgb.parser._read_dump_json(
        json_dump_no_stats, return_raw_lines=False
    )

    # check the equivalent parsed text/json outputs are equal
    pd.testing.assert_frame_equal(tree_df1, tree_df3)
    pd.testing.assert_frame_equal(tree_df2, tree_df4)
