import pandas as pd
import pytest
from eli5 import explain_prediction_df
from sklearn.datasets import load_boston
from pandas.testing import assert_series_equal
import tabular_trees

import build_model


def test_prediction_decomposition_eqal_eli5():
    """Test that the prediction decomposition outputs from xgb.explainer.decompose_prediction are eqaul to the outputs from eli5."""

    model = build_model.build_depth_3_model()

    boston = load_boston()

    boston_data = pd.DataFrame(boston["data"], columns=boston["feature_names"])

    row_data = boston_data.iloc[[0]]

    eli5_decomposition = explain_prediction_df(model, row_data)

    column_mapping = {"<BIAS>": "base"}

    # create mapping because eli5 output will have feature names x0, x1 etc..
    for i, x in enumerate(boston["feature_names"]):

        column_mapping[f"x{i}"] = x

    eli5_decomposition["feature_mapped"] = eli5_decomposition["feature"].map(
        column_mapping
    )

    tabular_trees_trees_df = tabular_trees.xgb.parser.parse_model(model)

    tabular_trees_decomposition = tabular_trees.xgb.explainer.decompose_prediction(
        tabular_trees_trees_df.tree_data, row_data
    )

    # aggregate tabular_trees output to variable level, by default it is at tree x node level
    tabular_trees_decomposition_agg = pd.DataFrame(
        tabular_trees_decomposition.groupby("contributing_var").contribution.sum()
    ).reset_index()

    decomposition_compare_df = tabular_trees_decomposition_agg.merge(
        eli5_decomposition[["feature_mapped", "weight"]],
        how="left",
        left_on="contributing_var",
        right_on="feature_mapped",
        indicator=True,
    )

    # check merge is 1:1 i.e. both have same variables
    if (
        decomposition_compare_df["_merge"] == "both"
    ).sum() < decomposition_compare_df.shape[0]:

        pytest.fail(
            f"different features in eli5 and tabular_trees (merge not 1:1)\n\n{decomposition_compare_df}"
        )

    # check equality between prediction decomposition values
    assert_series_equal(
        left=decomposition_compare_df["weight"],
        right=decomposition_compare_df["contribution"],
        check_names=False,
        check_exact=False,
    )
