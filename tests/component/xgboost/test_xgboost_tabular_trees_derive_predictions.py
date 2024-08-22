"""Tests for the XGBoostTabularTrees.derive_predictions method."""

import pandas as pd
import pytest
import xgboost as xgb

from tabular_trees.xgboost.xgboost_tabular_trees import XGBoostTabularTrees


def derive_depths(df) -> pd.DataFrame:
    """Derive node depths for all nodes.

    Returns
    -------
    pd.DataFrame
        Tree data (trees attribute) with 'Depth' column added.

    """
    if not (df.groupby("Tree")["Node"].first() == 0).all():
        raise ValueError("first node by tree must be the root node (0)")

    df["Depth"] = 0

    for row_number in range(df.shape[0]):
        row = df.iloc[row_number]

        # for non-leaf nodes, increase child node depths by 1
        if row["Feature"] != "Leaf":
            df.loc[df["ID"] == row["Yes"], "Depth"] = row["Depth"] + 1
            df.loc[df["ID"] == row["No"], "Depth"] = row["Depth"] + 1

    return df


@pytest.mark.parametrize("lambda_", [(0.0), (2.0)])
def test_predictions_calculated_correctly(lambda_, xgb_diabetes_dmatrix):
    """Test that the derived node prediction values are correct."""
    model_for_predictions = xgb.train(
        params={"verbosity": 0, "max_depth": 3, "lambda": lambda_},
        dtrain=xgb_diabetes_dmatrix,
        num_boost_round=10,
    )

    trees_data = model_for_predictions.trees_to_dataframe()

    xgboost_tabular_trees = XGBoostTabularTrees(trees_data, lambda_)

    predictions = xgboost_tabular_trees.derive_predictions()

    depths = derive_depths(xgboost_tabular_trees.trees.copy())

    predictions["Depth"] = depths["Depth"]

    # loop through internal nodes, non-root nodes
    for row_number in range(predictions.shape[0]):
        row = predictions.iloc[row_number]

        if (row["Feature"] != "Leaf") and (row["Node"] > 0):
            # build model with required number of trees and depth of the
            # current node, so in this tree the node is a leaf node
            if row["Tree"] == 0:
                model = xgb.train(
                    params={
                        "verbosity": 0,
                        "max_depth": row["Depth"],
                        "lambda": lambda_,
                    },
                    dtrain=xgb_diabetes_dmatrix,
                    num_boost_round=row["Tree"] + 1,
                )

            # if the number of trees required is > 1 then build the first n - 1
            # trees at the maximum depth, then build the last tree at the depth
            # of the current node
            else:
                model_n = xgb.train(
                    params={
                        "verbosity": 0,
                        "max_depth": predictions["Depth"].max(),
                        "lambda": lambda_,
                    },
                    dtrain=xgb_diabetes_dmatrix,
                    num_boost_round=row["Tree"],
                )

                model = xgb.train(
                    params={
                        "verbosity": 0,
                        "max_depth": row["Depth"],
                        "lambda": lambda_,
                    },
                    dtrain=xgb_diabetes_dmatrix,
                    num_boost_round=1,
                    xgb_model=model_n,
                )

            model_trees = model.trees_to_dataframe()

            round_to_digits = 4

            derived_prediction = round(row["weight"], round_to_digits)
            prediction_from_leaf_node = round(
                model_trees.loc[model_trees["ID"] == row["ID"], "Gain"].item(),
                round_to_digits,
            )

            assert derived_prediction == prediction_from_leaf_node, (
                f"""derived internal node prediction for node {row["ID"]} """
                "incorrect (rounding to 3dp)"
            )
