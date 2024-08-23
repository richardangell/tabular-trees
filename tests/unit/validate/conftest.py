import pandas as pd
import pytest
import xgboost as xgb


def build_depth_2_single_tree_xgb(data: xgb.DMatrix) -> xgb.Booster:
    """Single tree, max depth 2.

    Learnign rate is 1, lambda is 0.

    """
    model = xgb.train(
        params={"verbosity": 0, "max_depth": 2, "eta": 1, "lambda": 0},
        dtrain=data,
        num_boost_round=1,
    )

    return model


@pytest.fixture(scope="session")
def two_way_monotonic_increase_x2() -> pd.DataFrame:
    """2 way interaction that is monotonically increasing across both factors.

    Data is as follows:
     a   b      response
    -1  -1      130
    -1  -1      130
    -1   1      180
    -1   1      180
     1  -1      380
     1  -1      380
     1   1      420
     1   1      420

    """
    data = pd.DataFrame(
        {
            "a": [-1, -1, -1, -1, 1, 1, 1, 1],
            "b": [-1, -1, 1, 1, -1, -1, 1, 1],
            "response": [130, 130, 180, 180, 380, 380, 420, 420],
        }
    )

    return data


@pytest.fixture(scope="session")
def two_way_monotonic_increase_decrease() -> pd.DataFrame:
    """2 way interaction where factors are monotonically increasing then decreasing.

    Data is as follows:
     a   b      response
    -1  -1      180
    -1  -1      180
    -1   1      130
    -1   1      130
     1  -1      420
     1  -1      420
     1   1      380
     1   1      380

    """
    data = pd.DataFrame(
        {
            "a": [-1, -1, -1, -1, 1, 1, 1, 1],
            "b": [-1, -1, 1, 1, -1, -1, 1, 1],
            "response": [180, 180, 130, 130, 420, 420, 380, 380],
        }
    )

    return data


@pytest.fixture(scope="session")
def two_way_monotonic_decrease_increase() -> pd.DataFrame:
    """2 way interaction where factors are monotonically decreasing then increasing.

    Data is as follows:
     a   b      response
    -1  -1      380
    -1  -1      380
    -1   1      420
    -1   1      420
     1  -1      130
     1  -1      130
     1   1      180
     1   1      180

    """
    data = pd.DataFrame(
        {
            "a": [-1, -1, -1, -1, 1, 1, 1, 1],
            "b": [-1, -1, 1, 1, -1, -1, 1, 1],
            "response": [380, 380, 420, 420, 130, 130, 180, 180],
        }
    )

    return data


@pytest.fixture(scope="session")
def two_way_monotonic_decrease_x2() -> pd.DataFrame:
    """2 way interaction that is monotonically decreasing for boths factors.

    Data is as follows:
     a   b      response
    -1  -1      420
    -1  -1      420
    -1   1      380
    -1   1      380
     1  -1      180
     1  -1      180
     1   1      130
     1   1      130

    """
    data = pd.DataFrame(
        {
            "a": [-1, -1, -1, -1, 1, 1, 1, 1],
            "b": [-1, -1, 1, 1, -1, -1, 1, 1],
            "response": [420, 420, 380, 380, 180, 180, 130, 130],
        }
    )

    return data


@pytest.fixture(scope="session")
def two_way_monotonic_increase_x2_dmatrix(two_way_monotonic_increase_x2) -> xgb.DMatrix:
    """two_way_monotonic_increase_x2 data in DMatrix form."""
    return xgb.DMatrix(
        two_way_monotonic_increase_x2[["a", "b"]],
        label=two_way_monotonic_increase_x2["response"],
        base_margin=[0] * two_way_monotonic_increase_x2.shape[0],
    )


@pytest.fixture(scope="session")
def two_way_monotonic_increase_decrease_dmatrix(
    two_way_monotonic_increase_decrease,
) -> xgb.DMatrix:
    """two_way_monotonic_increase_decrease data in DMatrix form."""
    return xgb.DMatrix(
        two_way_monotonic_increase_decrease[["a", "b"]],
        label=two_way_monotonic_increase_decrease["response"],
        base_margin=[0] * two_way_monotonic_increase_decrease.shape[0],
    )


@pytest.fixture(scope="session")
def two_way_monotonic_decrease_increase_dmatrix(
    two_way_monotonic_decrease_increase,
) -> xgb.DMatrix:
    """two_way_monotonic_decrease_increase data in DMatrix form."""
    return xgb.DMatrix(
        two_way_monotonic_decrease_increase[["a", "b"]],
        label=two_way_monotonic_decrease_increase["response"],
        base_margin=[0] * two_way_monotonic_decrease_increase.shape[0],
    )


@pytest.fixture(scope="session")
def two_way_monotonic_decrease_x2_dmatrix(two_way_monotonic_decrease_x2) -> xgb.DMatrix:
    """two_way_monotonic_decrease_x2 data in DMatrix form."""
    return xgb.DMatrix(
        two_way_monotonic_decrease_x2[["a", "b"]],
        label=two_way_monotonic_decrease_x2["response"],
        base_margin=[0] * two_way_monotonic_decrease_x2.shape[0],
    )


@pytest.fixture(scope="session")
def two_way_monotonic_increase_x2_xgb_model(
    two_way_monotonic_increase_x2_dmatrix,
) -> xgb.Booster:
    """Model with single 2 way interaction both features monotonically increasing."""
    return build_depth_2_single_tree_xgb(two_way_monotonic_increase_x2_dmatrix)


@pytest.fixture(scope="session")
def two_way_monotonic_increase_decrease_xgb_model(
    two_way_monotonic_increase_decrease_dmatrix,
) -> xgb.Booster:
    """Model with single 2 way interaction.

    First feature is monotonically increasing, second feature is monotonically
    decreasing with the response.

    """
    return build_depth_2_single_tree_xgb(two_way_monotonic_increase_decrease_dmatrix)


@pytest.fixture(scope="session")
def two_way_monotonic_decrease_increase_xgb_model(
    two_way_monotonic_decrease_increase_dmatrix,
) -> xgb.Booster:
    """Model with single 2 way interaction.

    First feature is monotonically decreasing, second feature is monotonically
    increasing with the response.

    """
    return build_depth_2_single_tree_xgb(two_way_monotonic_decrease_increase_dmatrix)


@pytest.fixture(scope="session")
def two_way_monotonic_decrease_x2_xgb_model(
    two_way_monotonic_decrease_x2_dmatrix,
) -> xgb.Booster:
    """Model with single 2 way interaction both features monotonically decreasing."""
    return build_depth_2_single_tree_xgb(two_way_monotonic_decrease_x2_dmatrix)
