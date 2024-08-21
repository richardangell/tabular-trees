"""Module containing checks to be used elsewhere in the package."""

import abc
from typing import Any, Callable, Type

import pandas as pd


def check_type(
    obj: Any,
    expected_types: Type | tuple[Type, ...],
    obj_name: str,
    none_allowed: bool = False,
) -> None:
    """Check object is of given types and raise a TypeError if not.

    Parameters
    ----------
    obj : Any
        Any object to check the type of.

    expected_types : Union[Type, Tuple[Union[Type, Type[abc.ABCMeta]], ...]]
        Expected type or tuple of expected types of obj.

    none_allowed : bool = False
        Is None an allowed value for obj?

    """
    expected_types_types = [type, abc.ABCMeta, type(Callable)]

    if type(expected_types) is tuple:
        if not all(
            type(expected_type) in expected_types_types
            for expected_type in expected_types
        ):
            raise TypeError("all elements in expected_types must be types")

    else:
        if type(expected_types) not in expected_types_types:
            raise TypeError("expected_types must be a type when passing a single type")

    if obj is None and not none_allowed:
        raise TypeError(f"{obj_name} is None and not is not allowed")

    elif obj is not None and not isinstance(obj, expected_types):
        raise TypeError(
            f"{obj_name} is not in expected types {expected_types}, got {type(obj)}"
        )


def check_condition(condition: bool, error_message_text: str) -> None:
    """Check that condition, which evaluates to a bool, is True.

    Parameters
    ----------
    condition : bool
        Condition that evaluates to bool, to check.

    error_message_text : str
        Message describing condition. Will be included in the exception message if
        condition does not evaluate to True.

    Raises
    ------
    ValueError
        If condition does not evalute to True.

    """
    check_type(condition, bool, "condition")
    check_type(error_message_text, str, "error_message_text")

    if not condition:
        raise ValueError(f"condition: [{error_message_text}] not met")


def check_df_columns(
    df: pd.DataFrame,
    expected_columns: list[str],
    allow_unspecified_columns: bool = False,
) -> None:
    """Check if a pd.DataFrame has expected columns.

    Extra columns can be allowed by specifying the allow_unspecified_columns argument.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.

    expected_columns : list
        List of columns expected to be in df.

    allow_unspecified_columns : bool, default = False
        Should extra, unspecified columns in df be allowed?

    """
    check_type(df, pd.DataFrame, "df")
    check_type(expected_columns, list, "expected_columns")
    check_type(allow_unspecified_columns, bool, "allow_unspecified_columns")

    df_cols = df.columns.values.tolist()

    in_expected_not_df = sorted(set(expected_columns) - set(df_cols))

    if len(in_expected_not_df) > 0:
        raise ValueError(f"expected columns not in df; {in_expected_not_df}")

    if not allow_unspecified_columns:
        in_df_not_expected = sorted(set(df_cols) - set(expected_columns))

        if len(in_df_not_expected) > 0:
            raise ValueError(
                "extra columns in df when allow_unspecified_columns = False; "
                f"{in_df_not_expected}"
            )
