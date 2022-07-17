"""Module containing checks to be used elsewhere in the package."""

import pandas as pd
import abc
from typing import Any, Union, Type, Tuple


def check_type(
    obj: Any,
    expected_types: Union[Type, Tuple[Union[Type, Type[abc.ABCMeta]], ...]],
    obj_name: str,
    none_allowed: bool = False,
) -> None:
    """Function to check object is of given types and raise a TypeError if not.
    Parameters
    ----------
    obj : Any
        Any object to check the type of.
    expected_types : Union[Type, Tuple[Union[Type, Type[abc.ABCMeta]], ...]]
        Expected type or tuple of expected types of obj.
    none_allowed : bool = False
        Is None an allowed value for obj?
    """

    if type(expected_types) is tuple:

        if not all(
            [
                type(expected_type) in [type, abc.ABCMeta]
                for expected_type in expected_types
            ]
        ):

            raise TypeError("all elements in expected_types must be types")

    else:

        if not type(expected_types) in [type, abc.ABCMeta]:

            raise TypeError("expected_types must be a type when passing a single type")

    if obj is None and not none_allowed:

        raise TypeError(f"{obj_name} is None and not is not allowed")

    elif obj is not None:

        if not isinstance(obj, expected_types):

            raise TypeError(
                f"{obj_name} is not in expected types {expected_types}, got {type(obj)}"
            )


def check_condition(condition: bool, error_message_text: str):
    """Check that condition (which evaluates to a bool) is True and raise a
    ValueError if not.
    Parameters
    ----------
    condition : bool
        Condition that evaluates to bool, to check.
    error_message_text : str
        Message to print in ValueError if condition does not evalute to True.
    """

    check_type(condition, bool, "condition")
    check_type(error_message_text, str, "error_message_text")

    if not condition:

        raise ValueError(f"condition: [{error_message_text}] not met")


def check_df_columns(df, expected_columns, allow_unspecified_columns=False):
    """Function to check if a pd.DataFrame has expected columns.

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

    in_expected_not_df = sorted(list(set(expected_columns) - set(df_cols)))

    if len(in_expected_not_df) > 0:

        raise ValueError(f"expected columns not in df; {in_expected_not_df}")

    if not allow_unspecified_columns:

        in_df_not_expected = sorted(list(set(df_cols) - set(expected_columns)))

        if len(in_df_not_expected) > 0:

            raise ValueError(
                f"extra columns in df when allow_unspecified_columns = False; {in_df_not_expected}"
            )
