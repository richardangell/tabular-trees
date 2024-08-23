"""Helper functions."""

from typing import SupportsFloat, SupportsIndex, Union

from typing_extensions import Buffer, TypeAlias

ReadableBuffer: TypeAlias = Buffer
ConvertibleToFloat: TypeAlias = str | ReadableBuffer | SupportsFloat | SupportsIndex


class FloatFixedString(float):
    """Float with a defined string representation."""

    def __new__(
        cls, value: ConvertibleToFloat, string_representation: str
    ) -> "FloatFixedString":
        """Create and return a new object."""
        return float.__new__(cls, value)

    def __init__(self, value: ConvertibleToFloat, string_representation: str):
        float.__init__(value)
        self.string_representation = string_representation

    def __str__(self) -> str:
        """Return the fixed string representation for the float."""
        return self.string_representation

    def __repr__(self) -> str:
        """Return the fixed string representation for the float."""
        return self.string_representation


def try_convert_string_to_int_or_float(value: str) -> Union[int, FloatFixedString, str]:
    """Convert a string value to int, FloatFixedString or return the input string.

    Try to convert to int or FloatFixedString first, if this results in a ValueError
    then return the original string value.

    """
    try:
        return convert_string_to_int_or_float(value)
    except ValueError:
        return value


def convert_string_to_int_or_float(value: str) -> Union[int, FloatFixedString]:
    """Try to convert a string to int or FloatFixedString if int conversion fails."""
    try:
        return int(value)
    except ValueError:
        return FloatFixedString(value, value)


def remove_surrounding_brackets(value: str) -> str:
    """Remove surrounding square brackets from string."""
    return value[1:-1]
