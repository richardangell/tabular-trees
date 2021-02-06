import pandas as pd


def check_df_columns(df, expected_columns, allow_unspecified_columns = False):
    '''Function to check if a pd.DataFrame has expected columns.
    
    Extra columns can be allowed by specifying the allow_unspecified_columns argument.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to check.

    expected_columns : list
        List of columns expected to be in df.

    allow_unspecified_columns : bool, default = False
        Should extra, unspecified columns in df be allowed?

    '''

    check_type(df, 'df', pd.DataFrame)
    check_type(expected_columns, 'expected_columns', list)
    check_type(allow_unspecified_columns, 'allow_unspecified_columns', bool)

    df_cols = df.columns.values.tolist()

    in_expected_not_df = list(set(expected_columns) - set(df_cols))

    if len(in_expected_not_df) > 0:

        raise ValueError('Expected columns not in df; ' + str(in_expected_not_df))

    if not allow_unspecified_columns:

        in_df_not_expected = list(set(df_cols) - set(expected_columns))

        if len(in_df_not_expected) > 0:

            raise ValueError('Extra columns in df when allow_unspecified_columns = False; ' + str(in_df_not_expected))


def check_type(obj, obj_name, expected_type):
    """Check whether an object is of a given type.
    
    Parameters
    ----------
    obj : any
        Object to check.
    
    obj_name : str
        Name for object, will be printed in error message if not of expected type.

    expected_type : type
        Expected type of obj. 

    """

    if type(obj) is not expected_type:

        raise TypeError(f'{obj_name} is not expected type; {expected_type} got {type(obj)}')

