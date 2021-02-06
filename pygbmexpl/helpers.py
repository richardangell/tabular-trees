import pandas as pd



def check_df_columns(df, expected_columns, allow_unspecified_columns = False):
    '''Function to check if pandas DataFrame has expected columns.'''

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

    if type(obj) is not expected_type:

        raise TypeError(f'{obj_name} is not expected type; {expected_type} got {type(obj)}')

