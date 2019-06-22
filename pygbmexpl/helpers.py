import pandas as pd



def check_df_columns(df, expected_columns, allow_unspecified_columns = False):
    '''Function to check if pandas DataFrame has expected columns.'''

    if not isinstance(df, pd.DataFrame):

        raise TypeError('Expecting df to be a pd.DataFrame')

    if not isinstance(expected_columns, list):

        raise TypeError('Expecting expected_columns to be a list')

    if not isinstance(allow_unspecified_columns, bool):

        raise TypeError('Expecting allow_unspecified_columns to be a bool')

    df_cols = df.columns.values.tolist()

    in_expected_not_df = list(set(expected_columns) - set(df_cols))

    if len(in_expected_not_df) > 0:

        raise ValueError('Expected columns not in df; ' + str(in_expected_not_df))

    if not allow_unspecified_columns:

        in_df_not_expected = list(set(df_cols) - set(expected_columns))

        if len(in_df_not_expected) > 0:

            raise ValueError('Extra columns in df when allow_unspecified_columns = False; ' + str(in_df_not_expected))


