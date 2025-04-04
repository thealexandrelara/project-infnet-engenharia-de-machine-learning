"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""
import pandas as pd


def _rename_columns(df):
    """
    Rename columns to match the expected format for the model.
    """
    df = df.rename(columns={
        'lon': 'lng',
    })
    return df

def _filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame to keep only the relevant columns for the model.
    """
    return df[['lat', 'lng', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']].copy()

def preprocess_kobe_shots(kobe_shots: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the Kobe shots DataFrame by renaming columns and converting data types.

    The output dataframe is saved in the intermediate layer.

    Args:
        kobe_shots (pd.DataFrame): The raw Kobe shots DataFrame.
    Returns:
        pd.DataFrame: The preprocessed Kobe shots DataFrame.
    """
    kobe_shots = _rename_columns(kobe_shots)
    kobe_shots['shot_made_flag'] = kobe_shots['shot_made_flag'].astype('Int8')
    return kobe_shots

def create_model_input_table(kobe_shots: pd.DataFrame) -> pd.DataFrame:
    """
    Create the model input table by parsing, filtering and transforming the data.

    The output is saved in the primary layer.

    Args:
        kobe_shots (pd.DataFrame): The preprocessed Kobe shots DataFrame.
    Returns:
        pd.DataFrame: The model input table.
    """
    model_input_table = _filter_columns(kobe_shots)
    model_input_table = model_input_table.dropna()
    return model_input_table
