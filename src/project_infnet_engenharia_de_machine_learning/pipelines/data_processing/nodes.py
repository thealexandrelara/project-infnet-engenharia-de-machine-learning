"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""
import pandas as pd


def _rename_columns(df):
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
    Preprocess the Kobe shots DataFrame by renaming columns and filtering.
    """
    kobe_shots = _rename_columns(kobe_shots)
    return kobe_shots

def create_model_input_table(kobe_shots: pd.DataFrame) -> pd.DataFrame:
    """
    Create the model input table by parsing, filtering and transforming the data.
    """
    model_input_table = _filter_columns(kobe_shots)
    model_input_table = model_input_table.dropna()
    return model_input_table
