"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""
import pandas as pd


def _filter_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the DataFrame to keep only the relevant columns for the model.
    """
    return df[['lat', 'lon', 'minutes_remaining', 'period', 'playoffs', 'shot_distance', 'shot_made_flag']].copy()

def create_model_input_table(kobe_shots: pd.DataFrame) -> pd.DataFrame:
    """
    Create the model input table by parsing, filtering and transforming the data.
    """
    model_input_table = _filter_columns(kobe_shots)
    model_input_table = model_input_table.dropna()
    return model_input_table
