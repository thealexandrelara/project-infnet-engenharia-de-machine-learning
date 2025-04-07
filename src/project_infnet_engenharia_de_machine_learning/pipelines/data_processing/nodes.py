"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""
import pandas as pd
from kedro_mlflow.io.metrics import MlflowMetricDataset
from pycaret.classification import ClassificationExperiment

test_percent_metric = MlflowMetricDataset(key="test_percent")
base_train_size = MlflowMetricDataset(key="base_train_size")
base_test_size = MlflowMetricDataset(key="base_test_size")

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

def split_data(
    df: pd.DataFrame,
    parameters: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the DataFrame into training and testing sets.

    Args:
        df (pd.DataFrame): The input DataFrame to be split.
        train_size (float): The proportion of the dataset to include in the training set.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training and testing DataFrames.
    """
    experiment = ClassificationExperiment()
    experiment.setup(
        data=df,
        target=parameters['target'],
        train_size=parameters['train_size'],
        session_id=parameters['random_state'],
        data_split_stratify=True,
        data_split_shuffle=True,
    )
    X_train, X_test = experiment.get_config('X_train'), experiment.get_config('X_test')
    y_train, y_test = experiment.get_config('y_train'), experiment.get_config('y_test')

    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)

    test_percent_metric.save(1.0 - parameters['train_size'])
    base_train_size.save(len(train_data))
    base_test_size.save(len(test_data))

    return train_data, test_data
