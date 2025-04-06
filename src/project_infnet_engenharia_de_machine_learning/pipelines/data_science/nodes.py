"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

import pandas as pd
from pycaret.classification import ClassificationExperiment


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

    return train_data, test_data
