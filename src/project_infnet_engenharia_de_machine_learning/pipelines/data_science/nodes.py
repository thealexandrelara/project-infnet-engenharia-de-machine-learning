"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

import pandas as pd
from pycaret.classification import ClassificationExperiment


def train_logistic_regression_model(
    train_data: pd.DataFrame,
    parameters: dict
) -> tuple[ClassificationExperiment, pd.DataFrame]:
    """
    Train a logistic regression model using the provided training data.

    Args:
        train_data (pd.DataFrame): The training data.
        parameters (dict): Parameters for the model.

    Returns:
        tuple[ClassificationExperiment, pd.DataFrame]: A tuple containing the trained experiment and the test data.
    """
    experiment = ClassificationExperiment()
    experiment.setup(
        data=train_data,
        target=parameters['target'],
        train_size=1.0,
        session_id=parameters['random_state'],
    )
    model = experiment.create_model('lr')

    return experiment

def train_decision_tree_model(
    train_data: pd.DataFrame,
    parameters: dict
) -> tuple[ClassificationExperiment, pd.DataFrame]:
    """
    Train a decision tree model using the provided training data.
    Args:
        train_data (pd.DataFrame): The training data.
        parameters (dict): Parameters for the model.
    Returns:
        tuple[ClassificationExperiment, pd.DataFrame]: A tuple containing the trained experiment and the test data.
    """
    experiment = ClassificationExperiment()
    experiment.setup(
        data=train_data,
        target=parameters['target'],
        train_size=1.0,
        session_id=parameters['random_state'],
    )
    model = experiment.create_model('dt')

    return experiment
