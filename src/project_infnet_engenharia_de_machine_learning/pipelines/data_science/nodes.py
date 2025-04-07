"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

import pandas as pd
from kedro_mlflow.io.metrics import MlflowMetricDataset
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import f1_score, log_loss

logistic_regression_log_loss_metric = MlflowMetricDataset(key="logistic_regression_model_log_loss")
decision_tree_log_loss_metric = MlflowMetricDataset(key="decision_tree_model_log_loss")
decision_tree_f1_score_metric = MlflowMetricDataset(key="decision_tree_model_f1_score")


def train_logistic_regression_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
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
        test_data=test_data,
        target=parameters['target'],
        session_id=parameters['random_state'],
    )
    logistic_regression_model = experiment.create_model('lr')
    logistic_regression_predictions= experiment.predict_model(logistic_regression_model, data=test_data)
    logistic_regression_log_loss = log_loss(test_data[parameters['target']], logistic_regression_predictions["prediction_score"])
    logistic_regression_log_loss_metric.save(logistic_regression_log_loss)

    return experiment

def train_decision_tree_model(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
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
        test_data=test_data,
        target=parameters['target'],
        session_id=parameters['random_state'],
    )
    decision_tree_model = experiment.create_model('dt')
    decision_tree_predictions= experiment.predict_model(decision_tree_model, data=test_data)
    decision_tree_log_loss = log_loss(test_data[parameters['target']], decision_tree_predictions["prediction_score"])
    decision_tree_f1_score = f1_score(test_data[parameters['target']], decision_tree_predictions["prediction_label"])
    decision_tree_log_loss_metric.save(decision_tree_log_loss)
    decision_tree_f1_score_metric.save(decision_tree_f1_score)

    return experiment
