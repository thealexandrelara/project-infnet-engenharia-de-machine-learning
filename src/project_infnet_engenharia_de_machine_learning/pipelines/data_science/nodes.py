"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

import os

import pandas as pd
from kedro_mlflow.io.metrics import MlflowMetricDataset
from PIL import Image
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import f1_score, log_loss

logistic_regression_log_loss_metric = MlflowMetricDataset(key="logistic_regression_model_log_loss")
logistic_regression_f1_score_metric = MlflowMetricDataset(key="logistic_regression_model_f1_score")
decision_tree_log_loss_metric = MlflowMetricDataset(key="decision_tree_model_log_loss")
decision_tree_f1_score_metric = MlflowMetricDataset(key="decision_tree_model_f1_score")

def _createClassificationExperiment(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    parameters: dict
) -> ClassificationExperiment:
    experiment = ClassificationExperiment()
    experiment.setup(
        data=train_data,
        test_data=test_data,
        target=parameters['target'],
        session_id=parameters['random_state'],
        log_data=True,
        log_plots=True,
        experiment_name='project_infnet_engenharia_de_machine_learning',
    )
    experiment.add_metric('logloss', 'Log Loss', log_loss, greater_is_better=False)

    return experiment


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
    experiment = _createClassificationExperiment(train_data=train_data, test_data=test_data, parameters=parameters)
    logistic_regression_model = experiment.create_model('lr')
    logistic_regression_predictions= experiment.predict_model(logistic_regression_model, data=test_data)
    logistic_regression_log_loss = log_loss(test_data[parameters['target']], logistic_regression_predictions["prediction_score"])
    logistic_regression_f1_score = f1_score(test_data[parameters['target']], logistic_regression_predictions["prediction_label"])
    logistic_regression_f1_score_metric.save(logistic_regression_f1_score)
    logistic_regression_log_loss_metric.save(logistic_regression_log_loss)

    return logistic_regression_model

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
    experiment = _createClassificationExperiment(train_data=train_data, test_data=test_data, parameters=parameters)
    decision_tree_model = experiment.create_model('dt')
    decision_tree_predictions= experiment.predict_model(decision_tree_model, data=test_data)
    decision_tree_log_loss = log_loss(test_data[parameters['target']], decision_tree_predictions["prediction_score"])
    decision_tree_f1_score = f1_score(test_data[parameters['target']], decision_tree_predictions["prediction_label"])
    decision_tree_log_loss_metric.save(decision_tree_log_loss)
    decision_tree_f1_score_metric.save(decision_tree_f1_score)

    return decision_tree_model

def plot_auc(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    parameters: dict,
    model,
    output_path: str,
):
    """
    Plot the AUC curve for the given model.
    Args:
        model: The trained model.
    """
    experiment = _createClassificationExperiment(train_data=train_data, test_data=test_data, parameters=parameters)
    saved_image_path = experiment.plot_model(model, plot='auc', save=True)

    if os.path.exists(saved_image_path):
        os.replace(saved_image_path, output_path)

    with Image.open(output_path) as img:
        img.load()
        return img.copy()
