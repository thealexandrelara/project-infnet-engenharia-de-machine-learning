"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

import os

import pandas as pd
from kedro_mlflow.io.metrics import MlflowMetricDataset
from PIL import Image
from pycaret.classification import ClassificationExperiment
from sklearn.metrics import log_loss

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

def _save_plot_image_to_reporting_folder(
    plot_name: str,
    experiment: ClassificationExperiment,
    model,
    output_path: str,
):
    """
    Save the image to the specified output path.
    Args:
        image (Image): The image to save.
        output_path (str): The path where the image will be saved.
    """
    saved_image_path = experiment.plot_model(model, plot=plot_name, save=True)

    if os.path.exists(saved_image_path):
        os.replace(saved_image_path, output_path)

    with Image.open(output_path) as img:
        img.load()
        return img.copy()

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
    scoring_grid = experiment.pull()
    log_loss_score = scoring_grid['Log Loss'][0]
    f1_score = scoring_grid['F1'][0]
    logistic_regression_f1_score_metric.save(log_loss_score)
    logistic_regression_log_loss_metric.save(f1_score)

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
    scoring_grid = experiment.pull()
    log_loss_score = scoring_grid['Log Loss'][0]
    f1_score = scoring_grid['F1'][0]
    decision_tree_log_loss_metric.save(log_loss_score)
    decision_tree_f1_score_metric.save(f1_score)

    return decision_tree_model

def save_auc_plot(
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
    return _save_plot_image_to_reporting_folder(
        plot_name='auc',
        experiment=experiment,
        model=model,
        output_path=output_path,
    )

def save_confusion_matrix_plot(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    parameters: dict,
    model,
    output_path: str,
):
    """
    Plot the Confusion Matrix for the given model.
    Args:
        model: The trained model.
    Returns:
        Image: The confusion matrix image.
    """
    experiment = _createClassificationExperiment(train_data=train_data, test_data=test_data, parameters=parameters)
    return _save_plot_image_to_reporting_folder(
        plot_name='confusion_matrix',
        experiment=experiment,
        model=model,
        output_path=output_path,
    )

def save_feature_importance_plot(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    parameters: dict,
    model,
    output_path: str,
):
    """
    Plot the Confusion Matrix for the given model.
    Args:
        model: The trained model.
    Returns:
        Image: The confusion matrix image.
    """
    experiment = _createClassificationExperiment(train_data=train_data, test_data=test_data, parameters=parameters)
    return _save_plot_image_to_reporting_folder(
        plot_name='feature',
        experiment=experiment,
        model=model,
        output_path=output_path,
    )
