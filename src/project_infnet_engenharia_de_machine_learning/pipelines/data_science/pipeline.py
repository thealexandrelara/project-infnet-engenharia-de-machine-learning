"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import train_logistic_regression_model, train_decision_tree_model, save_plots


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_logistic_regression_model,
            inputs=["base_train", "base_test", "params:model_options"],
            outputs=['logistic_regression_model', 'logistic_regression_model_with_proba'],
            name="train_logistic_regression_model_node",
        ),
        node(
            func=train_decision_tree_model,
            inputs=["base_train", "base_test", "params:model_options"],
            outputs='decision_tree_model',
            name="train_decision_tree_model_node",
        ),
        node(
            func=save_plots,
            inputs=["base_train", "base_test", "params:model_options", "logistic_regression_model", "params:logistic_regression_model_plot_images_path"],
            outputs=['logistic_regression_model_auc', 'logistic_regression_model_confusion_matrix', 'logistic_regression_model_feature_importance'],
            name="save_logistic_regression_model_plots_node",
        ),
        node(
            func=save_plots,
            inputs=["base_train", "base_test", "params:model_options", "logistic_regression_model", "params:decision_tree_model_plot_images_path"],
            outputs=['decision_tree_model_auc', 'decision_tree_model_confusion_matrix', 'decision_tree_model_feature_importance'],
            name="save_decision_tree_model_plots_node",
        ),
    ])
