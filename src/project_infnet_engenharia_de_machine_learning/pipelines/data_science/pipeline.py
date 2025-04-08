"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import train_logistic_regression_model, train_decision_tree_model, save_auc_plot, save_confusion_matrix_plot, save_feature_importance_plot


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_logistic_regression_model,
            inputs=["base_train", "base_test", "params:model_options"],
            outputs='logistic_regression_model',
            name="train_logistic_regression_model_node",
        ),
        node(
            func=train_decision_tree_model,
            inputs=["base_train", "base_test", "params:model_options"],
            outputs='decision_tree_model',
            name="train_decision_tree_model_node",
        ),
        node(
            func=save_auc_plot,
            inputs=["base_train", "base_test", "params:model_options", "logistic_regression_model", "params:logistic_regression_model_auc_filename_path"],
            outputs='logistic_regression_model_auc',
            name="save_logistic_regression_model_auc_plot_node",
        ),
        node(
            func=save_auc_plot,
            inputs=["base_train", "base_test", "params:model_options", "decision_tree_model", "params:decision_tree_model_auc_filename_path"],
            outputs='decision_tree_model_auc',
            name="save_decision_tree_model_auc_plot_node",
        ),
        node(
            func=save_confusion_matrix_plot,
            inputs=["base_train", "base_test", "params:model_options", "logistic_regression_model", "params:logistic_regression_model_confusion_matrix_filename_path"],
            outputs='logistic_regression_model_confusion_matrix',
            name="save_logistic_regression_model_confusion_matrix_plot_node",
        ),
        node(
            func=save_confusion_matrix_plot,
            inputs=["base_train", "base_test", "params:model_options", "decision_tree_model", "params:decision_tree_model_confusion_matrix_filename_path"],
            outputs='decision_tree_model_confusion_matrix',
            name="save_decision_tree_model_confusion_matrix_plot_node",
        ),
        node(
            func=save_feature_importance_plot,
            inputs=["base_train", "base_test", "params:model_options", "logistic_regression_model", "params:logistic_regression_model_feature_importance_filename_path"],
            outputs='logistic_regression_model_feature_importance',
            name="save_logistic_regression_model_feature_importance_plot_node",
        ),
        node(
            func=save_feature_importance_plot,
            inputs=["base_train", "base_test", "params:model_options", "decision_tree_model", "params:decision_tree_model_feature_importance_filename_path"],
            outputs='decision_tree_model_feature_importance',
            name="save_decision_tree_model_feature_importance_plot_node",
        )
    ])
