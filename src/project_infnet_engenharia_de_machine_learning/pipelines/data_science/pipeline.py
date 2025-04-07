"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import train_logistic_regression_model, train_decision_tree_model, plot_auc


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
            func=plot_auc,
            inputs=["base_train", "base_test", "params:model_options", "logistic_regression_model", "params:logistic_regression_model_auc_filename_path"],
            outputs='logistic_regression_model_auc',
            name="plot_logistic_regression_model_auc_node",
        ),
        node(
            func=plot_auc,
            inputs=["base_train", "base_test", "params:model_options", "decision_tree_model", "params:decision_tree_model_auc_filename_path"],
            outputs='decision_tree_model_auc',
            name="plot_decision_tree_model_auc_node",
        )
    ])
