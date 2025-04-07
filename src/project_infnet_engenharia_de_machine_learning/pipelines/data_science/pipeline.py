"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import train_logistic_regression_model, train_decision_tree_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=train_logistic_regression_model,
            inputs=["base_train", "base_test", "params:model_options"],
            outputs=None,
            name="train_logistic_regression_model_node",
        ),
        node(
            func=train_decision_tree_model,
            inputs=["base_train", "base_test", "params:model_options"],
            outputs=None,
            name="train_decision_tree_model_node",
        )
    ])
