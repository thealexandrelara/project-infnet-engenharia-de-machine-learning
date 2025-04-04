"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import create_model_input_table


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=create_model_input_table,
            inputs="raw_kobe_shots_dev",
            outputs="model_input_table",
            name="create_model_input_table_node",
        ),
    ])
