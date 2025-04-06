"""
This is a boilerplate pipeline 'data_science'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=split_data,
            inputs=["model_input_table", "params:model_options"],
            outputs=["base_train", "base_test"],
            name="split_data_node",
        )
    ])
