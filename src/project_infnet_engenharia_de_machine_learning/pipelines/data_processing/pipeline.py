"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import preprocess_kobe_shots, create_model_input_table, split_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_kobe_shots,
            inputs="raw_kobe_shots_dev",
            outputs="preprocessed_kobe_shots",
            name="create_preprocessed_kobe_shots_node",
        ),
        node(
            func=preprocess_kobe_shots,
            inputs="raw_kobe_shots_prod",
            outputs="preprocessed_kobe_shots_prod",
            name="create_preprocessed_kobe_shots_prod_node",
        ),
        node(
            func=create_model_input_table,
            inputs="preprocessed_kobe_shots",
            outputs="model_input_table",
            name="create_model_input_table_node",
        ),
        node(
            func=split_data,
            inputs=["model_input_table", "params:model_options"],
            outputs=["base_train", "base_test"],
            name="split_data_node",
        )
    ])
