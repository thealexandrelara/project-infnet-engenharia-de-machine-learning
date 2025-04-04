"""
This is a boilerplate pipeline 'data_processing'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import preprocess_dataset_kobe_dev


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=preprocess_dataset_kobe_dev,
            inputs="dataset_kobe_dev",
            outputs="preprocessed_kobe_dev_dataset",
            name="preprocess_dataset_kobe_dev_node",
        ),
    ])
