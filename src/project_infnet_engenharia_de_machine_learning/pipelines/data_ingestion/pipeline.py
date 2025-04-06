"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import download_kobe_shots_dev_dataset, download_kobe_shots_prod_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=download_kobe_shots_dev_dataset,
            inputs=None,
            outputs="raw_kobe_shots_dev",
            name="download_kobe_shots_dev_dataset_node",
        ),
        node(
            func=download_kobe_shots_prod_dataset,
            inputs=None,
            outputs="raw_kobe_shots_prod",
            name="download_kobe_shots_prod_dataset_node",
        ),
    ])
