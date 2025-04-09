"""
This is a boilerplate pipeline 'model_serving'
generated using Kedro 0.19.12
"""

from kedro.pipeline import node, Pipeline, pipeline  # noqa
from .nodes import predict_production_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline([
        node(
            func=predict_production_data,
            inputs=["model", "production_data", "params:features"],
            outputs="predictions",
            name="predict_production_data_node",
        )
    ])
