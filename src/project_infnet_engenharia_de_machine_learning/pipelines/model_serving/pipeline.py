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
            inputs=["best_model", "logistic_regression_model_dev", "preprocessed_kobe_shots_prod", "params:features", "params:target"],
            outputs="production_data_predictions",
            name="predict_production_data_node",
        )
    ])
