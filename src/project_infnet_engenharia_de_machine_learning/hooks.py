import typing as t

import mlflow
from kedro.framework.hooks import hook_impl

PIPELINE_NAME_MAP = {
    "data_ingestion": "ColetaDados",
    "data_processing": "PreparacaoDados",
    "data_science": "Treinamento",
}

def get_pipeline_run_name(pipeline_name: str) -> str:
    return PIPELINE_NAME_MAP.get(pipeline_name, pipeline_name) or "__default__"


class ExtraMLflowHooks:
    @hook_impl
    def before_pipeline_run(self, run_params: dict[str, t.Any]):
        pipeline_name = get_pipeline_run_name(run_params["pipeline_name"])
        mlflow.set_tag("mlflow.runName", pipeline_name)
