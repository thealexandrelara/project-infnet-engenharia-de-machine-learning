"""
This is a boilerplate pipeline 'model_serving'
generated using Kedro 0.19.12
"""
import pandas as pd
from kedro_mlflow.io.metrics import MlflowMetricDataset
from mlflow.pyfunc import PyFuncModel
from sklearn.metrics import f1_score, log_loss

log_loss_metric = MlflowMetricDataset(key="log_loss")
f1_score_metric = MlflowMetricDataset(key="f1_score")

def predict_production_data(model: PyFuncModel, model_dev: PyFuncModel, production_data: pd.DataFrame, features: dict, target: str) -> pd.DataFrame:
    """
    Predict the production data using the trained model.
    Args:
        production_data (pd.DataFrame): The production data to predict.
        model: The trained model to use for prediction.
    Returns:
        pd.DataFrame: The predicted production data.
    """
    processed_production_data = production_data.dropna()
    X = processed_production_data[features]
    y_true = processed_production_data[target]
    y_pred_dev = model_dev.predict(X)[:, 1]
    y_pred = model.predict(X)

    model_log_loss = log_loss(y_true, y_pred_dev)
    model_f1_score = f1_score(y_true, y_pred)

    log_loss_metric.save(model_log_loss)
    f1_score_metric.save(model_f1_score)

    predictions_df = pd.DataFrame({
        "y_true": y_true,
        "y_pred": y_pred,
        "model_log_loss": model_log_loss,
        "model_f1_score": model_f1_score
    })

    return predictions_df
