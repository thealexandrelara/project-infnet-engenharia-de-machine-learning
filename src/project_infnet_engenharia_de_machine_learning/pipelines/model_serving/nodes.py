"""
This is a boilerplate pipeline 'model_serving'
generated using Kedro 0.19.12
"""
import pandas as pd
from mlflow.pyfunc import PyFuncModel


def predict_production_data(model: PyFuncModel, production_data: pd.DataFrame, features: dict) -> pd.DataFrame:
    """
    Predict the production data using the trained model.
    Args:
        production_data (pd.DataFrame): The production data to predict.
        model: The trained model to use for prediction.
    Returns:
        pd.DataFrame: The predicted production data.
    """
    processed_production_data = production_data.dropna()
    production_data = processed_production_data[features]
    predictions = model.predict(production_data)

    # Convert predictions to DataFrame
    predictions_df = pd.DataFrame(predictions, columns=["predictions"])

    return predictions_df
