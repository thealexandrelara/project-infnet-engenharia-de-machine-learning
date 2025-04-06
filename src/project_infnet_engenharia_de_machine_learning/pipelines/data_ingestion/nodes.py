"""
The download_kobe_shots_dev_dataset and download_kobe_shots_prod_dataset functions
download the Kobe shots dataset from a URL and return it as a pandas DataFrame to be saved in the `raw` folder.
"""
import io

import pandas as pd
import requests


def _get_parquet_file_from_url(url: str) -> pd.DataFrame:
    """Download the dataset from the URL and return it as a pandas DataFrame."""
    response = requests.get(url)
    response.raise_for_status()
    df = pd.read_parquet(io.BytesIO(response.content))

    return df


def download_kobe_shots_dev_dataset() -> pd.DataFrame:
    """Download the dataset from the URL and return it as a pandas DataFrame."""
    url = "https://github.com/tciodaro/eng_ml/raw/refs/heads/main/data/dataset_kobe_dev.parquet"
    df = _get_parquet_file_from_url(url)

    return df

def download_kobe_shots_prod_dataset() -> pd.DataFrame:
    """Download the dataset from the URL and return it as a pandas DataFrame."""
    url = "https://github.com/tciodaro/eng_ml/raw/refs/heads/main/data/dataset_kobe_prod.parquet"
    df = _get_parquet_file_from_url(url)

    return df
