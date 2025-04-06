Projeto de machine learning com o objetivo de prever se Kobe Bryant acertou ou errou uma tentativa de arremesso, utilizando abordagens de classificação e regressão. O projeto é baseado no dataset [Kobe Bryant Shot Selectio](https://www.kaggle.com/c/kobe-bryant-shot-selection/overview), disponível no Kaggle.

```mermaid
flowchart
    subgraph data_ingestion[pipeline: data_ingestion]
        direction TB
        coletaDeDados[Coleta de dados: request na api do Github] --> raw_kobe_shots_dev_dataset[file: data/01_raw/dataset_kobe_dev.parquet]
        coletaDeDados --> raw_kobe_shots_prod_dataset[file: data/01_raw/dataset_kobe_prod.parquet]
    end
    data_ingestion[Coleta de dados] --> raw_files[Raw Files]
```
