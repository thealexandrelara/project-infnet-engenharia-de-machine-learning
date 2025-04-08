Projeto de machine learning com o objetivo de prever se Kobe Bryant acertou ou errou uma tentativa de arremesso, utilizando abordagens de classificação e regressão. O projeto é baseado no dataset [Kobe Bryant Shot Selectio](https://www.kaggle.com/c/kobe-bryant-shot-selection/overview), disponível no Kaggle.

```mermaid
flowchart
    subgraph data_ingestion[pipeline: data_ingestion]
        direction TB
        get_data[Coleta de dados: request na api do Github] -->
        save_data[Armazenamento dos dados]
        save_data --> raw_kobe_shots_dev_dataset[file: data/01_raw/dataset_kobe_dev.parquet]
        save_data --> raw_kobe_shots_prod_dataset[file: data/01_raw/dataset_kobe_prod.parquet]
    end
    subgraph data_processing[pipeline: data_processing]
        direction TB
        preprocess_kobe_shots[Pré-processamento de dados] --> split_data
        split_data[Separação treino e teste] --> save_preprocessed_data[Salvar dados]
        save_preprocessed_data --> intermediate_preprocessed_kobe_shots[file: data/02_intermediate/preprocessed_kobe_shots.parquet]
        save_preprocessed_data --> primary_data_filtered[file: data/03_primary/data_filtered.parquet]
        save_preprocessed_data --> primary_base_train[file: data/03_primary/base_train.parquet]
        save_preprocessed_data --> primary_base_test[file: data/03_primary/base_test.parquet]
    end
    subgraph data_science[pipeline: data_science]
        direction TB
        train_logistic_regression_model[Treinar modelo de regressão logística] --> save_metrics[Salvar métricas do modelo]
        train_decision_tree_model[Treinar modelo de árvore de decisão] --> save_metrics[Salvar métricas do modelo]
        save_metrics[Salvar métricas do modelo] --> save_model[Salvar modelo]
        save_model --> file_logistic_regression_model[mlflow artifact: logistic_regression_model]
        save_model --> file_decision_tree_model[mlflow artifact: decision_tree_model]
    end
    raw_kobe_shots_dev_dataset --> data_processing
    data_processing --> data_science
```
