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

### Como as ferramentas Streamlit, MLflow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines?

🧪 Rastreamento de Experimentos (Experiment Tracking)

- MLflow é utilizado para rastrear cada execução (run) dos experimentos, registrando hiperparâmetros, métricas (como log_loss e f1_score), artefatos (como o modelo treinado) e versões dos dados e código utilizados.
- O Kedro facilita a integração desses experimentos em pipelines reprodutíveis e versionados.

⚙️ Funções de Treinamento

- PyCaret simplifica a criação, comparação, tuning e validação de múltiplos modelos de classificação com poucas linhas de código.
- Scikit-Learn é utilizado nos bastidores do PyCaret e também diretamente para etapas de pré-processamento personalizadas ou no pós-processamento.
- As funções de treinamento são integradas ao pipeline do Kedro, que garante modularidade e organização.

📈 Monitoramento da Saúde do Modelo

- Utilizamos MLflow para registrar métricas de desempenho que ajudam a monitorar a saúde do modelo ao longo do tempo.
- Em produção, a saúde pode ser acompanhada via métricas como log_loss, f1_score, proporção de classes previstas, entre outras, comparadas com as dos dados de treino.

🔁 Atualização de Modelo

- O pipeline foi estruturado para permitir reexecuções programadas ou sob demanda (reativo ou preditivo).
- Alterações no conjunto de dados de produção podem ser detectadas com data drift e model drift, sinalizando a necessidade de retreinamento.
- O modelo pode ser atualizado automaticamente com novos dados rotulados, com logs completos sendo mantidos pelo MLflow.

🚀 Provisionamento (Deployment)

- O modelo é versionado e registrado no MLflow Model Registry.
- Pode ser servido via API local com mlflow models serve ou embarcado diretamente na aplicação Streamlit, garantindo inferência direta.
- A interface desenvolvida em Streamlit permite interação com o modelo, visualização dos dados e resultados das previsões.
