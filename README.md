Projeto de machine learning com o objetivo de prever se Kobe Bryant acertou ou errou uma tentativa de arremesso, utilizando abordagens de classificação e regressão. O projeto é baseado no dataset [Kobe Bryant Shot Selectio](https://www.kaggle.com/c/kobe-bryant-shot-selection/overview), disponível no Kaggle.

```mermaid
flowchart
    subgraph data_ingestion[pipeline: data_ingestion]
        direction TB
        subgraph data_ingestion_nodes[nodes]
            direction LR
            download_kobe_shots_dev_dataset[node: download_kobe_shots_dev_dataset]
            download_kobe_shots_prod_dataset[node: download_kobe_shots_prod_dataset]
        end
        subgraph data_ingestion_outputs[outputs]
            direction LR
            raw_kobe_shots_dev_dataset{{output 'raw_kobe_shots_dev': data/01_raw/dataset_kobe_dev.parquet}}
            raw_kobe_shots_prod_dataset{{output 'raw_kobe_shots_prod': data/01_raw/dataset_kobe_prod.parquet}}
        end
        download_kobe_shots_dev_dataset --> raw_kobe_shots_dev_dataset
        download_kobe_shots_prod_dataset --> raw_kobe_shots_prod_dataset
    end
    subgraph data_processing[pipeline: data_processing]
        direction TB
        subgraph data_processing_nodes[nodes]
            direction LR
            preprocess_kobe_shots_dev_node[node: preprocess_kobe_shots]
            preprocess_kobe_shots_prod_node[node: preprocess_kobe_shots_prod]
            create_model_input_table_node[node: create_model_input_table_node]
            split_data_node[node: split_data_node]
        end
        subgraph data_processing_outputs[outputs]
            direction LR
            preprocessed_kobe_shots[output 'preprocessed_kobe_shots': data/02_intermediate/preprocessed_kobe_shots.parquet]
            preprocessed_kobe_shots_prod[output 'preprocessed_kobe_shots_prod': data/02_intermediate/preprocessed_kobe_shots.parquet]
            model_input_table[output 'model_input_table': data/03_primary/data_filtered.parquet]
            base_train[output 'base_train': data/03_primary/data_filtered.parquet]
            base_test[output 'base_test': data/03_primary/data_filtered.parquet]
        end
        preprocess_kobe_shots_dev_node --> preprocessed_kobe_shots
        preprocess_kobe_shots_prod_node --> preprocessed_kobe_shots_prod
        create_model_input_table_node --> model_input_table
        split_data_node --> base_train
        split_data_node --> base_test
    end
    subgraph data_science[pipeline: data_science]
        direction TB
        subgraph data_science_nodes[nodes]
            direction LR
            train_logistic_regression_model_node[node: train_logistic_regression_model_node]
            train_decision_tree_model_node[node: train_decision_tree_model_node]
            save_logistic_regression_model_plots_node[node: save_logistic_regression_model_plots_node]
            save_decision_tree_model_plots_node[node: save_decision_tree_model_plots_node]
        end
        subgraph data_science_outputs[outputs]
            direction LR
            logistic_regression_model[output 'logistic_regression_model': logistic-regression-model]
            logistic_regression_model_with_proba[output 'logistic_regression_model_with_proba': logistic-regression-model-dev]
            decision_tree_model[output 'decision_tree_model': decision-tree-model]
            logistic_regression_model_auc[output 'logistic_regression_model_auc': data/08_reporting/logistic_regression_model_auc.png]
            logistic_regression_model_confusion_matrix[output 'logistic_regression_model_confusion_matrix': data/08_reporting/logistic_regression_model_confusion_matrix.png]
            logistic_regression_model_feature_importance[output 'logistic_regression_model_feature_importance': data/08_reporting/logistic_regression_model_feature_importance.png]
            decision_tree_auc[output 'decision_tree_auc': data/08_reporting/decision_tree_model_auc.png]
            decision_tree_model_confusion_matrix[output 'decision_tree_model_confusion_matrix': data/08_reporting/decision_tree_model_confusion_matrix.png]
            decision_tree_model_feature_importance[output 'decision_tree_model_feature_importance': data/08_reporting/decision_tree_model_feature_importance.png]
        end
        train_logistic_regression_model_node --> logistic_regression_model
        train_logistic_regression_model_node --> logistic_regression_model_with_proba
        train_decision_tree_model_node --> decision_tree_model
        save_logistic_regression_model_plots_node --> logistic_regression_model_auc
        save_logistic_regression_model_plots_node --> logistic_regression_model_confusion_matrix
        save_logistic_regression_model_plots_node --> logistic_regression_model_feature_importance
        save_decision_tree_model_plots_node --> decision_tree_auc
        save_decision_tree_model_plots_node --> decision_tree_model_confusion_matrix
        save_decision_tree_model_plots_node --> decision_tree_model_feature_importance
    end
    subgraph data_science[pipeline: data_science]
        direction TB
        subgraph data_science_nodes[nodes]
            direction LR
            train_logistic_regression_model_node[node: train_logistic_regression_model_node]
            train_decision_tree_model_node[node: train_decision_tree_model_node]
            save_logistic_regression_model_plots_node[node: save_logistic_regression_model_plots_node]
            save_decision_tree_model_plots_node[node: save_decision_tree_model_plots_node]
        end
        subgraph data_science_outputs[outputs]
            direction LR
            logistic_regression_model[output 'logistic_regression_model': logistic-regression-model]
            logistic_regression_model_with_proba[output 'logistic_regression_model_with_proba': logistic-regression-model-dev]
            decision_tree_model[output 'decision_tree_model': decision-tree-model]
            logistic_regression_model_auc[output 'logistic_regression_model_auc': data/08_reporting/logistic_regression_model_auc.png]
            logistic_regression_model_confusion_matrix[output 'logistic_regression_model_confusion_matrix': data/08_reporting/logistic_regression_model_confusion_matrix.png]
            logistic_regression_model_feature_importance[output 'logistic_regression_model_feature_importance': data/08_reporting/logistic_regression_model_feature_importance.png]
            decision_tree_auc[output 'decision_tree_auc': data/08_reporting/decision_tree_model_auc.png]
            decision_tree_model_confusion_matrix[output 'decision_tree_model_confusion_matrix': data/08_reporting/decision_tree_model_confusion_matrix.png]
            decision_tree_model_feature_importance[output 'decision_tree_model_feature_importance': data/08_reporting/decision_tree_model_feature_importance.png]
        end
        train_logistic_regression_model_node --> logistic_regression_model
        train_logistic_regression_model_node --> logistic_regression_model_with_proba
        train_decision_tree_model_node --> decision_tree_model
        save_logistic_regression_model_plots_node --> logistic_regression_model_auc
        save_logistic_regression_model_plots_node --> logistic_regression_model_confusion_matrix
        save_logistic_regression_model_plots_node --> logistic_regression_model_feature_importance
        save_decision_tree_model_plots_node --> decision_tree_auc
        save_decision_tree_model_plots_node --> decision_tree_model_confusion_matrix
        save_decision_tree_model_plots_node --> decision_tree_model_feature_importance
    end
    data_ingestion --> data_processing
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
