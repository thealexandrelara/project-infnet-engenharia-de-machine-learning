Projeto de machine learning com o objetivo de prever se Kobe Bryant acertou ou errou uma tentativa de arremesso, utilizando abordagens de classificação e regressão. O projeto é baseado no dataset [Kobe Bryant Shot Selectio](https://www.kaggle.com/c/kobe-bryant-shot-selection/overview), disponível no Kaggle.

# Diagrama

Abaixo encontra-se o diagrama contendo todas as etapas necessárias para este projeto que vai desde a pipeline de aquisição até a operação do modelo:

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
            preprocessed_kobe_shots{{output 'preprocessed_kobe_shots': data/02_intermediate/preprocessed_kobe_shots.parquet}}
            preprocessed_kobe_shots_prod{{output 'preprocessed_kobe_shots_prod': data/02_intermediate/preprocessed_kobe_shots.parquet}}
            model_input_table{{output 'model_input_table': data/03_primary/data_filtered.parquet}}
            base_train{{output 'base_train': data/03_primary/data_filtered.parquet}}
            base_test{{output 'base_test': data/03_primary/data_filtered.parquet}}
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

            logistic_regression_model_plots{{output 'logistic_regression_model_auc': data/08_reporting/logistic_regression_model_auc.png<br/><br/>output 'logistic_regression_model_confusion_matrix': data/08_reporting/logistic_regression_model_confusion_matrix.png<br/><br/>output 'logistic_regression_model_feature_importance': data/08_reporting/logistic_regression_model_feature_importance.png}}

            decision_tree_plots{{output 'decision_tree_auc': data/08_reporting/decision_tree_model_auc.png<br/><br/>output 'decision_tree_model_confusion_matrix': data/08_reporting/decision_tree_model_confusion_matrix.png<br/><br/>output 'decision_tree_model_feature_importance': data/08_reporting/decision_tree_model_feature_importance.png}}
        end
        train_logistic_regression_model_node --> logistic_regression_model
        train_logistic_regression_model_node --> logistic_regression_model_with_proba
        train_decision_tree_model_node --> decision_tree_model
        save_logistic_regression_model_plots_node --> logistic_regression_model_plots
        save_decision_tree_model_plots_node --> decision_tree_plots
    end
    subgraph serving_model[pipeline: serving_model]
        direction TB
        subgraph serving_model_nodes[nodes]
            direction LR
            predict_production_data_node[node: predict_production_data_node]
        end
        subgraph serving_model_outputs[outputs]
            direction LR
            production_data_predictions{{output 'production_data_predictions': data/08_reporting/production_data_predictions.parquet}}
        end
        predict_production_data_node --> production_data_predictions
    end
    data_ingestion --> data_processing
    data_processing --> data_science
    data_science --> serving_model
    data_science --> mlflow_server[ML Flow Models Serve]
    mlflow_server --> streamlit[Streamlit App]
```

### Como as ferramentas Streamlit, MLflow, PyCaret e Scikit-Learn auxiliam na construção dos pipelines?

Rastreamento de Experimentos (Experiment Tracking)

- MLflow é utilizado para na etapa de rastreamento dos experimentos, registrando hiperparâmetros, métricas (no caso deste projeto log_loss e f1_score são métricas sendo registradas), artefatos (como o modelo treinado, plots), versões das dependências, etc.
- A integração do kedro-mlflow foi utilizada no projeto facilitando o processo de registro dos experimentos por meio da execução das pipelines.

Funções de Treinamento

- PyCaret é uma ferramenta de AutoML que simplifica a criação, comparação, tuning e validação de múltiplos modelos de classificação.
- O PyCaret tem o Scikit-Learn como dependência que auxilia na parte do treinamento, preparação dos dados, pré e pós-processamento.
- Neste projeto, o PyCaret foi utilizado para separação de treino e teste (nos bastidores é utilizado o Scikit-learn) e também para a parte de treinamento do modelo, assim como obtenção das métricas que foram posteriormente salvas no MLFlow.

Monitoramento da Saúde do Modelo

- O monitoramento da saúde do modelo pode ser feito por meio da análise de Data Drift, Feature Drift e Concept Drift. Essas mudanças são identificadas através de comparações estatísticas entre dados históricos e novos () e pelo monitoramento contínuo das métricas de performance.
- Atualmente, estamos registrando as métricas utilizando o MLFlow, então a cada versão do modelo treinado podemos realizar a comparação dessas métricas. Caso a gente queira fazer uma análise mais detalhada de drift, precisaríamos fazer a coleta e armazenamento dos dados para podermos aplicar por exemplo testes de Kolmogorov-Smirnov ou Qui-quadrado.

Atualização de Modelo

- Ao rodar a pipeline `data_science` ocorre o treinamento do modelo e o armazenamento de uma nova versão do modelo treinado no MLFlow Model Registry.
- Uma vez disponibilizada a nova versão pelo MLFlow, podemos servir este modelo com `mlflow models serve -m models:/logistic-regression-model/latest -p 5001`
- Com o Streamlit podemos consumir o modelo. Como estamos passando `latest` ao servir o modelo, garantimos que sempre estaremos consumindo o último modelo disponível.

Provisionamento (Deployment)

- O modelo é versionado e registrado no MLflow Model Registry.
- Pode ser servido via API local com mlflow models serve ou embarcado diretamente na aplicação Streamlit, garantindo inferência direta.
- A interface desenvolvida em Streamlit permite interação com o modelo, visualização dos dados e resultados das previsões.
