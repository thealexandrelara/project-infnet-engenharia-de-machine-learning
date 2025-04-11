# Avaliação por rúbricas

Aqui encontram-se todas as rúbricas do projeto e as respectivas justificativas de como elas estão sendo atendidas. A ideia é facilitar a avaliação de forma a facilmente analisar cada ponto exigido pelo projeto.

1. Coleta de Dados via APIs Públicas

- ✅ O aluno categorizou corretamente os dados?

  - A categorização dos dados foi feito na [seção Artefatos do README do projeto](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning#artefatos).

- ✅ O aluno integrou a leitura dos dados corretamente à sua solução?

  - No Kedro, foi criado uma pipeline chamada `data_ingestion` que é responsável por obter os dados a partir do Github, foram criados dois `nodes` para isso, que podem ser conferidos [aqui](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/blob/main/src/project_infnet_engenharia_de_machine_learning/pipelines/data_ingestion/nodes.py).

- ✅ O aluno aplicou o modelo em produção (servindo como API ou como solução embarcada)?

  - Conforme definido no README, após rodar as pipelines deve-se rodar o comando: `mlflow models serve -m models:/logistic-regression-model/latest -p 5001`.
  - Uma vez que o modelo estiver sendo servido pelo mlflow, deve-se rodar a aplicação Streamlit: `streamlit run streamlit/app.py`
  - Ao acessar a aplicação streamlit, é possível fazer a inferência do modelo clicando no botão de fazer a previsão do arremesso.

- ✅ O aluno indicou se o modelo é aderente a nova base de dados?
  - Isto foi detalhado na seção [Aderência do Modelo à base de produção](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/tree/main#ader%C3%AAncia-do-modelo-%C3%A0-base-de-produ%C3%A7%C3%A3o) localizado no README do projeto.

⸻

2. Criação de Pipeline de Streaming de Dados

- ✅ O aluno criou um repositório git com a estrutura de projeto baseado no Framework TDSP da Microsoft?

  - O link para o repositório git é: https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning.
  - Conforme combinado com o professor, fomos autorizados e orientados a utilizar o Kedro.

- ✅ O aluno criou um diagrama que mostra todas as etapas necessárias para a criação de modelos?

  - A pipeline com o diagrama do projeto se encontra no README, que pode ser acessada [clicando aqui](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/tree/main#diagrama).

- ✅ O aluno treinou um modelo de regressão usando PyCaret e MLflow?

  - O treinamento do modelo de regressão logística foi feito na pipeline `data_science` utilizando o node `train_logistic_regression_model`, pode-se conferir [clicando aqui](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/blob/main/src/project_infnet_engenharia_de_machine_learning/pipelines/data_science/nodes.py#L59-L89).
  - No catalog.yml do Kedro, é possível conferir que foi utilizado a implementação do kedro-mlflow para registrar o modelo no MLFlow, para conferir, [clique aqui](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/blob/main/conf/base/catalog.yml#L55C1-L72C44).

- ✅ O aluno calculou o Log Loss para o modelo de regressão e registrou no mlflow?

  - Sim, no node `train_logistic_regression_model` foi calculado o log_loss e salvo no MLFlow, como pode ser visto [aqui](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/blob/e923e17827432c21a0a3c4269479f5b1e4dc0be2/src/project_infnet_engenharia_de_machine_learning/pipelines/data_science/nodes.py#L83).

- ✅ O aluno treinou um modelo de árvore de decisao usando PyCaret e MLflow?

  - O treinamento do modelo de árvore de decisão foi feito na pipeline `data_science` utilizando o node `train_decision_tree_model`, pode-se conferir [clicando aqui](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/blob/main/src/project_infnet_engenharia_de_machine_learning/pipelines/data_science/nodes.py#L91-L118).
  - No catalog.yml do Kedro, é possível conferir que foi utilizado a implementação do kedro-mlflow para registrar o modelo no MLFlow, para conferir, [clique aqui](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/blob/main/conf/base/catalog.yml#L101-L106).

- ✅ O aluno calculou o Log Loss e F1 Score para o modelo de árvore de decisão e registrou no mlflow?
  - Sim, no node `train_decision_tree_model` foi calculado o f1 score e log_loss e salvo no MLFlow, como pode ser visto [aqui](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/blob/main/src/project_infnet_engenharia_de_machine_learning/pipelines/data_science/nodes.py#L108-L112).

⸻

3. Preparar um modelo previamente treinado para uma solução de streaming de dados

- ✅ O aluno indicou o objetivo e descreveu detalhadamente cada artefato criado no projeto?

  - Sim, encontram-se devidamente detalhados na [seção de Artefatos do README](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/tree/main#artefatos).

- ✅ O aluno cobriu todos os artefatos do diagrama proposto?

  - Sim, pode ser conferido na [seção de Artefatos do README](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/tree/main#artefatos).

- ✅ O aluno usou o MLFlow para registrar a rodada "Preparação de Dados" com as métricas e argumentos relevantes?

  - Sim, para isso basta executar o comando `kedro run --pipeline data_processing` e conferir no MLFlow, conforme descrito no README do projeto.

- ✅ O aluno removeu os dados faltantes da base?

  - Sim, pode ser conferido no node `create_model_input_table` da pipeline `data_processing`, [clicando aqui](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/blob/e923e17827432c21a0a3c4269479f5b1e4dc0be2/src/project_infnet_engenharia_de_machine_learning/pipelines/data_processing/nodes.py#L55).

- ✅ O aluno selecionou as colunas indicadas para criar o modelo?

  - Sim, pode ser conferido no node `create_model_input_table` da pipeline `data_processing`, [clicando aqui](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/blob/e923e17827432c21a0a3c4269479f5b1e4dc0be2/src/project_infnet_engenharia_de_machine_learning/pipelines/data_processing/nodes.py#L26).

- ✅ O aluno indicou quais as dimensões para a base preprocessada?

  - Sim, pode ser conferido na seção de Artefatos, na subseção [Camada primary do README](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/tree/main?tab=readme-ov-file#camada-primary--dados-prontos-para-treinoteste). A dimensão destacada lá é de 20285 registros (linhas) e 7 colunas

- ✅ O aluno criou arquivos para cada fase do processamento e os armazenou nas pastas indicadas?

  - Sim, conforme configuração do catálogo é possível notar que foi salvo nas pastas corretas do Kedro, [clique aqui](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/blob/main/conf/base/catalog.yml#L34-L53) para conferir.

- ✅ O aluno separou em duas bases, uma para treino e outra para teste?

  - Sim, pode ser visto no catalog.yml do Kedro, [clicando aqui](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/blob/main/conf/base/catalog.yml#L41-L53).

- ✅ O aluno criou um pipeline chamado "Treinamento" no MlFlow?
  - Sim, para isso basta executar o comando `kedro run --pipeline data_science` e conferir no MLFlow, conforme descrito no README do projeto.

⸻

4. Atualização e Monitoramento do Modelo em Produção

- ✅ Diferença entre base de desenvolvimento e produção identificada?

  - Sim, conforme foi destacado na [seção Aderência do modelo à base de produção](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/tree/main?tab=readme-ov-file#ader%C3%AAncia-do-modelo-%C3%A0-base-de-produ%C3%A7%C3%A3o) no README do projeto.

- ✅ Monitoramento do modelo com/sem variável alvo descrito?

  - Sim, conforme a seção [Monitoramento de saúde do modelo com e sem a disponibilidade da variável resposta](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/tree/main?tab=readme-ov-file#monitoramento-de-sa%C3%BAde-do-modelo-com-e-sem-a-disponibilidade-da-vari%C3%A1vel-resposta) no README do projeto.

- ✅ Dashboard de monitoramento com Streamlit implementado?

  - Sim, para o correto funcionamento, sirva o modelo pelo MLFlow:

  ```
    mlflow models serve -m models:/logistic-regression-model/latest -p 5001
  ```

  E execute a aplicação Streamlit:

  ```
  streamlit run streamlit/app.py
  ```

  Com isso, será possível fazer a inferência do modelo.

- ✅ Estratégias reativa e preditiva de re-treinamento descritas?
  - Sim, conforme a seção [Estratégias de retreinamento](https://github.com/thealexandrelara/project-infnet-engenharia-de-machine-learning/tree/main?tab=readme-ov-file#estrat%C3%A9gias-de-retreinamento) do README do projeto.
