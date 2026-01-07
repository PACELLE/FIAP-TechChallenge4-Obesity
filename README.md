# 📊 FIAP Tech Challenge 4 — Obesity Prediction

## 🧠 Descrição do Projeto

Este projeto foi desenvolvido como parte do **Tech Challenge 4 da FIAP** e tem como objetivo aplicar técnicas de **Machine Learning** para análise e predição de **níveis de obesidade**, utilizando dados relacionados a características físicas, hábitos alimentares e estilo de vida.

Além da modelagem preditiva, o projeto também contempla a criação de um **dashboard interativo**, permitindo a visualização de insights relevantes sobre os dados e a realização de previsões a partir dos modelos treinados.

---

## 🎯 Objetivos

- Explorar e analisar dados relacionados à obesidade
- Desenvolver modelos de Machine Learning para classificação do nível de obesidade
- Avaliar o desempenho dos modelos por meio de métricas apropriadas
- Disponibilizar uma aplicação interativa para visualização e predição
- Consolidar os conhecimentos adquiridos ao longo da pós-graduação

---

## 🗂 Estrutura do Projeto
FIAP-TechChallenge4-Obesity/

├── data/ # Conjunto de dados utilizados

├── models/ # Modelos treinados e serializados

├── notebook_V1.ipynb # Análise exploratória e modelagem (versão 1)

├── notebook_V2.ipynb # Análise exploratória e modelagem (versão 2)

├── app_V2.py # Aplicação principal (dashboard / predição)

├── dicionario_obesity_fiap.pdf # Dicionário de dados

├── requirements.txt # Dependências do projeto

├── README.md # Documentação do projeto

└── .gitignore

---

## 📊 Dataset

O conjunto de dados contém informações demográficas, físicas e comportamentais, como:

- Idade
- Sexo
- Altura e peso
- Frequência de consumo alimentar
- Nível de atividade física
- Hábitos relacionados à saúde

As classes de saída representam diferentes **níveis de obesidade**, incluindo:

- Insufficient Weight
- Normal Weight
- Overweight Level I
- Overweight Level II
- Obesity Type I
- Obesity Type II
- Obesity Type III

Para mais detalhes sobre cada atributo, consulte o arquivo **`dicionario_obesity_fiap.pdf`**.

---

## 🤖 Modelagem e Machine Learning

O projeto utiliza algoritmos de Machine Learning para resolver um problema de **classificação multiclasse**, incluindo:

- Pré-processamento dos dados
- Análise exploratória (EDA)
- Treinamento de modelos
- Avaliação por métricas como acurácia e matriz de confusão
- Salvamento dos modelos treinados para uso em produção

Os experimentos e análises estão documentados nos notebooks Jupyter incluídos no repositório.

---

## 📈 Dashboard e Aplicação

A aplicação web permite:

- Visualizar dados de forma interativa
- Explorar estatísticas e distribuições
- Realizar previsões de obesidade a partir de novos dados
- Comparar resultados entre diferentes versões do modelo

A aplicação foi desenvolvida em Python, utilizando bibliotecas voltadas para ciência de dados e visualização.

---

## 🚀 Como Executar o Projeto

### 1️⃣ Clonar o Repositório

```bash
git clone https://github.com/PACELLE/FIAP-TechChallenge4-Obesity.git
cd FIAP-TechChallenge4-Obesity

### 2️⃣ Criar Ambiente Virtual (Recomendado)

python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows

