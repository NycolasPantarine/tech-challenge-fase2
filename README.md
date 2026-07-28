# Tech Challenge — Fase 2
## Machine Learning Engineering — FIAP PosTech

**Aluno:** Nycolas Pantarine Ortiz Barbosa  
**Repositório:** [github.com/NycolasPantarine/tech-challenge-fase2](https://github.com/NycolasPantarine/tech-challenge-fase2)

---

## Problema de Negócio

Uma empresa de e-commerce precisa identificar a propensão de compra de usuários com base em seu comportamento de navegação. O objetivo é construir um pipeline completo de Machine Learning — desde o pré-processamento dos dados até o registro do modelo — seguindo padrões profissionais de engenharia.

---

## Dataset

**Online Shoppers Purchasing Intention Dataset**  
- Fonte: UCI Machine Learning Repository  
- 12.330 registros, 17 features, 1 target binário (`Revenue`)  
- Tarefa: classificação binária — o usuário vai ou não realizar uma compra?

---

## Stack Tecnológica

| Ferramenta | Uso |
|---|---|
| Python 3.11 | Linguagem principal |
| Poetry | Gerenciamento de dependências |
| Scikit-Learn | Pré-processamento e modelagem |
| MLflow | Tracking de experimentos e Model Registry |
| DVC | Versionamento de dados e pipeline reprodutível |
| Docker | Containerização do pipeline |
| Ruff | Linting e formatação |
| Pytest | Testes unitários |

---

## Estrutura do Projeto

```
tech-challenge-fase2/
├── src/
│   ├── data/
│   │   └── loader.py          # Carregamento e validação do dataset
│   ├── features/
│   │   └── preprocessor.py    # Pipeline de pré-processamento
│   ├── models/
│   │   └── trainer.py         # Treinamento, métricas e MLflow
│   ├── utils/
│   │   └── config.py          # Configurações via variáveis de ambiente
│   ├── preprocess.py          # Entrypoint do estágio preprocess
│   └── pipeline.py            # Entrypoint do estágio train
├── data/
│   └── raw/
│       └── online_shoppers.csv
├── tests/
│   ├── test_loader.py         # Testes do módulo de carregamento
│   ├── test_preprocessor.py   # Testes do pré-processamento
│   └── test_trainer.py        # Testes do treinamento
├── configs/
│   └── params.yaml            # Hiperparâmetros do modelo
├── dvc.yaml                   # Pipeline DVC (preprocess → train)
├── dvc.lock                   # Lock do pipeline DVC
├── Dockerfile                 # Container do pipeline
├── pyproject.toml             # Dependências e configuração do projeto
├── poetry.lock                # Lock file das dependências
├── .env.example               # Variáveis de ambiente necessárias
└── metrics.json               # Métricas do último treinamento
```

---

## Como Rodar o Projeto

### Pré-requisitos

- Python 3.11
- Poetry 2.x
- Docker Desktop (opcional)
- Git

---

### 1. Clonar o repositório

```bash
git clone https://github.com/NycolasPantarine/tech-challenge-fase2.git
cd tech-challenge-fase2
```

### 2. Instalar dependências

```bash
poetry install
```

### 3. Configurar variáveis de ambiente

```bash
cp .env.example .env
```

### 4. Rodar o pipeline completo via DVC

```bash
poetry run dvc repro
```

Esse comando executa automaticamente os dois estágios em sequência:

- **preprocess** — lê o dataset bruto, aplica transformações e salva os dados processados
- **train** — treina o modelo, loga métricas e parâmetros no MLflow e registra no Model Registry

### 5. Visualizar experimentos no MLflow

```bash
poetry run mlflow ui
```

Acesse [http://localhost:5000](http://localhost:5000) no navegador.

---

### Rodar via Docker

```bash
# Build da imagem
docker build -t tech-challenge-fase2 .

# Executar o pipeline
docker run --rm tech-challenge-fase2
```

---

### Rodar os testes

```bash
poetry run pytest
```

---

### Rodar o linter

```bash
poetry run ruff check src/
```

---

## Resultados do Modelo

| Métrica | Valor |
|---|---|
| Accuracy | 0.8642 |
| Precision | 0.5416 |
| Recall | 0.8010 |
| F1 Score | 0.6463 |
| ROC-AUC | **0.9245** |

**Modelo:** Random Forest Classifier  
**Hiperparâmetros:** n_estimators=100, max_depth=10, class_weight=balanced, random_state=42

---

## Pipeline DVC

```
data/raw/online_shoppers.csv
        │
        ▼
   [preprocess]
        │
        ▼
data/processed/dataset_processed.csv
        │
        ▼
     [train]
        │
        ▼
MLflow Model Registry
```

---

## Variáveis de Ambiente

| Variável | Descrição | Padrão |
|---|---|---|
| `MLFLOW_TRACKING_URI` | URI do MLflow | `sqlite:///mlflow.db` |
| `MLFLOW_EXPERIMENT_NAME` | Nome do experimento | `ecommerce-purchase-propensity` |
| `MODEL_NAME` | Nome do modelo no Registry | `purchase-propensity-classifier` |
| `RANDOM_STATE` | Seed para reprodutibilidade | `42` |
| `TEST_SIZE` | Proporção do conjunto de teste | `0.2` |
| `DATA_RAW_PATH` | Caminho do dataset bruto | `data/raw/online_shoppers.csv` |
| `DATA_PROCESSED_PATH` | Caminho dos dados processados | `data/processed/dataset_processed.csv` |