"""Módulo de configuração: lê variáveis de ambiente e centraliza constantes."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Configurações centralizadas da aplicação."""

    mlflow_tracking_uri: str
    mlflow_experiment_name: str
    model_name: str
    random_state: int
    test_size: float
    data_raw_path: str
    data_processed_path: str


def load_config() -> AppConfig:
    """Carrega e valida as configurações a partir das variáveis de ambiente.

    Returns:
        AppConfig com todos os parâmetros necessários para execução.

    Raises:
        ValueError: se alguma variável obrigatória estiver ausente.
    """
    required_vars = [
        "MLFLOW_TRACKING_URI",
        "MLFLOW_EXPERIMENT_NAME",
        "MODEL_NAME",
    ]

    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Variáveis de ambiente obrigatórias ausentes: {missing}")

    return AppConfig(
        mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
        mlflow_experiment_name=os.getenv(
            "MLFLOW_EXPERIMENT_NAME", "ecommerce-purchase-propensity"
        ),
        model_name=os.getenv("MODEL_NAME", "purchase-propensity-classifier"),
        random_state=int(os.getenv("RANDOM_STATE", "42")),
        test_size=float(os.getenv("TEST_SIZE", "0.2")),
        data_raw_path=os.getenv("DATA_RAW_PATH", "data/raw/online_shoppers.csv"),
        data_processed_path=os.getenv(
            "DATA_PROCESSED_PATH", "data/processed/dataset_processed.csv"
        ),
    )