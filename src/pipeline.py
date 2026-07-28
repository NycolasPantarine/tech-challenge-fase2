"""Pipeline principal: orquestra treinamento e registro do modelo."""

import json
import logging

import pandas as pd
import yaml

from src.data.loader import get_feature_target_split
from src.models.trainer import train_and_register
from src.utils.config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_params(params_path: str = "configs/params.yaml") -> dict:
    """Carrega os hiperparâmetros do arquivo de configuração.

    Args:
        params_path: caminho para o arquivo params.yaml.

    Returns:
        Dicionário com os parâmetros do modelo.
    """
    with open(params_path) as f:
        return yaml.safe_load(f)


def save_metrics(metrics: dict[str, float], output_path: str = "metrics.json") -> None:
    """Salva as métricas do modelo em arquivo JSON para o DVC.

    Args:
        metrics: dicionário com métricas calculadas.
        output_path: caminho do arquivo de saída.
    """
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Métricas salvas em: %s", output_path)


def run_pipeline() -> None:
    """Executa o pipeline de treinamento de ponta a ponta."""
    logger.info("Iniciando pipeline de propensão de compra")

    config = load_config()
    params = load_params()

    logger.info("Carregando dados processados")
    df = pd.read_csv(config.data_processed_path)

    X, y = get_feature_target_split(df, params["data"]["target_column"])  # noqa: N806

    logger.info("Iniciando treinamento e tracking com MLflow")
    run_id, metrics = train_and_register(X, y, params["model"], config)

    save_metrics(metrics)
    logger.info("Pipeline concluído com sucesso | run_id: %s", run_id)


if __name__ == "__main__":
    run_pipeline()