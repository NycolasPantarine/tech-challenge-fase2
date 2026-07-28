"""Pipeline principal: orquestra carregamento, pré-processamento e treinamento."""

import logging

import yaml

from src.data.loader import get_feature_target_split, load_dataset
from src.features.preprocessor import encode_target
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
        params = yaml.safe_load(f)
    return params


def run_pipeline() -> None:
    """Executa o pipeline completo de ML de ponta a ponta."""
    logger.info("Iniciando pipeline de propensão de compra")

    config = load_config()
    params = load_params()

    logger.info("Carregando dataset")
    df = load_dataset(config.data_raw_path)

    logger.info("Separando features e target")
    X, y = get_feature_target_split(df, params["data"]["target_column"])  # noqa: N806
    y = encode_target(y)

    logger.info("Iniciando treinamento e tracking com MLflow")
    run_id = train_and_register(X, y, params["model"], config)

    logger.info("Pipeline concluído com sucesso | run_id: %s", run_id)


if __name__ == "__main__":
    run_pipeline()