"""Etapa de pré-processamento: lê dados brutos e salva dados processados."""

import logging
import os
from pathlib import Path

import pandas as pd
import yaml

from src.data.loader import get_feature_target_split, load_dataset
from src.features.preprocessor import build_preprocessor, encode_target

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger(__name__)


def run_preprocessing() -> None:
    """Executa o pré-processamento e salva o resultado em disco."""
    with open("configs/params.yaml") as f:
        params = yaml.safe_load(f)

    data_raw_path = os.getenv("DATA_RAW_PATH", "data/raw/online_shoppers.csv")
    data_processed_path = os.getenv(
        "DATA_PROCESSED_PATH", "data/processed/dataset_processed.csv"
    )

    logger.info("Carregando dataset bruto")
    df = load_dataset(data_raw_path)

    X, y = get_feature_target_split(df, params["data"]["target_column"])  # noqa: N806

    preprocessor = build_preprocessor()
    X_processed = preprocessor.fit_transform(X)  # noqa: N806

    feature_names = preprocessor.get_feature_names_out()
    df_processed = pd.DataFrame(X_processed, columns=feature_names)
    df_processed["Revenue"] = encode_target(y)

    Path(data_processed_path).parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(data_processed_path, index=False)

    logger.info("Dados processados salvos em: %s", data_processed_path)


if __name__ == "__main__":
    run_preprocessing()