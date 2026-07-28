"""Módulo de carregamento e validação do dataset."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "PageValues",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
    "Revenue",
}


def load_dataset(file_path: str) -> pd.DataFrame:
    """Carrega o dataset a partir de um arquivo CSV.

    Args:
        file_path: caminho para o arquivo CSV.

    Returns:
        DataFrame com os dados carregados.

    Raises:
        FileNotFoundError: se o arquivo não existir.
        ValueError: se colunas obrigatórias estiverem ausentes.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Dataset não encontrado: {file_path}")

    logger.info("Carregando dataset: %s", file_path)
    df = pd.read_csv(path)

    _validate_columns(df)
    logger.info("Dataset carregado: %d linhas, %d colunas", len(df), len(df.columns))

    return df


def _validate_columns(df: pd.DataFrame) -> None:
    """Verifica se todas as colunas obrigatórias estão presentes.

    Args:
        df: DataFrame a ser validado.

    Raises:
        ValueError: se colunas obrigatórias estiverem ausentes.
    """
    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        raise ValueError(f"Colunas ausentes no dataset: {missing_columns}")


def get_feature_target_split(
    df: pd.DataFrame, target_column: str
) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features e target do DataFrame.

    Args:
        df: DataFrame completo.
        target_column: nome da coluna alvo.

    Returns:
        Tupla (X, y) com features e target.
    """
    if target_column not in df.columns:
        raise ValueError(f"Coluna alvo '{target_column}' não encontrada no dataset.")

    X = df.drop(columns=[target_column])  # noqa: N806
    y = df[target_column].astype(int)

    return X, y