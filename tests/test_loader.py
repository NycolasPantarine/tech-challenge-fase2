"""Testes unitários do módulo de carregamento de dados."""

import pandas as pd
import pytest

from src.data.loader import get_feature_target_split, load_dataset


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    """DataFrame mínimo com todas as colunas obrigatórias."""
    return pd.DataFrame({
        "Administrative": [1, 2],
        "Administrative_Duration": [10.0, 20.0],
        "Informational": [0, 1],
        "Informational_Duration": [0.0, 5.0],
        "ProductRelated": [5, 10],
        "ProductRelated_Duration": [100.0, 200.0],
        "BounceRates": [0.1, 0.2],
        "ExitRates": [0.2, 0.3],
        "PageValues": [10.0, 0.0],
        "SpecialDay": [0.0, 0.4],
        "Month": ["Feb", "Mar"],
        "OperatingSystems": [1, 2],
        "Browser": [1, 2],
        "Region": [1, 3],
        "TrafficType": [1, 2],
        "VisitorType": ["Returning_Visitor", "New_Visitor"],
        "Weekend": [False, True],
        "Revenue": [False, True],
    })


def test_load_dataset_file_not_found() -> None:
    """Deve lançar FileNotFoundError para caminho inexistente."""
    with pytest.raises(FileNotFoundError):
        load_dataset("caminho/inexistente.csv")


def test_get_feature_target_split_shape(sample_dataframe: pd.DataFrame) -> None:
    """X deve ter uma coluna a menos que o DataFrame original."""
    X, y = get_feature_target_split(sample_dataframe, "Revenue")  # noqa: N806
    assert X.shape[1] == sample_dataframe.shape[1] - 1
    assert len(y) == len(sample_dataframe)


def test_get_feature_target_split_target_values(sample_dataframe: pd.DataFrame) -> None:
    """Target deve conter apenas valores 0 e 1."""
    _, y = get_feature_target_split(sample_dataframe, "Revenue")
    assert set(y.unique()).issubset({0, 1})


def test_get_feature_target_split_invalid_column(sample_dataframe: pd.DataFrame) -> None:
    """Deve lançar ValueError para coluna alvo inexistente."""
    with pytest.raises(ValueError, match="não encontrada"):
        get_feature_target_split(sample_dataframe, "coluna_inexistente")