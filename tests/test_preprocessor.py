"""Testes unitários do módulo de pré-processamento."""

import numpy as np
import pandas as pd
import pytest

from src.features.preprocessor import build_preprocessor, encode_target


@pytest.fixture
def sample_features() -> pd.DataFrame:
    """DataFrame com features mínimas para teste do preprocessor."""
    return pd.DataFrame({
        "Administrative": [1, 2, 3],
        "Administrative_Duration": [10.0, 20.0, 30.0],
        "Informational": [0, 1, 0],
        "Informational_Duration": [0.0, 5.0, 0.0],
        "ProductRelated": [5, 10, 15],
        "ProductRelated_Duration": [100.0, 200.0, 300.0],
        "BounceRates": [0.1, 0.2, 0.0],
        "ExitRates": [0.2, 0.3, 0.1],
        "PageValues": [10.0, 0.0, 5.0],
        "SpecialDay": [0.0, 0.4, 0.0],
        "Month": ["Feb", "Mar", "May"],
        "OperatingSystems": [1, 2, 3],
        "Browser": [1, 2, 1],
        "Region": [1, 3, 2],
        "TrafficType": [1, 2, 3],
        "VisitorType": ["Returning_Visitor", "New_Visitor", "Returning_Visitor"],
        "Weekend": [False, True, False],
    })


def test_build_preprocessor_output_shape(sample_features: pd.DataFrame) -> None:
    """Preprocessor deve retornar array com número de linhas correto."""
    preprocessor = build_preprocessor()
    result = preprocessor.fit_transform(sample_features)
    assert result.shape[0] == len(sample_features)


def test_build_preprocessor_no_nan(sample_features: pd.DataFrame) -> None:
    """Preprocessor não deve gerar valores nulos."""
    preprocessor = build_preprocessor()
    result = preprocessor.fit_transform(sample_features)
    assert not np.isnan(result).any()


def test_encode_target_binary() -> None:
    """encode_target deve retornar array com apenas 0 e 1."""
    y = pd.Series([True, False, True, False])
    result = encode_target(y)
    assert set(result).issubset({0, 1})


def test_encode_target_length() -> None:
    """encode_target deve preservar o tamanho da série."""
    y = pd.Series([True, False, True])
    result = encode_target(y)
    assert len(result) == len(y)