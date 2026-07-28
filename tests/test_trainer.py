"""Testes unitários do módulo de treinamento."""

import numpy as np
import pytest

from src.models.trainer import build_classifier, compute_metrics


@pytest.fixture
def default_params() -> dict:
    """Parâmetros padrão do modelo para testes."""
    return {
        "n_estimators": 10,
        "max_depth": 5,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "class_weight": "balanced",
        "random_state": 42,
    }


def test_build_classifier_returns_fitted_model(default_params: dict) -> None:
    """Classificador deve treinar sem erros com dados simples."""
    X = np.random.rand(100, 5)  # noqa: N806
    y = np.random.randint(0, 2, 100)

    classifier = build_classifier(default_params)
    classifier.fit(X, y)

    predictions = classifier.predict(X)
    assert len(predictions) == len(y)


def test_compute_metrics_keys() -> None:
    """compute_metrics deve retornar todas as métricas esperadas."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.8])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    expected_keys = {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert expected_keys == set(metrics.keys())


def test_compute_metrics_range() -> None:
    """Todas as métricas devem estar entre 0 e 1."""
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.8])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    for name, value in metrics.items():
        assert 0.0 <= value <= 1.0, f"Métrica {name} fora do intervalo: {value}"