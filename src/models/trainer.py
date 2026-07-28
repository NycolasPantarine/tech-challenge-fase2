"""Módulo de treinamento, avaliação e registro do modelo no MLflow."""

import logging
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


def build_classifier(params: dict[str, Any]) -> RandomForestClassifier:
    """Constrói o classificador com os hiperparâmetros fornecidos.

    Args:
        params: dicionário com hiperparâmetros do modelo.

    Returns:
        RandomForestClassifier configurado.
    """
    return RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        min_samples_leaf=params["min_samples_leaf"],
        class_weight=params["class_weight"],
        random_state=params["random_state"],
        n_jobs=-1,
    )


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    """Calcula as métricas de avaliação do modelo.

    Args:
        y_true: valores reais.
        y_pred: valores preditos.
        y_proba: probabilidades preditas para a classe positiva.

    Returns:
        Dicionário com as métricas calculadas.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def train_and_register(
    X: Any,  # noqa: N803
    y: np.ndarray,
    params: dict[str, Any],
    config: Any,
) -> tuple[str, dict[str, float]]:
    """Treina o modelo, loga no MLflow e registra no Model Registry.

    Args:
        X: features já pré-processadas.
        y: target binário.
        params: hiperparâmetros do modelo.
        config: AppConfig com configurações da aplicação.

    Returns:
        Tupla com run_id e dicionário de métricas.
    """
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(config.mlflow_experiment_name)

    X_train, X_test, y_train, y_test = train_test_split(  # noqa: N806
        X, y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    with mlflow.start_run() as run:
        mlflow.log_params(params)

        classifier = build_classifier(params)
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)[:, 1]

        metrics = compute_metrics(y_test, y_pred, y_proba)
        mlflow.log_metrics(metrics)
        _log_metrics(metrics)

        mlflow.sklearn.log_model(
            sk_model=classifier,
            name="model",
            registered_model_name=config.model_name,
        )

        logger.info("Modelo registrado no MLflow Registry: %s", config.model_name)

        return run.info.run_id, metrics


def _log_metrics(metrics: dict[str, float]) -> None:
    """Loga as métricas no logger da aplicação.

    Args:
        metrics: dicionário com métricas calculadas.
    """
    for name, value in metrics.items():
        logger.info("  %s: %.4f", name, value)