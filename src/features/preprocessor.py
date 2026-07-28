"""Módulo de pré-processamento de features."""

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)

NUMERICAL_FEATURES = [
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
]

CATEGORICAL_FEATURES = [
    "Month",
    "VisitorType",
]

BINARY_FEATURES = [
    "Weekend",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
]


def build_preprocessor() -> ColumnTransformer:
    """Constrói o pipeline de pré-processamento de features.

    Returns:
        ColumnTransformer configurado com transformações por tipo de feature.
    """
    numerical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    binary_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    return ColumnTransformer(transformers=[
        ("numerical", numerical_pipeline, NUMERICAL_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ("binary", binary_pipeline, BINARY_FEATURES),
    ])


def encode_target(y: pd.Series) -> np.ndarray:
    """Converte a coluna alvo booleana em inteiro binário.

    Args:
        y: Series com valores booleanos ou inteiros.

    Returns:
        Array numpy com valores 0 e 1.
    """
    return y.astype(int).to_numpy()