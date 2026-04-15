"""
Test dataset creation and model testing functions.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from MLtrain import DiabetesModelTrainer


def create_test_dataset(
    X: pd.DataFrame, y: pd.Series, n_samples: int = 100, random_state: int = 42
) -> tuple:
    """
    Create a test dataset with randomly picked samples from the full dataset.

    Args:
        X: Feature dataframe.
        y: Target series.
        n_samples: Number of samples to randomly pick.
        random_state: Random seed for reproducibility.

    Returns:
        tuple: (X_test, y_test)
    """
    if n_samples > len(X):
        n_samples = len(X)

    indices = np.random.RandomState(random_state).choice(len(X), size=n_samples, replace=False)
    X_test = X.iloc[indices].reset_index(drop=True)
    y_test = y.iloc[indices].reset_index(drop=True)

    return X_test, y_test


def test_model_with_inputs(
    trainer: DiabetesModelTrainer, X_test: pd.DataFrame, y_test: pd.Series
) -> dict:
    """
    Test the model with test inputs and evaluate performance.

    Args:
        trainer: Trained DiabetesModelTrainer instance.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        dict: Dictionary containing predictions and metrics.
    """
    predictions = trainer.model.predict(X_test)

    metrics = {
        "predictions": predictions,
        "true_labels": y_test.values,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "f1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "precision_weighted": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(y_test, predictions, average="weighted", zero_division=0),
        "f1_weighted": f1_score(y_test, predictions, average="weighted", zero_division=0),
    }

    return metrics
