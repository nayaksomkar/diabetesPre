"""
Visualization functions for model performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from config import FIGURE_SIZE, FIGURE_DPI, CONFUSION_MATRIX_LABELS


def plot_accuracy_metrics(metrics: dict, save_path: str = "accuracy_metrics.png"):
    """
    Plot accuracy and precision/recall/F1 metrics.

    Args:
        metrics: Dictionary containing evaluation metrics.
        save_path: Path to save the plot.
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    metric_names = [
        "Accuracy",
        "Precision (Macro)",
        "Recall (Macro)",
        "F1 (Macro)",
        "Precision (Weighted)",
        "Recall (Weighted)",
        "F1 (Weighted)",
    ]
    metric_values = [
        metrics["accuracy"],
        metrics["precision_macro"],
        metrics["recall_macro"],
        metrics["f1_macro"],
        metrics["precision_weighted"],
        metrics["recall_weighted"],
        metrics["f1_weighted"],
    ]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(metric_names)))
    bars = ax.barh(metric_names, metric_values, color=colors)

    ax.set_xlim(0, 1)
    ax.set_xlabel("Score", fontsize=12)
    ax.set_title("Model Performance Metrics (100 Test Samples)", fontsize=14, fontweight="bold")
    ax.axvline(x=0.85, color="red", linestyle="--", label="Minimum Threshold (0.85)")
    ax.legend()

    for bar, value in zip(bars, metric_values):
        ax.text(
            value + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.4f}",
            va="center",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Accuracy metrics plot saved to: {save_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "test_confusion_matrix.png",
    label_names: list = None,
):
    """
    Plot confusion matrix heatmap.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        save_path: Path to save the plot.
        label_names: List of class names for labeling.
    """
    cm = confusion_matrix(y_true, y_pred)
    n_classes = cm.shape[0]

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)

    if label_names is None:
        label_names = CONFUSION_MATRIX_LABELS

    ax.set(
        xticks=np.arange(n_classes),
        yticks=np.arange(n_classes),
        xticklabels=label_names[:n_classes],
        yticklabels=label_names[:n_classes],
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), fontsize=8)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=10,
            )

    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.title("Test Set - Confusion Matrix (100 Samples)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Confusion matrix plot saved to: {save_path}")


def plot_metrics_comparison(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "metrics_comparison.png",
    label_names: list = None,
):
    """
    Plot precision, recall, and F1 scores per class.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        save_path: Path to save the plot.
        label_names: List of class names for labeling.
    """
    unique_labels = np.unique(y_true)
    n_classes = len(unique_labels)

    precision_per_class = []
    recall_per_class = []
    f1_per_class = []

    for i in unique_labels:
        precision_per_class.append(
            precision_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)
        )
        recall_per_class.append(
            recall_score(y_true, y_pred, labels=[i], average="micro", zero_division=0)
        )
        f1_per_class.append(f1_score(y_true, y_pred, labels=[i], average="micro", zero_division=0))

    if label_names is None:
        class_labels = [f"Class {i}" for i in range(n_classes)]
    else:
        class_labels = label_names[:n_classes]

    x = np.arange(n_classes)
    width = 0.25

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)
    ax.bar(x - width, precision_per_class, width, label="Precision", color="steelblue")
    ax.bar(x, recall_per_class, width, label="Recall", color="darkorange")
    ax.bar(x + width, f1_per_class, width, label="F1 Score", color="seagreen")

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Performance Metrics (100 Test Samples)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.85, color="red", linestyle="--", alpha=0.7, label="Threshold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Metrics comparison plot saved to: {save_path}")
