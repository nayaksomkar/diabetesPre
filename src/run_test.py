"""
Main execution file for model testing.
Run this file to test the model with 100 random samples.
"""

import json
from test_data import load_test_data
from test_testing import create_test_dataset, test_model_with_inputs
from test_plots import plot_accuracy_metrics, plot_confusion_matrix, plot_metrics_comparison
from MLtrain import DiabetesModelTrainer
from config import RANDOM_FOREST_PARAMS


def save_test_logs(metrics: dict, n_samples: int, model_params: dict):
    """
    Save test results to a JSON log file.

    Args:
        metrics: Dictionary containing evaluation metrics.
        n_samples: Number of samples tested.
        model_params: Model hyperparameters used.
    """
    log_data = {
        "test_info": {
            "n_samples_tested": n_samples,
            "model_type": "RandomForestClassifier",
            "model_params": model_params,
        },
        "metrics": {
            "accuracy": float(metrics["accuracy"]),
            "precision_macro": float(metrics["precision_macro"]),
            "recall_macro": float(metrics["recall_macro"]),
            "f1_macro": float(metrics["f1_macro"]),
            "precision_weighted": float(metrics["precision_weighted"]),
            "recall_weighted": float(metrics["recall_weighted"]),
            "f1_weighted": float(metrics["f1_weighted"]),
        },
        "summary": {
            "accuracy_percentage": f"{metrics['accuracy'] * 100:.2f}%",
            "total_correct": int(metrics["accuracy"] * n_samples),
            "total_incorrect": int((1 - metrics["accuracy"]) * n_samples),
        },
    }

    with open("test_logs.json", "w") as f:
        json.dump(log_data, f, indent=4)

    print("[INFO] Test logs saved to test_logs.json")


def run_model_tests(n_samples: int = 100, random_state: int = 42):
    """
    Run complete model testing pipeline with specified number of samples.

    Args:
        n_samples: Number of random samples to test with.
        random_state: Random seed for reproducibility.
    """
    print("\n" + "=" * 60)
    print("   DIABETES PREDICTION SYSTEM - MODEL TESTING")
    print("=" * 60 + "\n")

    print("[STEP 1/6] Loading and preprocessing data...")
    X, y, processor = load_test_data()
    print(f"   - Features shape: {X.shape}")
    print(f"   - Target distribution:\n{y.value_counts()}\n")

    print(f"[STEP 2/6] Creating test dataset with {n_samples} random samples...")
    X_test, y_test = create_test_dataset(X, y, n_samples=n_samples, random_state=random_state)
    print(f"   - Test set size: {len(X_test)}")

    print("[STEP 3/6] Training model on full dataset...")
    trainer = DiabetesModelTrainer()
    trainer.split_data(X, y)
    trainer.train(trainer.X_train, trainer.y_train)
    print("   - Model training completed.")

    print(f"[STEP 4/6] Testing model on {n_samples} random samples...")
    metrics = test_model_with_inputs(trainer, X_test, y_test)

    print("\n" + "=" * 40)
    print("[TEST RESULTS]")
    print("=" * 40)
    print(f"   Total samples tested: {n_samples}")
    print(f"   Accuracy:            {metrics['accuracy']:.4f} ({metrics['accuracy'] * 100:.2f}%)")
    print(f"   Precision (Macro):  {metrics['precision_macro']:.4f}")
    print(f"   Recall (Macro):    {metrics['recall_macro']:.4f}")
    print(f"   F1 Score (Macro):  {metrics['f1_macro']:.4f}")
    print(f"   Precision (Weighted): {metrics['precision_weighted']:.4f}")
    print(f"   Recall (Weighted):    {metrics['recall_weighted']:.4f}")
    print(f"   F1 Score (Weighted):  {metrics['f1_weighted']:.4f}")
    print(f"   Correct predictions: {int(metrics['accuracy'] * n_samples)}")
    print(f"   Incorrect predictions: {int((1 - metrics['accuracy']) * n_samples)}")

    print("\n[CLASSIFICATION REPORT]")
    target_names = (
        sorted(y_test.unique().tolist()) if hasattr(y_test, "unique") else list(set(y_test))
    )
    from sklearn.metrics import classification_report

    print(
        classification_report(
            metrics["true_labels"],
            metrics["predictions"],
            target_names=target_names,
            zero_division=0.0,
        )
    )

    print("[STEP 5/6] Generating performance graphs...")
    target_names = (
        sorted(y_test.unique().tolist()) if hasattr(y_test, "unique") else list(set(y_test))
    )
    plot_accuracy_metrics(metrics, "images/test_accuracy_metrics.png")
    plot_confusion_matrix(
        metrics["true_labels"],
        metrics["predictions"],
        "images/test_confusion_matrix.png",
        label_names=target_names,
    )
    plot_metrics_comparison(
        metrics["true_labels"],
        metrics["predictions"],
        "images/test_metrics_comparison.png",
        label_names=target_names,
    )

    print("[STEP 6/6] Saving test logs...")
    save_test_logs(metrics, n_samples, RANDOM_FOREST_PARAMS)

    print("\n" + "=" * 60)
    print("   Model testing completed successfully!")
    print("=" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    run_model_tests(n_samples=100, random_state=42)
