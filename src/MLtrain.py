"""
Machine Learning Training Module for Diabetes Prediction System.

This module handles all ML model training operations including:
- Model initialization with configurable hyperparameters
- Train-test split operations
- Model training and evaluation
- Model persistence (saving/loading)
- Performance metrics calculation
- Confusion matrix visualization
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Dict, Any
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

from config import (
    TRAINED_MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    CONFUSION_MATRIX_LABELS,
    FIGURE_SIZE,
    FIGURE_DPI,
    COLORMAP,
    RANDOM_FOREST_PARAMS,
    TRAIN_TEST_SPLIT_PARAMS,
    MIN_ACCEPTABLE_ACCURACY,
    CV_FOLDS,
    VERBOSE_LOGGING,
)


class DiabetesModelTrainer:
    """
    DiabetesModelTrainer handles all machine learning model training operations.

    Attributes:
        model: The trained classifier model.
        X_train: Training features.
        X_test: Test features.
        y_train: Training labels.
        y_test: Test labels.
        accuracy: Model accuracy score.
    """

    def __init__(self, model_params: Optional[Dict[str, Any]] = None):
        """
        Initialize the DiabetesModelTrainer.

        Args:
            model_params: Optional custom parameters for RandomForestClassifier.
                        Defaults to config.RANDOM_FOREST_PARAMS.
        """
        self.model_params = model_params or RANDOM_FOREST_PARAMS
        self.model: Optional[RandomForestClassifier] = None
        self.X_train: Optional[pd.DataFrame] = None
        self.X_test: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.y_test: Optional[pd.Series] = None
        self.accuracy: float = 0.0
        self.cv_scores: Optional[np.ndarray] = None

    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: Optional[float] = None,
        random_state: Optional[int] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Split data into training and test sets.

        Args:
            X: Feature dataframe.
            y: Target series.
            test_size: Proportion of data for testing. Defaults to config value.
            random_state: Random seed for reproducibility. Defaults to config value.

        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        test_size = test_size or TRAIN_TEST_SPLIT_PARAMS["test_size"]
        random_state = random_state or TRAIN_TEST_SPLIT_PARAMS["random_state"]

        if VERBOSE_LOGGING:
            print(f"[INFO] Splitting data with test_size={test_size}, random_state={random_state}")

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=TRAIN_TEST_SPLIT_PARAMS.get("shuffle", True),
            stratify=TRAIN_TEST_SPLIT_PARAMS.get("stratify"),
        )

        if VERBOSE_LOGGING:
            print(f"[INFO] Training set size: {len(self.X_train)}")
            print(f"[INFO] Test set size: {len(self.X_test)}")

        return self.X_train, self.X_test, self.y_train, self.y_test

    def initialize_model(self) -> RandomForestClassifier:
        """
        Initialize the Random Forest classifier with configured parameters.

        Returns:
            RandomForestClassifier: The initialized model.
        """
        if VERBOSE_LOGGING:
            print("[INFO] Initializing Random Forest model...")
            print(f"[INFO] Model parameters: {self.model_params}")

        self.model = RandomForestClassifier(**self.model_params)

        return self.model

    def train(self, X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
        """
        Train the model on the provided data.

        Args:
            X: Feature dataframe.
            y: Target series.

        Returns:
            RandomForestClassifier: The trained model.
        """
        if self.model is None:
            self.initialize_model()

        if VERBOSE_LOGGING:
            print("[INFO] Training model...")

        self.model.fit(X, y)

        if VERBOSE_LOGGING:
            print("[INFO] Model training completed.")

        return self.model

    def evaluate(self) -> Dict[str, Any]:
        """
        Evaluate the model on the test set.

        Returns:
            dict: Dictionary containing evaluation metrics.
        """
        if self.model is None or self.X_test is None or self.y_test is None:
            raise ValueError("Model not trained. Call train() first.")

        if VERBOSE_LOGGING:
            print("[INFO] Evaluating model on test set...")

        y_pred = self.model.predict(self.X_test)

        self.accuracy = accuracy_score(self.y_test, y_pred)

        # Calculate cross-validation scores
        if self.X_train is not None and self.y_train is not None:
            self.cv_scores = cross_val_score(self.model, self.X_train, self.y_train, cv=CV_FOLDS)

        metrics = {
            "accuracy": self.accuracy,
            "cross_val_mean": self.cv_scores.mean() if self.cv_scores is not None else None,
            "cross_val_std": self.cv_scores.std() if self.cv_scores is not None else None,
            "classification_report": classification_report(
                self.y_test, y_pred, target_names=CONFUSION_MATRIX_LABELS[: len(np.unique(y_pred))]
            ),
        }

        if VERBOSE_LOGGING:
            print(f"[INFO] Model Accuracy: {self.accuracy:.4f} ({self.accuracy * 100:.2f}%)")
            if self.cv_scores is not None:
                print(
                    f"[INFO] Cross-Validation Score: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std() * 2:.4f})"
                )

        return metrics

    def generate_confusion_matrix(self, save_path: Optional[str] = None) -> np.ndarray:
        """
        Generate and optionally save the confusion matrix visualization.

        Args:
            save_path: Optional path to save the confusion matrix image.
                      Defaults to config.CONFUSION_MATRIX_PATH.

        Returns:
            np.ndarray: The confusion matrix.
        """
        save_path = save_path or str(CONFUSION_MATRIX_PATH)

        if self.model is None or self.X_test is None or self.y_test is None:
            raise ValueError("Model not trained. Call train() first.")

        if VERBOSE_LOGGING:
            print("[INFO] Generating confusion matrix...")

        y_pred = self.model.predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)

        # Create visualization
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CONFUSION_MATRIX_LABELS)
        disp.plot(ax=ax, cmap=COLORMAP, values_format="d")

        plt.title("Diabetes Prediction - Confusion Matrix", fontsize=14, fontweight="bold")
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        # Save the figure
        plt.savefig(save_path, dpi=FIGURE_DPI, bbox_inches="tight")

        if VERBOSE_LOGGING:
            print(f"[INFO] Confusion matrix saved to: {save_path}")

        plt.close()

        return cm

    def save_model(self, save_path: Optional[str] = None) -> str:
        """
        Save the trained model to a pickle file.

        Args:
            save_path: Optional path to save the model.
                      Defaults to config.TRAINED_MODEL_PATH.

        Returns:
            str: Path where the model was saved.
        """
        save_path = save_path or str(TRAINED_MODEL_PATH)

        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if VERBOSE_LOGGING:
            print(f"[INFO] Saving model to: {save_path}")

        # Ensure directory exists
        import os

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "wb") as f:
            pickle.dump(self.model, f)

        if VERBOSE_LOGGING:
            print("[INFO] Model saved successfully.")

        return save_path

    @staticmethod
    def load_model(model_path: str) -> RandomForestClassifier:
        """
        Load a trained model from a pickle file.

        Args:
            model_path: Path to the saved model file.

        Returns:
            RandomForestClassifier: The loaded model.
        """
        if VERBOSE_LOGGING:
            print(f"[INFO] Loading model from: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        if VERBOSE_LOGGING:
            print("[INFO] Model loaded successfully.")

        return model

    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance rankings from the trained model.

        Returns:
            pd.DataFrame: DataFrame with feature names and importance scores.
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")

        if not hasattr(self.model, "feature_importances_"):
            raise ValueError("Model does not support feature importance.")

        feature_names = self.X_train.columns if hasattr(self.X_train, "columns") else None

        importance_df = pd.DataFrame(
            {
                "feature": feature_names
                or [f"feature_{i}" for i in range(len(self.model.feature_importances_))],
                "importance": self.model.feature_importances_,
            }
        ).sort_values("importance", ascending=False)

        if VERBOSE_LOGGING:
            print("[INFO] Feature importance rankings generated.")

        return importance_df

    def train_pipeline(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        save_model: bool = True,
        generate_cm: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute the complete training pipeline.

        Pipeline steps:
        1. Split data into train/test sets
        2. Train the model
        3. Evaluate model performance
        4. Save the trained model (optional)
        5. Generate confusion matrix (optional)

        Args:
            X: Feature dataframe.
            y: Target series.
            save_model: Whether to save the trained model.
            generate_cm: Whether to generate confusion matrix visualization.

        Returns:
            dict: Dictionary containing all results and metrics.
        """
        if VERBOSE_LOGGING:
            print("\n" + "=" * 60)
            print("[INFO] Starting training pipeline...")
            print("=" * 60 + "\n")

        # Step 1: Split data
        self.split_data(X, y)

        # Step 2: Train model
        self.train(self.X_train, self.y_train)

        # Step 3: Evaluate
        metrics = self.evaluate()

        # Step 4: Save model if requested
        model_path = None
        if save_model:
            model_path = self.save_model()
            metrics["model_path"] = model_path

        # Step 5: Generate confusion matrix if requested
        cm_path = None
        if generate_cm:
            cm_path = self.generate_confusion_matrix()
            metrics["confusion_matrix_path"] = str(cm_path)

        # Get feature importance
        feature_importance = self.get_feature_importance()
        metrics["feature_importance"] = feature_importance

        # Check if accuracy meets minimum threshold
        if self.accuracy >= MIN_ACCEPTABLE_ACCURACY:
            if VERBOSE_LOGGING:
                print(
                    f"\n[SUCCESS] Model accuracy ({self.accuracy:.4f}) meets minimum threshold ({MIN_ACCEPTABLE_ACCURACY})"
                )
        else:
            if VERBOSE_LOGGING:
                print(
                    f"\n[WARNING] Model accuracy ({self.accuracy:.4f}) below minimum threshold ({MIN_ACCEPTABLE_ACCURACY})"
                )

        if VERBOSE_LOGGING:
            print("\n" + "=" * 60)
            print("[INFO] Training pipeline completed successfully!")
            print("=" * 60 + "\n")

        return metrics


def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    save_model: bool = True,
    generate_cm: bool = True,
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """
    Convenience function to train the diabetes prediction model.

    Args:
        X: Feature dataframe.
        y: Target series.
        save_model: Whether to save the trained model.
        generate_cm: Whether to generate confusion matrix visualization.

    Returns:
        tuple: (trained model, metrics dictionary)
    """
    trainer = DiabetesModelTrainer()
    results = trainer.train_pipeline(X, y, save_model, generate_cm)
    return trainer.model, results


def predict_diabetes(input_data: np.ndarray, model_path: str) -> np.ndarray:
    """
    Make predictions using a trained model.

    Args:
        input_data: Input features for prediction.
        model_path: Path to the trained model file.

    Returns:
        np.ndarray: Predicted class labels.
    """
    model = DiabetesModelTrainer.load_model(model_path)
    predictions = model.predict(input_data)
    return predictions


def get_model_metrics(model_path: str, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Calculate metrics for a loaded model.

    Args:
        model_path: Path to the trained model file.
        X_test: Test features.
        y_test: True test labels.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    model = DiabetesModelTrainer.load_model(model_path)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }

    return metrics
