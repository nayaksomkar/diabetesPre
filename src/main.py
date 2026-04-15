"""
Main Entry Point for Diabetes Prediction System.

This module provides the primary interface for the diabetes prediction ML pipeline.
It orchestrates the complete workflow: data preprocessing, model training, 
evaluation, and prediction.

Usage:
    python main.py                    # Run full training pipeline
    python main.py --predict <file>  # Make predictions on new data
    python main.py --evaluate        # Evaluate existing model
    python main.py --help            # Show help message
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from config import (
    PROJECT_ROOT,
    INPUT_DATASET_PATH,
    TRAINED_MODEL_PATH,
    CONFUSION_MATRIX_PATH,
    MIN_ACCEPTABLE_ACCURACY,
    VERBOSE_LOGGING,
)
from proData import DataProcessor, load_and_preprocess_data
from MLtrain import DiabetesModelTrainer, train_model, predict_diabetes


def setup_argparse() -> argparse.ArgumentParser:
    """
    Set up command-line argument parser.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Diabetes Prediction System - ML Pipeline",
        epilog="For more information, visit the documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--predict", 
        metavar="FILE",
        help="Path to CSV file containing data to predict"
    )
    
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Evaluate the trained model on test data"
    )
    
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only run training without evaluation"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the trained model"
    )
    
    parser.add_argument(
        "--no-cm",
        action="store_true",
        help="Do not generate confusion matrix"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging output"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version="%(prog)s 1.0.0"
    )
    
    return parser


def run_training_pipeline(
    save_model: bool = True,
    generate_cm: bool = True,
) -> dict:
    """
    Execute the complete training pipeline.
    
    Args:
        save_model: Whether to save the trained model.
        generate_cm: Whether to generate confusion matrix.
        
    Returns:
        dict: Training results and metrics.
    """
    if VERBOSE_LOGGING:
        print("\n" + "="*70)
        print("   DIABETES PREDICTION SYSTEM - MACHINE LEARNING PIPELINE")
        print("="*70 + "\n")
    
    try:
        # Step 1: Load and preprocess data
        if VERBOSE_LOGGING:
            print("[STEP 1/4] Loading and preprocessing data...")
            
        X, y = load_and_preprocess_data()
        
        if VERBOSE_LOGGING:
            print(f"   - Features shape: {X.shape}")
            print(f"   - Target distribution:\n{y.value_counts()}\n")
        
        # Step 2: Train model
        if VERBOSE_LOGGING:
            print("[STEP 2/4] Training model...")
            
        model, results = train_model(
            X=X,
            y=y,
            save_model=save_model,
            generate_cm=generate_cm,
        )
        
        # Step 3: Display results
        if VERBOSE_LOGGING:
            print("[STEP 3/4] Displaying results...")
            print(f"   - Model Accuracy: {results['accuracy']*100:.2f}%")
            
            if results.get('cross_val_mean') is not None:
                print(f"   - Cross-Validation Score: {results['cross_val_mean']*100:.2f}%")
                
            print(f"   - Model saved to: {results.get('model_path', 'N/A')}")
            print(f"   - Confusion matrix saved to: {results.get('confusion_matrix_path', 'N/A')}")
        
        # Step 4: Feature importance
        if VERBOSE_LOGGING:
            print("\n[STEP 4/4] Top 10 Important Features:")
            print("-" * 40)
            
            feature_importance = results.get('feature_importance', pd.DataFrame())
            if not feature_importance.empty:
                for idx, row in feature_importance.head(10).iterrows():
                    print(f"   {row['feature']}: {row['importance']:.4f}")
        
        # Final status
        if results['accuracy'] >= MIN_ACCEPTABLE_ACCURACY:
            status = "SUCCESS"
        else:
            status = "WARNING"
            
        if VERBOSE_LOGGING:
            print("\n" + "="*70)
            print(f"   [{status}] Training pipeline completed!")
            print("="*70 + "\n")
        
        return results
        
    except FileNotFoundError as e:
        print(f"\n[ERROR] File not found: {e}")
        print("Please ensure the dataset exists at the configured path.")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n[ERROR] Training pipeline failed: {e}")
        if VERBOSE_LOGGING:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_prediction(input_file: str) -> np.ndarray:
    """
    Make predictions on new input data.
    
    Args:
        input_file: Path to CSV file containing input data.
        
    Returns:
        np.ndarray: Predicted class labels.
    """
    if VERBOSE_LOGGING:
        print("\n" + "="*70)
        print("   DIABETES PREDICTION SYSTEM - PREDICTION MODE")
        print("="*70 + "\n")
    
    try:
        # Load input data
        if VERBOSE_LOGGING:
            print(f"[INFO] Loading input data from: {input_file}")
            
        input_data = pd.read_csv(input_file)
        
        # Load trained model
        if not TRAINED_MODEL_PATH.exists():
            print(f"\n[ERROR] Model file not found: {TRAINED_MODEL_PATH}")
            print("Please train the model first using: python main.py")
            sys.exit(1)
            
        if VERBOSE_LOGGING:
            print(f"[INFO] Loading trained model from: {TRAINED_MODEL_PATH}")
            
        model = DiabetesModelTrainer.load_model(str(TRAINED_MODEL_PATH))
        
        # Make predictions
        if VERBOSE_LOGGING:
            print("[INFO] Making predictions...")
            
        predictions = model.predict(input_data)
        
        if VERBOSE_LOGGING:
            print(f"[INFO] Predictions generated for {len(predictions)} samples")
            print(f"Predictions: {predictions}")
        
        return predictions
        
    except Exception as e:
        print(f"\n[ERROR] Prediction failed: {e}")
        if VERBOSE_LOGGING:
            import traceback
            traceback.print_exc()
        sys.exit(1)


def run_evaluation() -> dict:
    """
    Evaluate the trained model on the dataset.
    
    Returns:
        dict: Evaluation metrics.
    """
    if VERBOSE_LOGGING:
        print("\n" + "="*70)
        print("   DIABETES PREDICTION SYSTEM - EVALUATION MODE")
        print("="*70 + "\n")
    
    try:
        # Load data
        if VERBOSE_LOGGING:
            print("[INFO] Loading data for evaluation...")
            
        X, y = load_and_preprocess_data()
        
        # Load model
        if not TRAINED_MODEL_PATH.exists():
            print(f"\n[ERROR] Model file not found: {TRAINED_MODEL_PATH}")
            print("Please train the model first using: python main.py")
            sys.exit(1)
            
        model = DiabetesModelTrainer.load_model(str(TRAINED_MODEL_PATH))
        
        # Split and evaluate
        if VERBOSE_LOGGING:
            print("[INFO] Evaluating model...")
            
        trainer = DiabetesModelTrainer()
        X_train, X_test, y_train, y_test = trainer.split_data(X, y)
        
        # Retrain on training set
        model.fit(X_train, y_train)
        
        # Evaluate
        metrics = trainer.evaluate()
        
        if VERBOSE_LOGGING:
            print(f"\n[RESULTS] Model Evaluation:")
            print(f"   - Accuracy: {metrics['accuracy']*100:.2f}%")
            print(f"\n[CLASSIFICATION REPORT]")
            print(metrics['classification_report'])
        
        return metrics
        
    except Exception as e:
        print(f"\n[ERROR] Evaluation failed: {e}")
        sys.exit(1)


def main():
    """Main entry point for the Diabetes Prediction System."""
    parser = setup_argparse()
    args = parser.parse_args()
    
    # Handle quiet mode
    global VERBOSE_LOGGING
    if args.quiet:
        VERBOSE_LOGGING = False
    
    # Route to appropriate function
    if args.predict:
        # Prediction mode
        run_prediction(args.predict)
        
    elif args.evaluate:
        # Evaluation mode
        run_evaluation()
        
    else:
        # Training mode (default)
        save_model = not args.no_save
        generate_cm = not args.no_cm
        
        run_training_pipeline(
            save_model=save_model,
            generate_cm=generate_cm,
        )


if __name__ == "__main__":
    main()
