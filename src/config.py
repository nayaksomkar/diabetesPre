"""
Configuration module for Diabetes Prediction System.

This module contains all configuration settings, paths, and hyperparameters
required for the diabetes prediction machine learning pipeline.
"""

from pathlib import Path

# ==============================================================================
# PROJECT CONFIGURATION
# ==============================================================================

# Project root directory
PROJECT_ROOT: Path = Path(__file__).parent.parent

# Source code directory
SRC_DIR: Path = PROJECT_ROOT / "src"

# Data directory
DATA_DIR: Path = PROJECT_ROOT / "db"

# Models directory
MODELS_DIR: Path = PROJECT_ROOT / "models"

# Images output directory
IMAGES_DIR: Path = PROJECT_ROOT / "images"

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================

# Input dataset filename
INPUT_DATASET_FILENAME: str = "diabetesDataset.csv"

# Full path to input dataset
INPUT_DATASET_PATH: Path = DATA_DIR / INPUT_DATASET_FILENAME

# Target column name in the dataset
TARGET_COLUMN_NAME: str = "Target"

# Columns to be excluded from training features
COLUMNS_TO_EXCLUDE: list[str] = [TARGET_COLUMN_NAME, "index"]

# ==============================================================================
# MODEL CONFIGURATION
# ==============================================================================

# Random Forest classifier hyperparameters
RANDOM_FOREST_PARAMS: dict = {
    "n_estimators": 100,
    "criterion": "gini",
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
    "bootstrap": True,
    "oob_score": False,
    "random_state": 42,
    "n_jobs": -1,
}

# Train-test split configuration
TRAIN_TEST_SPLIT_PARAMS: dict = {
    "test_size": 0.2,
    "random_state": 42,
    "shuffle": True,
    "stratify": None,
}

# ==============================================================================
# OUTPUT CONFIGURATION
# ==============================================================================

# Trained model output filename
TRAINED_MODEL_FILENAME: str = "diabetesModel.pkl"

# Full path to save trained model
TRAINED_MODEL_PATH: Path = MODELS_DIR / TRAINED_MODEL_FILENAME

# Confusion matrix output filename
CONFUSION_MATRIX_FILENAME: str = "confusionMatrix.png"

# Full path to save confusion matrix visualization
CONFUSION_MATRIX_PATH: Path = IMAGES_DIR / CONFUSION_MATRIX_FILENAME

# ==============================================================================
# PREPROCESSING CONFIGURATION
# ==============================================================================

# Label encoder smoothing parameter
LABEL_ENCODER_smooth: bool = True

# Handle missing values strategy: 'drop', 'fill', or 'interpolate'
MISSING_VALUES_STRATEGY: str = "drop"

# Feature scaling method: 'standard', 'minmax', or None
SCALING_METHOD: str | None = None

# ==============================================================================
# VISUALIZATION CONFIGURATION
# ==============================================================================

# Confusion matrix display labels
CONFUSION_MATRIX_LABELS: list[str] = [
    "Cystic Fibrosis-Related Diabetes (CFRD)",
    "Gestational Diabetes",
    "LADA",
    "MODY",
    "Neonatal Diabetes Mellitus (NDM)",
    "Prediabetic",
    "Secondary Diabetes",
    "Steroid-Induced Diabetes",
    "Type 1 Diabetes",
    "Type 2 Diabetes",
    "Type 3c Diabetes (Pancreatogenic Diabetes)",
    "Wolcott-Rallison Syndrome",
    "Wolfram Syndrome",
]

# Figure size for plots (width, height)
FIGURE_SIZE: tuple[int, int] = (10, 8)

# DPI for saved figures
FIGURE_DPI: int = 100

# Colormap for heatmaps
COLORMAP: str = "Blues"

# ==============================================================================
# PERFORMANCE METRICS
# ==============================================================================

# Minimum acceptable model accuracy
MIN_ACCEPTABLE_ACCURACY: float = 0.85

# Cross-validation folds
CV_FOLDS: int = 5

# ==============================================================================
# LOGGING CONFIGURATION
# ==============================================================================

# Enable verbose logging
VERBOSE_LOGGING: bool = True

# Log level: 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'
LOG_LEVEL: str = "INFO"

# ==============================================================================
# FEATURE ENGINEERING CONFIGURATION
# ==============================================================================

# Categorical columns that need encoding
CATEGORICAL_COLUMNS: list[str] = [
    "Genetic Markers",
    "Autoantibodies",
    "Family History",
    "Environmental Factors",
    "Physical Activity",
    "Dietary Habits",
    "Ethnicity",
    "Socioeconomic Factors",
    "Smoking Status",
    "Alcohol Consumption",
    "Glucose Tolerance Test",
    "History of PCOS",
    "Previous Gestational Diabetes",
    "Pregnancy History",
    "Cystic Fibrosis Diagnosis",
    "Steroid Use History",
    "Genetic Testing",
    "Liver Function Tests",
    "Urine Test",
    "Early Onset Symptoms",
]

# Numerical columns
NUMERICAL_COLUMNS: list[str] = [
    "Insulin Levels",
    "Age",
    "BMI",
    "Blood Pressure",
    "Cholesterol Levels",
    "Waist Circumference",
    "Blood Glucose Levels",
    "Weight Gain During Pregnancy",
    "Pancreatic Health",
    "Pulmonary Function",
    "Neurological Assessments",
    "Digestive Enzyme Levels",
    "Birth Weight",
]
