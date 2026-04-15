"""
Data Processing Module for Diabetes Prediction System.

This module handles all data preprocessing tasks including:
- Loading datasets from CSV files
- Handling missing values
- Encoding categorical variables
- Feature extraction and transformation
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler

from config import (
    INPUT_DATASET_PATH,
    TARGET_COLUMN_NAME,
    COLUMNS_TO_EXCLUDE,
    MISSING_VALUES_STRATEGY,
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    SCALING_METHOD,
    VERBOSE_LOGGING,
)


class DataProcessor:
    """
    DataProcessor handles all preprocessing operations for the diabetes dataset.
    
    Attributes:
        dataframe (pd.DataFrame): The loaded and processed dataframe.
        label_encoder (LabelEncoder): Encoder for categorical variables.
        scaler (StandardScaler): Scaler for numerical features (if enabled).
    """
    
    def __init__(self, dataset_path: Optional[str] = None):
        """
        Initialize the DataProcessor.
        
        Args:
            dataset_path: Optional custom path to the dataset. 
                         Defaults to config.INPUT_DATASET_PATH.
        """
        self.dataset_path = dataset_path or INPUT_DATASET_PATH
        self.dataframe: Optional[pd.DataFrame] = None
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self._feature_columns: list[str] = []
        self._label_encoder_cache: dict[str, LabelEncoder] = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the diabetes dataset from CSV file.
        
        Returns:
            pd.DataFrame: The loaded dataframe.
            
        Raises:
            FileNotFoundError: If the dataset file does not exist.
            ValueError: If the dataset is empty or invalid.
        """
        if VERBOSE_LOGGING:
            print(f"[INFO] Loading dataset from: {self.dataset_path}")
            
        self.dataframe = pd.read_csv(self.dataset_path)
        
        if self.dataframe.empty:
            raise ValueError("Loaded dataset is empty.")
            
        if VERBOSE_LOGGING:
            print(f"[INFO] Dataset loaded successfully. Shape: {self.dataframe.shape}")
            
        return self.dataframe
    
    def handle_missing_values(self, strategy: Optional[str] = None) -> pd.DataFrame:
        """
        Handle missing values in the dataset using the specified strategy.
        
        Args:
            strategy: Strategy for handling missing values. 
                     Options: 'drop', 'fill', 'interpolate'.
                     Defaults to config.MISSING_VALUES_STRATEGY.
                     
        Returns:
            pd.DataFrame: Dataframe with missing values handled.
        """
        strategy = strategy or MISSING_VALUES_STRATEGY
        
        if self.dataframe is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        initial_rows = len(self.dataframe)
        
        if strategy == "drop":
            self.dataframe = self.dataframe.dropna()
        elif strategy == "fill":
            self.dataframe = self.dataframe.fillna(self.dataframe.mode().iloc[0])
        elif strategy == "interpolate":
            self.dataframe = self.dataframe.interpolate(method="linear", limit_direction="both")
        else:
            raise ValueError(f"Unknown missing values strategy: {strategy}")
            
        if VERBOSE_LOGGING:
            rows_removed = initial_rows - len(self.dataframe)
            print(f"[INFO] Missing values handled using '{strategy}'. Rows removed: {rows_removed}")
            
        return self.dataframe
    
    def encode_categorical_columns(self) -> pd.DataFrame:
        """
        Encode all categorical columns using Label Encoding.
        
        Returns:
            pd.DataFrame: Dataframe with categorical columns encoded.
        """
        if self.dataframe is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if VERBOSE_LOGGING:
            print("[INFO] Encoding categorical columns...")
            
        for column in CATEGORICAL_COLUMNS:
            if column in self.dataframe.columns:
                if column not in self._label_encoder_cache:
                    self._label_encoder_cache[column] = LabelEncoder()
                    
                encoder = self._label_encoder_cache[column]
                self.dataframe[column] = encoder.fit_transform(
                    self.dataframe[column].astype(str)
                )
                
        if VERBOSE_LOGGING:
            print(f"[INFO] Encoded {len(self._label_encoder_cache)} categorical columns.")
            
        return self.dataframe
    
    def select_features_and_target(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Extract feature columns and target variable from the dataframe.
        
        Returns:
            tuple: (features DataFrame, target Series)
        """
        if self.dataframe is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        # Identify feature columns (exclude target and specified columns)
        self._feature_columns = [
            col for col in self.dataframe.columns 
            if col not in COLUMNS_TO_EXCLUDE
        ]
        
        X = self.dataframe[self._feature_columns].copy()
        y = self.dataframe[TARGET_COLUMN_NAME].copy()
        
        if VERBOSE_LOGGING:
            print(f"[INFO] Selected {len(self._feature_columns)} features for training.")
            
        return X, y
    
    def scale_numerical_features(self, method: Optional[str] = None) -> pd.DataFrame:
        """
        Scale numerical features using the specified method.
        
        Args:
            method: Scaling method. Options: 'standard', 'minmax', or None.
                   Defaults to config.SCALING_METHOD.
                   
        Returns:
            pd.DataFrame: Dataframe with scaled numerical features.
        """
        method = method or SCALING_METHOD
        
        if self.dataframe is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        if method is None:
            return self.dataframe
            
        if VERBOSE_LOGGING:
            print(f"[INFO] Scaling numerical features using '{method}'...")
            
        numerical_cols = [
            col for col in NUMERICAL_COLUMNS 
            if col in self.dataframe.columns
        ]
        
        if method == "standard":
            self.dataframe[numerical_cols] = StandardScaler().fit_transform(
                self.dataframe[numerical_cols]
            )
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.dataframe[numerical_cols] = MinMaxScaler().fit_transform(
                self.dataframe[numerical_cols]
            )
            
        if VERBOSE_LOGGING:
            print(f"[INFO] Scaled {len(numerical_cols)} numerical columns.")
            
        return self.dataframe
    
    def get_data_summary(self) -> dict:
        """
        Get a summary of the current dataset.
        
        Returns:
            dict: Summary statistics of the dataset.
        """
        if self.dataframe is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        summary = {
            "total_rows": len(self.dataframe),
            "total_columns": len(self.dataframe.columns),
            "feature_columns": self._feature_columns,
            "missing_values": self.dataframe.isnull().sum().to_dict(),
            "dtypes": self.dataframe.dtypes.astype(str).to_dict(),
        }
        
        return summary
    
    def preprocess_pipeline(self) -> tuple[pd.DataFrame, pd.Series]:
        """
        Execute the complete preprocessing pipeline.
        
        Pipeline steps:
        1. Load data
        2. Handle missing values
        3. Encode categorical columns
        4. Scale numerical features (if configured)
        5. Extract features and target
        
        Returns:
            tuple: (features DataFrame, target Series)
        """
        if VERBOSE_LOGGING:
            print("[INFO] Starting preprocessing pipeline...")
            
        self.load_data()
        self.handle_missing_values()
        self.encode_categorical_columns()
        self.scale_numerical_features()
        X, y = self.select_features_and_target()
        
        if VERBOSE_LOGGING:
            print("[INFO] Preprocessing pipeline completed successfully.")
            
        return X, y


def load_and_preprocess_data() -> tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to load and preprocess the diabetes dataset.
    
    Returns:
        tuple: (features DataFrame, target Series)
    """
    processor = DataProcessor()
    return processor.preprocess_pipeline()


def get_encoders() -> dict[str, LabelEncoder]:
    """
    Get the cached label encoders for inverse transformation.
    
    Returns:
        dict: Dictionary mapping column names to their LabelEncoders.
    """
    processor = DataProcessor()
    processor.load_data()
    processor.encode_categorical_columns()
    return processor._label_encoder_cache
