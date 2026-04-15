"""
Data loading and preprocessing functions.
"""

import pandas as pd
from proData import DataProcessor


def load_test_data():
    """
    Load and preprocess the dataset.

    Returns:
        tuple: (X, y) features and target
    """
    processor = DataProcessor()
    X, y = processor.preprocess_pipeline()
    return X, y, processor
