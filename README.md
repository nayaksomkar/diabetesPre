# Diabetes Prediction System

A simple machine learning project to predict diabetes types from clinical data using Random Forest classifier.

## Quick Start

```bash
cd src
python main.py
```

## Project Structure

```
diabetesPre/
├── db/diabetesDataset.csv     # Training data
├── models/diabetesModel.pkl   # Trained model
├── images/                   # Generated images
│   ├── confusionMatrix.png   # Training evaluation
│   ├── test_accuracy_metrics.png  # Test accuracy metrics
│   ├── test_confusion_matrix.png  # Test confusion matrix
│   └── test_metrics_comparison.png # Per-class metrics
├── src/
│   ├── config.py             # Settings
│   ├── main.py               # Training entry point
│   ├── MLtrain.py            # Training module
│   ├── proData.py            # Data processing
│   ├── run_test.py           # Test execution (main)
│   ├── test_data.py          # Data loading functions
│   ├── test_testing.py       # Testing & evaluation functions
│   └── test_plots.py         # Visualization functions
└── README.md
```

## Usage

| Command | Description |
|---------|-------------|
| `python main.py` | Run full training pipeline |
| `python main.py --predict data.csv` | Predict on new data |
| `python main.py --evaluate` | Evaluate model |
| `python main.py --help` | Show options |
| `python test_model.py` | Test model with 100 random samples |

## Testing

Run the test module to evaluate model performance on 100 random samples:

```bash
python src/run_test.py
```

### Test Module Structure

| File | Purpose |
|------|---------|
| `run_test.py` | Main execution (imports all modules) |
| `test_data.py` | Data loading and preprocessing |
| `test_testing.py` | Test dataset creation & model evaluation |
| `test_plots.py` | Visualization functions |

### Test Results

- **Total samples tested**: 100 random entries from dataset
- **Model used**: Random Forest Classifier
- **Metrics tracked**:
  - Accuracy (90.00%)
  - Precision (Macro/Weighted)
  - Recall (Macro/Weighted)
  - F1 Score (Macro/Weighted)
  - Correct/Incorrect predictions

### Output Files

- `test_logs.json` - Complete test results in JSON format
- `images/test_accuracy_metrics.png` - Overall accuracy visualization
- `images/test_confusion_matrix.png` - Confusion matrix heatmap
- `images/test_metrics_comparison.png` - Per-class performance chart

## Features

- Data preprocessing (missing values, encoding)
- Random Forest classifier
- Model persistence (pickle)
- Confusion matrix visualization
- Model testing with random sample evaluation
- Performance metric graphs (accuracy, precision, recall, F1)

## Requirements

- Python 3.10+
- pandas, scikit-learn, matplotlib

## Install

```bash
pip install -r requirements.txt
```

## Configuration

Edit `src/config.py` to change:
- Data paths
- Model hyperparameters
- Train/test split ratio
- Output settings

## Output

- Trained model: `models/diabetesModel.pkl`
- Confusion matrix: `images/confusionMatrix.png`
- Test metrics: `images/test_*.png`

## Github & HuggingFace
- GithubRepo : https://github.com/nayaksomkar/diabetesPre
- HuggingFace : https://huggingface.co/nayaksomkar/DiabetesPre