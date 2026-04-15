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
├── images/confusionMatrix.png # Evaluation
├── src/
│   ├── config.py             # Settings
│   ├── main.py               # Entry point
│   ├── MLtrain.py            # Training module
│   └── proData.py            # Data processing
└── README.md
```

## Usage

| Command | Description |
|---------|-------------|
| `python main.py` | Run full training pipeline |
| `python main.py --predict data.csv` | Predict on new data |
| `python main.py --evaluate` | Evaluate model |
| `python main.py --help` | Show options |

## Features

- Data preprocessing (missing values, encoding)
- Random Forest classifier
- Model persistence (pickle)
- Confusion matrix visualization

## Requirements

- Python 3.10+
- pandas, scikit-learn, matplotlib, seaborn

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


GithubRepo : https://github.com/nayaksomkar/diabetesPre
HuggingFace : https://huggingface.co/nayaksomkar/DiabetesPre