# Diabetes Prediction 🩺

Predicts 13 types of diabetes from clinical data using Random Forest.

---

## Run

```bash
python src/main.py        # Train model
python src/run_test.py    # Test with 100 samples
```

---

## Results

**Accuracy: 90%** | Precision: 0.91 | Recall: 0.91 | F1: 0.90

### Test Confusion Matrix
![Confusion Matrix](images/test_confusion_matrix.png)

### Accuracy Metrics
![Accuracy](images/test_accuracy_metrics.png)

### Per-Class Performance
![Metrics](images/test_metrics_comparison.png)

### Training Results
![Training](images/confusionMatrix.png)

---

## Project Structure

```
diabetesPre/
├── db/diabetesDataset.csv          # 70,000 samples
├── models/diabetesModel.pkl        # Trained model
├── images/                         # Visualizations
├── test_logs.json                  # Test results
└── src/
    ├── main.py                     # Train
    ├── run_test.py                 # Test
    ├── config.py                   # Settings
    ├── MLtrain.py                  # Model
    ├── proData.py                  # Data processing
    ├── test_data.py                # Test loader
    ├── test_testing.py             # Test functions
    └── test_plots.py               # Charts
```

---

## Links

[GitHub](https://github.com/nayaksomkar/diabetesPre) | [HuggingFace](https://huggingface.co/nayaksomkar/DiabetesPre)