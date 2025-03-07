# Model Evaluation & Comparison Suite

## Overview
This CLI tool helps data scientists load, evaluate, and compare multiple trained machine learning models on a given test dataset. It computes evaluation metrics, performs subgroup analysis, and generates comprehensive reports in Markdown and/or HTML.

## Features
- **Model Loading:** Load models saved as pickle or joblib files.
- **Data Validation:** Load and validate test datasets (CSV) ensuring the target column exists.
- **Metric Computation:**
  - **Classification:** Accuracy, precision, recall, F1 score, ROC-AUC, and confusion matrix.
  - **Regression:** Mean Squared Error (MSE), Mean Absolute Error (MAE), and R².
- **Subgroup Analysis:** Automatically bin numerical features (if too many unique values) or analyze user-specified subgroups.
- **Model Comparison:** Compare global metrics and highlight significant prediction differences.
- **Visualization:** Generate bar charts, ROC curves (if applicable), confusion matrices, and residual plots.
- **Report Generation:** Create comprehensive Markdown and/or HTML reports using Jinja2 templates.
- **CLI Interface:** User-friendly command-line interface with robust error handling.
- **Extensibility:** Designed to be extended (e.g., adding a lightweight web UI).

## Installation
Make sure you have Python 3.x installed.

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/model-eval-suite.git
   cd model-eval-suite
