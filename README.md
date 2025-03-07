# Model Evaluation & Comparison Suite - Streamlit App

## Overview
This repository contains a Streamlit application that allows data scientists to upload machine learning model files (pickle or joblib), along with a test dataset (CSV), and then automatically evaluate and compare the models. The app computes key metrics, performs subgroup analysis, and generates interactive visualizations. Additionally, users can generate a downloadable HTML report of the evaluation results.

## Features
- Model Loading: Upload and compare multiple models at once.
- Data Validation: Automatically validates the test dataset and target column.
- Metric Computation:
  - For Classification: Accuracy, Precision, Recall, F1 Score, ROC-AUC, and confusion matrix.
  - For Regression: Mean Squared Error (MSE), Mean Absolute Error (MAE), and R².
- Subgroup Analysis: Automatically bins numeric features with many unique values, or analyzes user-specified subgroups.
- Interactive Visualization: Utilizes Plotly for interactive charts including bar charts, heatmaps (confusion matrices), ROC curves, and residual plots.
- Downloadable Report: Generate and download a comprehensive HTML report that includes tables and interactive charts.
- Custom Domain Deployment: Instructions provided for embedding the app in a GitHub Pages site and linking it to your custom domain.

## Installation
1. Clone the Repository:
   git clone https://github.com/yourusername/model-eval-suite.git
   cd model-eval-suite

2. Install Dependencies:
   Make sure you have Python 3.10+ installed, then run:
   pip install -e .

3. Run the App Locally:
   Launch the Streamlit app by running:
   streamlit run streamlit_app.py

## Usage
- Upload Files: Use the sidebar to upload one or more model files (in .pkl or .joblib format) and a CSV test dataset.
- Configuration: Choose the task type (classification or regression), specify the target column name, and optionally enter subgroup features (comma-separated).
- Evaluation: Click "Run Evaluation" to compute metrics, generate visualizations, and view a comparison summary.
- Download Report: If desired, check the option to generate a downloadable HTML report of the results.

## Testing
Unit tests are provided in the tests/ directory. To run tests:
   pytest

