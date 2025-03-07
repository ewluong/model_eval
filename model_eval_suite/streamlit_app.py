import streamlit as st
import pandas as pd
import os
import tempfile

from model_eval_suite import model_loader, data_loader, metrics, comparer, visualization, utils

# Initialize logging
utils.setup_logging()

st.title("Model Evaluation & Comparison Suite")
st.write("Upload your models and test dataset to evaluate and compare model performance.")

st.sidebar.header("Configuration")

# Upload model files
uploaded_model_files = st.sidebar.file_uploader("Upload Model Files (Pickle/Joblib)", type=["pkl", "joblib"], accept_multiple_files=True)
# Upload test dataset CSV
uploaded_data_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

task = st.sidebar.selectbox("Select Task", ["classification", "regression"])
target_column = st.sidebar.text_input("Target Column Name", value="target")
subgroup_features = st.sidebar.text_input("Subgroup Features (comma separated)", value="")

if st.sidebar.button("Run Evaluation"):
    if not uploaded_model_files:
        st.error("Please upload at least one model file.")
    elif not uploaded_data_file:
        st.error("Please upload a test dataset CSV.")
    elif target_column == "":
        st.error("Please specify the target column.")
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save uploaded models to temp files
            model_paths = []
            for uploaded_file in uploaded_model_files:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                model_paths.append(file_path)
            
            # Save uploaded dataset to temp file
            data_path = os.path.join(temp_dir, uploaded_data_file.name)
            with open(data_path, "wb") as f:
                f.write(uploaded_data_file.getbuffer())
            
            # Load dataset
            try:
                data = data_loader.load_data(data_path, target_column)
            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()
            
            # Load models
            models = {}
            for path in model_paths:
                try:
                    model = model_loader.load_model(path)
                    models[os.path.basename(path)] = model
                except Exception as e:
                    st.error(f"Error loading model from {path}: {e}")
            if not models:
                st.error("No models were successfully loaded.")
                st.stop()
            
            # Compute predictions, metrics, and probabilities (if classification)
            predictions = {}
            probabilities = {}
            global_metrics = {}
            for model_name, model in models.items():
                try:
                    preds = model.predict(data.drop(columns=[target_column]))
                except Exception as e:
                    st.error(f"Error in prediction for model {model_name}: {e}")
                    continue
                predictions[model_name] = preds
                if task == 'classification':
                    if hasattr(model, 'predict_proba'):
                        try:
                            proba = model.predict_proba(data.drop(columns=[target_column]))
                        except Exception as e:
                            st.warning(f"Error in predict_proba for model {model_name}: {e}")
                            proba = None
                    else:
                        proba = None
                    probabilities[model_name] = proba
                    metric_result = metrics.compute_classification_metrics(data[target_column], preds)
                else:
                    metric_result = metrics.compute_regression_metrics(data[target_column], preds)
                global_metrics[model_name] = metric_result
            
            # Subgroup analysis
            subgroup_results = {}
            if subgroup_features:
                features = [f.strip() for f in subgroup_features.split(",") if f.strip() != ""]
                for feature in features:
                    if feature not in data.columns:
                        st.warning(f"Feature '{feature}' not found in dataset. Skipping subgroup analysis for this feature.")
                        continue
                    subgroup_results[feature] = {}
                    if pd.api.types.is_numeric_dtype(data[feature]) and data[feature].nunique() > 10:
                        binned_feature = feature + '_binned'
                        data[binned_feature] = pd.qcut(data[feature], q=4, duplicates='drop')
                        group_key = binned_feature
                    else:
                        group_key = feature
                    unique_values = data[group_key].unique()
                    for val in unique_values:
                        subset = data[data[group_key] == val]
                        subgroup_metrics = {}
                        for model_name, model in models.items():
                            try:
                                subset_preds = model.predict(subset.drop(columns=[target_column]))
                            except Exception as e:
                                st.warning(f"Error in prediction for model {model_name} on subgroup {feature}={val}: {e}")
                                continue
                            if task == 'classification':
                                subgroup_metric = metrics.compute_classification_metrics(subset[target_column], subset_preds)
                            else:
                                subgroup_metric = metrics.compute_regression_metrics(subset[target_column], subset_preds)
                            subgroup_metrics[model_name] = subgroup_metric
                        subgroup_results[feature][str(val)] = subgroup_metrics
            
            # Compare models
            comparison_summary = comparer.compare_models(global_metrics, predictions, data, target_column, task)
            
            # Generate visualizations (returning Plotly figure objects)
            charts = visualization.generate_visualizations(global_metrics, task, data, target_column, predictions, probabilities)
            
            # Display Results
            st.header("Global Metrics")
            for model, mtr in global_metrics.items():
                st.subheader(model)
                st.json(mtr)
            
            st.header("Comparison Summary")
            st.json(comparison_summary)
            
            if subgroup_results:
                st.header("Subgroup Analysis")
                st.json(subgroup_results)
            
            st.header("Visualizations")
            for name, fig in charts.items():
                st.subheader(name)
                st.plotly_chart(fig, use_container_width=True)
            
            st.success("Evaluation completed!")
