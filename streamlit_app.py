import streamlit as st
import pandas as pd
import os
import tempfile
import base64

from model_eval_suite import model_loader, data_loader, metrics, comparer, visualization, utils

# Initialize logging
utils.setup_logging()

st.title("Model Evaluation & Comparison Suite")
st.write("Upload your models and test dataset to evaluate and compare model performance.")

st.sidebar.header("Configuration")

# File upload widgets
uploaded_model_files = st.sidebar.file_uploader(
    "Upload Model Files (Pickle/Joblib)", type=["pkl", "joblib"], accept_multiple_files=True
)
uploaded_data_file = st.sidebar.file_uploader("Upload Test Dataset (CSV)", type=["csv"])

# Task configuration
task = st.sidebar.selectbox("Select Task", ["classification", "regression"])
target_column = st.sidebar.text_input("Target Column Name", value="target")
subgroup_features = st.sidebar.text_input("Subgroup Features (comma separated)", value="")

# Let the user choose which metric to visualize
if task == "classification":
    metric_options = ["accuracy", "precision", "recall", "f1", "roc_auc"]
else:
    metric_options = ["mse", "mae", "r2"]
selected_metric = st.sidebar.selectbox("Select Metric for Visualization", metric_options)

# Option for downloadable report
generate_report_option = st.sidebar.checkbox("Generate Downloadable HTML Report", value=True)

# --- Caching wrappers ---
@st.cache_data(show_spinner=False)
def load_uploaded_data(file_path, target):
    return data_loader.load_data(file_path, target)

@st.cache_resource(show_spinner=False)
def load_model_cached(file_path):
    return model_loader.load_model(file_path)

# Updated: Renamed parameter to _model so it isn't hashed
@st.cache_data(show_spinner=False)
def compute_predictions(_model, data_without_target):
    return _model.predict(data_without_target)

def generate_html_report(global_metrics, comparison_summary, subgroup_results, charts):
    # Build an HTML report that consolidates tables and interactive charts
    # Escape curly braces in the CSS by doubling them.
    html = """
    <!DOCTYPE html>
    <html>
    <head>
      <meta charset="utf-8">
      <title>Model Evaluation Report</title>
      <style>
        body {{ font-family: Arial, sans-serif; padding: 20px; }}
        h1, h2, h3 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; }}
        th {{ background-color: #f2f2f2; }}
      </style>
    </head>
    <body>
      <h1>Model Evaluation Report</h1>
      <h2>Global Metrics</h2>
      {global_metrics_table}
      <h2>Comparison Summary</h2>
      {comparison_table}
      <h2>Subgroup Analysis</h2>
      {subgroup_tables}
      <h2>Visualizations</h2>
      {charts_section}
    </body>
    </html>
    """
    # Global metrics table
    global_list = []
    for model, mtr in global_metrics.items():
        metrics_for_table = {k: v for k, v in mtr.items() if k != "confusion_matrix"}
        metrics_for_table["Model"] = model
        global_list.append(metrics_for_table)
    df_global = pd.DataFrame(global_list).set_index("Model").to_html(classes="table", border=0)
    
    # Comparison summary table
    pred_diff = comparison_summary.get("prediction_differences", [])
    if pred_diff:
        diff_list = []
        for diff in pred_diff:
            row = {"Row Index": diff.get("index"), "True Value": diff.get("true")}
            for model, pred in diff.get("predictions", {}).items():
                row[model] = pred
            diff_list.append(row)
        df_diff = pd.DataFrame(diff_list).to_html(classes="table", border=0)
    else:
        df_diff = "<p>No significant prediction differences detected.</p>"
    
    # Subgroup analysis tables
    subgroup_html = ""
    if subgroup_results:
        for feature, groups in subgroup_results.items():
            subgroup_html += f"<h3>Feature: {feature}</h3>"
            for subgroup, metrics in groups.items():
                subgroup_html += f"<h4>{feature} = {subgroup}</h4>"
                subgroup_list = []
                for model, mtr in metrics.items():
                    row = {"Model": model}
                    for metric, value in mtr.items():
                        if metric != "confusion_matrix":
                            row[metric] = value
                    subgroup_list.append(row)
                df_sub = pd.DataFrame(subgroup_list).set_index("Model").to_html(classes="table", border=0)
                subgroup_html += df_sub
    else:
        subgroup_html = "<p>No subgroup analysis was performed.</p>"
    
    # Visualizations section
    charts_html = ""
    for name, fig in charts.items():
        charts_html += f"<h3>{name}</h3>" + fig.to_html(full_html=False, include_plotlyjs="cdn")
    
    report_html = html.format(
        global_metrics_table=df_global,
        comparison_table=df_diff,
        subgroup_tables=subgroup_html,
        charts_section=charts_html
    )
    return report_html

# ---------------- Main Evaluation Logic ----------------
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
            
            # Load dataset (using caching)
            try:
                data = load_uploaded_data(data_path, target_column)
            except Exception as e:
                st.error(f"Error loading data: {e}")
                st.stop()
            
            # Load models (using caching)
            models = {}
            for path in model_paths:
                try:
                    model = load_model_cached(path)
                    models[os.path.basename(path)] = model
                except Exception as e:
                    st.error(f"Error loading model from {path}: {e}")
            if not models:
                st.error("No models were successfully loaded.")
                st.stop()
            
            # Compute predictions and metrics for each model
            predictions = {}
            probabilities = {}
            global_metrics = {}
            data_X = data.drop(columns=[target_column]).values
            for model_name, model in models.items():
                try:
                    preds = compute_predictions(model, data_X)
                except Exception as e:
                    st.error(f"Error in prediction for model {model_name}: {e}")
                    continue
                predictions[model_name] = preds
                if task == "classification":
                    if hasattr(model, "predict_proba"):
                        try:
                            proba = model.predict_proba(data_X)
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
            
            # Perform subgroup analysis if specified
            subgroup_results = {}
            if subgroup_features:
                features = [f.strip() for f in subgroup_features.split(",") if f.strip()]
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
                                subset_preds = compute_predictions(model, subset.drop(columns=[target_column]).values)
                            except Exception as e:
                                st.warning(f"Error in prediction for model {model_name} on subgroup {feature}={val}: {e}")
                                continue
                            if task == "classification":
                                subgroup_metric = metrics.compute_classification_metrics(subset[target_column], subset_preds)
                            else:
                                subgroup_metric = metrics.compute_regression_metrics(subset[target_column], subset_preds)
                            subgroup_metrics[model_name] = subgroup_metric
                        subgroup_results[feature][str(val)] = subgroup_metrics
            
            # Compare models
            comparison_summary = comparer.compare_models(global_metrics, predictions, data, target_column, task)
            
            # Generate visualizations
            charts = visualization.generate_visualizations(global_metrics, task, data, target_column, predictions, probabilities)
            
            # ---------------- Display Results Using Improved Template ----------------
            st.header("Global Metrics")
            global_metrics_list = []
            for model, mtr in global_metrics.items():
                metrics_for_table = {k: v for k, v in mtr.items() if k != "confusion_matrix"}
                metrics_for_table["Model"] = model
                global_metrics_list.append(metrics_for_table)
            if global_metrics_list:
                df_global = pd.DataFrame(global_metrics_list).set_index("Model")
                st.table(df_global)
            else:
                st.write("No global metrics to display.")
            
            st.header("Comparison Summary")
            pred_diff = comparison_summary.get("prediction_differences", [])
            if pred_diff:
                diff_list = []
                for diff in pred_diff:
                    row = {"Row Index": diff.get("index"), "True Value": diff.get("true")}
                    for model, pred in diff.get("predictions", {}).items():
                        row[model] = pred
                    diff_list.append(row)
                df_diff = pd.DataFrame(diff_list)
                st.dataframe(df_diff)
            else:
                st.write("No significant prediction differences detected.")
            
            if subgroup_results:
                st.header("Subgroup Analysis")
                for feature, groups in subgroup_results.items():
                    st.subheader(f"Feature: {feature}")
                    for subgroup, metrics_dict in groups.items():
                        st.markdown(f"**{feature} = {subgroup}**")
                        subgroup_list = []
                        for model, mtr in metrics_dict.items():
                            row = {"Model": model}
                            for k, v in mtr.items():
                                if k != "confusion_matrix":
                                    row[k] = v
                            subgroup_list.append(row)
                        df_subgroup = pd.DataFrame(subgroup_list).set_index("Model")
                        st.table(df_subgroup)
            
            st.header("Visualizations")
            if selected_metric:
                st.subheader(f"Chart for {selected_metric}")
                if selected_metric in charts:
                    st.plotly_chart(charts[selected_metric], use_container_width=True, key=f"chart-selected-{selected_metric}")
                else:
                    st.write("Selected metric chart is not available.")
            with st.expander("Show All Visualizations"):
                for name, fig in charts.items():
                    st.subheader(name)
                    st.plotly_chart(fig, use_container_width=True, key=f"chart-all-{name}")
            
            if generate_report_option:
                report_html = generate_html_report(global_metrics, comparison_summary, subgroup_results, charts)
                b64 = base64.b64encode(report_html.encode()).decode()
                href = f'<a href="data:file/html;base64,{b64}" download="model_evaluation_report.html">Download HTML Report</a>'
                st.markdown(href, unsafe_allow_html=True)
            
            st.success("Evaluation completed!")
