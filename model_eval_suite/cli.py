import argparse
import os
import logging
import pandas as pd

from model_eval_suite import model_loader, data_loader, metrics, comparer, visualization, report_generator, utils

def main():
    parser = argparse.ArgumentParser(description="Model Evaluation & Comparison Suite")
    parser.add_argument('--models', nargs='+', required=True, help='Path(s) to model file(s)')
    parser.add_argument('--data', required=True, help='Path to test dataset CSV file')
    parser.add_argument('--task', choices=['classification', 'regression'], required=True, help='Task type')
    parser.add_argument('--target', required=True, help='Target column name in dataset')
    parser.add_argument('--subgroup_features', nargs='*', default=[], help='Features for subgroup analysis')
    parser.add_argument('--output', required=True, help='Output directory for the final HTML report')
    
    args = parser.parse_args()
    
    # Setup logging
    utils.setup_logging()
    logging.info("Starting Model Evaluation & Comparison Suite")

    # Ensure output directory exists
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # Load models
    models = {}
    for model_path in args.models:
        try:
            model = model_loader.load_model(model_path)
            models[os.path.basename(model_path)] = model
            logging.info(f"Loaded model from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {e}")
            return
    
    # Load dataset
    try:
        data = data_loader.load_data(args.data, args.target)
        logging.info(f"Loaded data from {args.data}")
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        return
    
    # Compute global metrics, predictions, and (if applicable) probabilities for classification
    predictions = {}
    probabilities = {}
    global_metrics = {}
    for model_name, model in models.items():
        try:
            preds = model.predict(data.drop(columns=[args.target]))
        except Exception as e:
            logging.error(f"Error in prediction for model {model_name}: {e}")
            continue
        predictions[model_name] = preds
        if args.task == 'classification':
            if hasattr(model, 'predict_proba'):
                try:
                    proba = model.predict_proba(data.drop(columns=[args.target]))
                except Exception as e:
                    logging.error(f"Error in predict_proba for model {model_name}: {e}")
                    proba = None
            else:
                proba = None
            probabilities[model_name] = proba
            metric_result = metrics.compute_classification_metrics(data[args.target], preds)
        else:
            metric_result = metrics.compute_regression_metrics(data[args.target], preds)
        global_metrics[model_name] = metric_result

    # Subgroup analysis with automatic binning for numeric features with many unique values
    subgroup_results = {}
    if args.subgroup_features:
        for feature in args.subgroup_features:
            subgroup_results[feature] = {}
            # If numeric and many unique values, bin the feature
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
                        subset_preds = model.predict(subset.drop(columns=[args.target]))
                    except Exception as e:
                        logging.error(f"Error in prediction for model {model_name} on subgroup {feature}={val}: {e}")
                        continue
                    if args.task == 'classification':
                        subgroup_metric = metrics.compute_classification_metrics(subset[args.target], subset_preds)
                    else:
                        subgroup_metric = metrics.compute_regression_metrics(subset[args.target], subset_preds)
                    subgroup_metrics[model_name] = subgroup_metric
                subgroup_results[feature][val] = subgroup_metrics

    # Comparison
    comparison_summary = comparer.compare_models(global_metrics, predictions, data, args.target, args.task)

    # Visualization (using Plotly for interactive futuristic charts, returning HTML snippets)
    charts = visualization.generate_visualizations(
        global_metrics, args.task, data, args.target, predictions, probabilities
    )

    # Report Generation: create one consolidated HTML report with all results and embedded visualizations
    report_path = report_generator.generate_report(
        global_metrics, subgroup_results, comparison_summary, charts, args.output
    )
    logging.info(f"Report generated successfully at: {report_path}")

if __name__ == '__main__':
    main()
