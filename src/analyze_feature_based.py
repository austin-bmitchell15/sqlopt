import pandas as pd
import numpy as np
import joblib
import ast
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


def preprocess_features(csv_path):
    df = pd.read_csv(csv_path)
    df["base_query"] = df["query_id"].str.extract(r"(query\d+)")
    df["node_types"] = df["node_types"].apply(ast.literal_eval)
    node_df = df["node_types"].apply(pd.Series).fillna(0).astype(int)
    node_df.columns = [f"node_{col.replace(' ', '_')}" for col in node_df.columns]
    df = df.drop(["node_types"], axis=1).join(node_df)

    bool_cols = ['has_seq_scan', 'has_sort', 'has_index_scan']
    df[bool_cols] = df[bool_cols].astype(int)

    numerical_cols = ['planning_time', 'total_rows', 'total_loops', 'max_depth', 'workers_launched']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    X = df.drop(["execution_time", "query_id", "base_query"], axis=1)
    y = np.log1p(df["execution_time"])
    query_ids = df["query_id"]
    return X, y, query_ids


def evaluate(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"\n📊 Evaluation for {model_name}")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R²:   {r2:.3f}")
    return mae, rmse, r2


def plot_predictions(y_true, y_pred, model_name, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
    plt.xlabel("Actual Execution Time (ms)")
    plt.ylabel("Predicted Execution Time (ms)")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

def plot_logscale(y_true, y_pred, model_name, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xscale("log")
    plt.yscale("log")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
    plt.xlabel("Actual Execution Time (ms, log)")
    plt.ylabel("Predicted Execution Time (ms, log)")
    plt.title(f"{model_name}: Log-Log Plot")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

def plot_zoomed(y_true, y_pred, model_name, filename, percentile=99):
    cutoff = np.percentile(y_true, percentile)
    mask = y_true < cutoff
    y_true_zoom = y_true[mask]
    y_pred_zoom = y_pred[mask]
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_zoom, y_pred_zoom, alpha=0.5)
    plt.plot([y_true_zoom.min(), y_true_zoom.max()], [y_true_zoom.min(), y_true_zoom.max()], '--r')
    plt.xlabel("Actual Execution Time (ms)")
    plt.ylabel("Predicted Execution Time (ms)")
    plt.title(f"{model_name}: Below {percentile}th Percentile")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

def plot_residuals(y_true, y_pred, model_name, filename):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, residuals, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Actual Execution Time (ms)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"{model_name}: Residual Plot")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved: {filename}")
    plt.close()

if __name__ == "__main__":
    # Load data
    X, y_log, query_ids = preprocess_features("training_data.csv")
    y_ms = np.expm1(y_log)

    # Load models
    rf = joblib.load("saved_models/feature_based_rf.joblib")
    xgb = joblib.load("saved_models/feature_based_xgb.joblib")

    # Predict
    rf_pred_log = rf.predict(X)
    xgb_pred_log = xgb.predict(X)

    rf_pred_ms = np.expm1(rf_pred_log)
    xgb_pred_ms = np.expm1(xgb_pred_log)

    # Evaluate
    evaluate(y_ms, rf_pred_ms, "RandomForest (ms)")
    evaluate(y_ms, xgb_pred_ms, "XGBoost (ms)")

    # Plot RF
    plot_predictions(y_ms, rf_pred_ms, "RandomForest", "images/feature_based_rf/scatter.png")
    plot_logscale(y_ms, rf_pred_ms, "RandomForest", "images/feature_based_rf/logscale.png")
    plot_zoomed(y_ms, rf_pred_ms, "RandomForest", "images/feature_based_rf/zoomed.png")
    plot_residuals(y_ms, rf_pred_ms, "RandomForest", "images/feature_based_rf/residuals.png")

    # Plot XGB
    plot_predictions(y_ms, xgb_pred_ms, "XGBoost", "images/feature_based_xgb/scatter.png")
    plot_logscale(y_ms, xgb_pred_ms, "XGBoost", "images/feature_based_xgb/logscale.png")
    plot_zoomed(y_ms, xgb_pred_ms, "XGBoost", "images/feature_based_xgb/zoomed.png")
    plot_residuals(y_ms, xgb_pred_ms, "XGBoost", "images/feature_based_xgb/residuals.png")

