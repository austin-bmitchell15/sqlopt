import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os

def evaluate_model(model_path, X, y_log, query_ids=None, model_name="Loaded Model"):
    model = joblib.load(model_path)
    y_pred_log = model.predict(X)

    y_true = np.expm1(y_log)
    y_pred = np.expm1(y_pred_log)

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\nEvaluation for {model_name}")
    print(f"MAE (ms):  {mae:.3f}")
    print(f"RMSE (ms): {rmse:.3f}")
    print(f"R²:        {r2:.3f}")

    return y_true, y_pred

def plot_predictions(y_true, y_pred, model_name, filename="predictions_eval.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Execution Time (ms)")
    plt.ylabel("Predicted Execution Time (ms)")
    plt.title(f"{model_name}: Actual vs Predicted")
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    print(f"Saved plot to {filename}")
    plt.close()

def plot_logscale(y_true, y_pred, model_name, filename="images/logscale_plot.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Actual Execution Time (ms, log scale)")
    plt.ylabel("Predicted Execution Time (ms, log scale)")
    plt.title(f"{model_name}: Log-Scale Prediction vs Actual")
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    print(f"Saved log-scale plot to {filename}")
    plt.close()

def plot_zoomed(y_true, y_pred, model_name, filename="images/zoomed_plot.png", percentile=99):
    cutoff = np.percentile(y_true, percentile)
    mask = y_true < cutoff
    y_true_zoom = y_true[mask]
    y_pred_zoom = y_pred[mask]

    plt.figure(figsize=(8, 6))
    plt.scatter(y_true_zoom, y_pred_zoom, alpha=0.6)
    plt.plot([y_true_zoom.min(), y_true_zoom.max()], [y_true_zoom.min(), y_true_zoom.max()], 'r--')
    plt.xlabel("Actual Execution Time (ms)")
    plt.ylabel("Predicted Execution Time (ms)")
    plt.title(f"{model_name}: Prediction vs Actual (Below {percentile}th percentile)")
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    print(f"Saved zoomed-in plot to {filename}")
    plt.close()

def plot_residuals(y_true, y_pred, model_name, filename="images/residuals_plot.png"):
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, residuals, alpha=0.6)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel("Actual Execution Time (ms)")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.title(f"{model_name}: Residual Plot")
    plt.tight_layout()
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename)
    print(f"Saved residual plot to {filename}")
    plt.close()

if __name__ == "__main__":
    # Load X and y from previously saved files
    df = pd.read_csv("query_metadata.csv").merge(
        pd.read_csv("training_data.csv"), on="query_id"
    )[["query_id", "execution_time", "raw_query"]].dropna()

    y_log = np.log1p(df["execution_time"])
    query_ids = df["query_id"]

    # Encode queries
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("s2593817/sft-sql-embedding")
    X = model.encode(df["raw_query"].tolist(), batch_size=32, show_progress_bar=True)

    # Evaluate both models
    y_true_rf, y_pred_rf = evaluate_model("saved_models/query_based_rf.joblib", X, y_log, query_ids, "RandomForest (Full Data)")
    plot_predictions(y_true_rf, y_pred_rf, "RandomForest", "images/query_based_rf/eval_rf_full.png")
    # After evaluating Random Forest
    plot_logscale(y_true_rf, y_pred_rf, "RandomForest", "images/query_based_rf/logscale_rf.png")
    plot_zoomed(y_true_rf, y_pred_rf, "RandomForest", "images/query_based_rf/zoomed_rf.png")
    plot_residuals(y_true_rf, y_pred_rf, "RandomForest", "images/query_based_rf/residuals_rf.png")
    
    y_true_xgb, y_pred_xgb = evaluate_model("saved_models/query_based_xgb.joblib", X, y_log, query_ids, "XGBoost (Full Data)")
    plot_predictions(y_true_xgb, y_pred_xgb, "XGBoost", "images/query_based_xgb/eval_xgb_full.png")
    # After evaluating XGBoost
    plot_logscale(y_true_xgb, y_pred_xgb, "XGBoost", "images/query_based_xgb/logscale.png")
    plot_zoomed(y_true_xgb, y_pred_xgb, "XGBoost", "images/query_based_xgb/zoomed.png")
    plot_residuals(y_true_xgb, y_pred_xgb, "XGBoost", "images/query_based_xgb/residuals.png")
