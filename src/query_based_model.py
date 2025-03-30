import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sentence_transformers import SentenceTransformer
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import os
import joblib

def load_and_prepare_data(query_path, feature_path):
    query_df = pd.read_csv(query_path)
    feature_df = pd.read_csv(feature_path)
    df = query_df.merge(feature_df, on='query_id')[["query_id", "execution_time", "raw_query"]]
    df = df.dropna(subset=['raw_query', 'execution_time'])
    df["execution_time_log"] = np.log1p(df["execution_time"])
    return df

def encode_queries(queries, model_name="s2593817/sft-sql-embedding"):
    print(f"Encoding queries using {model_name}...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(queries.tolist(), batch_size=32, show_progress_bar=True)
    return embeddings

def evaluate(y_true_log, y_pred_log, model_name="Model"):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n Evaluation for {model_name}")
    print(f"MAE (ms):  {mae:.3f}")
    print(f"RMSE (ms): {rmse:.3f}")
    print(f"R²:        {r2:.3f}")
    return mae, rmse, r2

def plot_predictions(y_true_log, y_pred_log, model_name, filename="predictions.png"):
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], '--r')
    plt.xlabel("Actual Execution Time (ms)")
    plt.ylabel("Predicted Execution Time (ms)")
    plt.title(f"Prediction vs Actual - {model_name}")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved plot: {filename}")
    plt.close()

def rmse_original_scale(y_true, y_pred):
    y_true_orig = np.expm1(y_true)
    y_pred_orig = np.expm1(y_pred)
    return np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))

def hyperparameter_tuning(X_train, y_train, model_type="rf"):
    if model_type == "rf":
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 10, 15, 20, 25],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }
        model = RandomForestRegressor(random_state=42)

    elif model_type == "xgb":
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
        }
        model = XGBRegressor(random_state=42)

    else:
        raise ValueError("Unsupported model_type. Use 'rf' or 'xgb'.")

    scorer = make_scorer(rmse_original_scale, greater_is_better=False)

    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scorer,
        cv=3,
        verbose=1,
        n_jobs=-1
    )
    
    grid.fit(X_train, y_train)

    print(f"\nBest Params for {model_type.upper()}:")
    print(grid.best_params_)
    print(f"Best RMSE: {-grid.best_score_:.3f}")
    
    return grid.best_estimator_

def save_models(models: dict, output_dir="saved_models"):
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        model_path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {name} model to {model_path}")


if __name__ == "__main__":
    df = load_and_prepare_data("query_metadata.csv", "training_data.csv")
    X = encode_queries(df["raw_query"])
    y = df["execution_time_log"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tune Hyperparameters
    best_rf = hyperparameter_tuning(X_train, y_train, model_type="rf")
    best_xgb = hyperparameter_tuning(X_train, y_train, model_type="xgb")

    # Predictions
    rf_y_pred = best_rf.predict(X_test)
    xgb_y_pred = best_xgb.predict(X_test)

    # Evaluation in log scale
    evaluate(y_test, rf_y_pred, "Tuned RandomForest (ms)")
    evaluate(y_test, xgb_y_pred, "Tuned XGBoost (ms)")

    # Plot and save results
    plot_predictions(y_test, rf_y_pred, "Tuned RandomForest", "images/query_based_rf.png")
    plot_predictions(y_test, xgb_y_pred, "Tuned XGBoost", "images/query_based_xgb.png")

    save_models({"query_based_rf": best_rf, "query_based_xgb": best_xgb})
