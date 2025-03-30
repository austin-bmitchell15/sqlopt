import pandas as pd
import numpy as np
import ast
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
import joblib
import os

def preprocess_features(csv_path):
    df = pd.read_csv(csv_path)
    df["base_query"] = df["query_id"].str.extract(r"(query\d+)")

    # Parse node_types string into columns
    df["node_types"] = df["node_types"].apply(ast.literal_eval)
    node_df = df["node_types"].apply(pd.Series).fillna(0).astype(int)
    node_df.columns = [f"node_{col.replace(' ', '_')}" for col in node_df.columns]

    # Merge expanded node types
    df = df.drop(["node_types"], axis=1).join(node_df)

    bool_cols = ['has_seq_scan', 'has_sort', 'has_index_scan']
    df[bool_cols] = df[bool_cols].astype(int)

    # Scale numeric columns
    numerical_cols = ['planning_time', 'total_rows', 'total_loops', 'max_depth', 'workers_launched']
    scaler = StandardScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

    # Drop unneeded columns
    X_full = df.drop(["execution_time", "query_id", "base_query"], axis=1)
    y_full = np.log1p(df["execution_time"])  # log transform target

    return X_full, y_full



def train_model(X_train, y_train, model_type="rf", params=None):
    if model_type == "rf":
        model = RandomForestRegressor(random_state=42, **(params or {}))
    elif model_type == "xgb":
        model = XGBRegressor(random_state=42, **(params or {}))
    else:
        raise ValueError("Unsupported model_type. Use 'rf' or 'xgb'.")

    model.fit(X_train, y_train)
    return model

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
        raise ValueError("Unsupported model_type")
    
    def rmse_original_scale(y_true, y_pred):
        y_true_orig = np.expm1(y_true)
        y_pred_orig = np.expm1(y_pred)
        return np.sqrt(np.mean((y_true_orig - y_pred_orig) ** 2))
    
    rmse_scorer = make_scorer(rmse_original_scale, greater_is_better=False)

    search = GridSearchCV(model, param_grid, cv=3, scoring=rmse_scorer, verbose=1, n_jobs=-1)
    search.fit(X_train, y_train)

    print(f"\nBest Params for {model_type.upper()}:")
    print(search.best_params_)
    print(f"Best RMSE: {-search.best_score_:.3f}")
    return search.best_estimator_


def evaluate(test, pred, model_name):
    mae = mean_absolute_error(test, pred)
    rmse = root_mean_squared_error(test, pred)
    r2 = r2_score(test, pred)

    print(f"Evaluation Metrics for {model_name}:")
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
    plt.title(f"Prediction vs Actual - {model_name}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Plot saved as {filename}")
    plt.close()

def save_models(models: dict, output_dir="saved_models"):
    os.makedirs(output_dir, exist_ok=True)
    for name, model in models.items():
        model_path = os.path.join(output_dir, f"{name}.joblib")
        joblib.dump(model, model_path)
        print(f"Saved {name} model to {model_path}")

if __name__ == "__main__":
    # Preprocess and split data
    X, y = preprocess_features("training_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Tune Hyperparameters
    best_rf = hyperparameter_tuning(X_train, y_train, model_type="rf")
    best_xgb = hyperparameter_tuning(X_train, y_train, model_type="xgb")

    # Predictions
    rf_y_pred = best_rf.predict(X_test)
    xgb_y_pred = best_xgb.predict(X_test)

    # Evaluation in log scale
    evaluate(y_test, rf_y_pred, "Tuned RandomForest (log)")
    evaluate(y_test, xgb_y_pred, "Tuned XGBoost (log)")

    # Inverse transform to milliseconds
    y_test_ms = np.expm1(y_test)
    rf_pred_ms = np.expm1(rf_y_pred)
    xgb_pred_ms = np.expm1(xgb_y_pred)

    evaluate(y_test_ms, rf_pred_ms, "Tuned RandomForest (ms)")
    evaluate(y_test_ms, xgb_pred_ms, "Tuned XGBoost (ms)")

    # Plot and save results
    plot_predictions(y_test_ms, rf_pred_ms, "Tuned RandomForest", "images/feature_based_rf.png")
    plot_predictions(y_test_ms, xgb_pred_ms, "Tuned XGBoost", "images/feature_based_xgb.png")

    save_models({"feature_based_rf": best_rf, "feature_based_xgb": best_xgb})