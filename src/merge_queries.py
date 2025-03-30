import pandas as pd
import psycopg2
import json
import time
from ast import literal_eval

# === FILES ===
TRAINING_DATA_CSV = "training_data.csv"
OPTIMIZED_RESULTS_CSV = "optimized_results.csv"
OUTPUT_CSV = "final_query_results.csv"


def merge_query_info():
    # Load the two input CSVs
    training_df = pd.read_csv(TRAINING_DATA_CSV)
    optimized_df = pd.read_csv(OPTIMIZED_RESULTS_CSV)

    # Make sure query IDs are stripped of whitespace
    training_df["query_id"] = training_df["query_id"].str.strip()
    optimized_df["query_id"] = optimized_df["query_id"].str.strip()

    # Merge on query_id
    merged = pd.merge(optimized_df, training_df, on="query_id", how="inner")

    # Add delta between actual and predicted time
    merged["delta_predicted_vs_actual"] = merged["execution_time"] - merged["predicted_time"]

    # Save final results
    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"Merged query data written to {OUTPUT_CSV}")


if __name__ == "__main__":
    merge_query_info()

