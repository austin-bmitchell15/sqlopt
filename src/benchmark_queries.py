import pandas as pd
import json
import psycopg2
from tqdm import tqdm
import re
import sqlparse

INPUT_CSV = "final_query_results.csv"
OUTPUT_CSV = "optimized_vs_original.csv"

DB_CONFIG = {
    'dbname': 'tpch',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}

def clean_sql_query(query: str) -> str:
    # Remove multi-line comments: /* ... */
    query = re.sub(r"/\*.*?\*/", "", query, flags=re.DOTALL)

    # Remove single-line comments: -- ...
    query = re.sub(r"--.*", "", query)

    # Remove excessive whitespace and trailing semicolon
    query = query.strip().rstrip(";")
    return query

def prettify_sql(query: str) -> str:
    return sqlparse.format(query, reindent=True, keyword_case="upper")

def run_explain(cursor, query, query_id, fallback_query=None):
    try:
        query = prettify_sql(query)
        print(query)

        # Catch common unfinished queries (ends with "desc", "group by", etc.)
        lower_query = query.lower()
        bad_endings = ["group by", "order by", "desc", "asc", "where"]
        if any(lower_query.endswith(e) for e in bad_endings):
            raise ValueError("Query ends suspiciously (likely incomplete).")

        cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
        plan_json = cursor.fetchone()[0][0]
        return plan_json.get("Execution Time", -1)

    except Exception as e:
        print(f"Failed query for {query_id}: {e}")
        with open("failed_queries.log", "a") as f:
            f.write(f"\n--- {query_id} ---\n{query}\nERROR: {e}\n\n")
        return -1


def run_all():
    df = pd.read_csv(INPUT_CSV)
    df["query_id"] = df["query_id"].astype(str).str.strip()
    results = []
    print(df["optimized_query"])

    conn = psycopg2.connect(**DB_CONFIG)
    cursor = conn.cursor()

    print(df[["query_id", "optimized_query", "execution_time"]].head())

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Running optimized queries"):
        query_id = row["query_id"]
        original_query = row["original_query"]
        optimized_query = row["optimized_query"]
        original_time = row["execution_time"]

        print(optimized_query)
        print(row["optimized_query"])


        opt_time = run_explain(cursor, optimized_query, query_id)

        results.append({
            "query_id": query_id,
            "original_query": original_query,
            "execution_time": original_time,
            "optimized_query": optimized_query,
            "opt_execution_time": opt_time
        })

    cursor.close()
    conn.close()

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Comparison saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    run_all()

