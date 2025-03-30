import sqlparse
import re
import random
import joblib
import numpy as np
import pandas as pd
import argparse
from itertools import permutations
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# === Lazy-load Models ===
_model = None
_encoder = None

MODEL_PATH = "saved_models/query_based_xgb.joblib"
ENCODER_NAME = "s2593817/sft-sql-embedding"

def get_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = SentenceTransformer(ENCODER_NAME)
    return _encoder

# === Convert implicit joins ===
def convert_implicit_to_explicit_joins(query: str) -> str:
    query = re.sub(r"\s+", " ", query.strip())
    from_match = re.search(r"FROM\s+([a-z0-9_,\s]+)\s+WHERE\s+", query, re.IGNORECASE)
    if not from_match:
        return query

    from_clause = from_match.group(1).strip()
    tables = [t.strip() for t in from_clause.split(",")]

    where_match = re.search(r"WHERE\s+(.+?)(GROUP BY|ORDER BY|LIMIT|$)", query, re.IGNORECASE)
    if not where_match:
        return query

    full_where = where_match.group(1).strip()
    rest_clause = query[where_match.end(1):].strip()

    conditions = [cond.strip() for cond in re.split(r"\bAND\b", full_where, flags=re.IGNORECASE)]

    join_conditions = []
    filter_conditions = []

    for cond in conditions:
        tokens = re.split(r"\s*=\s*", cond)
        if len(tokens) == 2:
            left, right = tokens
            if "." in left and "." in right:
                join_conditions.append(cond)
            else:
                filter_conditions.append(cond)
        else:
            filter_conditions.append(cond)

    if not join_conditions:
        return query

    joins_built = tables[:1]
    join_clause = f"FROM {joins_built[0]}"
    used_tables = {joins_built[0]}

    while len(joins_built) < len(tables):
        for cond in join_conditions:
            left, right = re.split(r"\s*=\s*", cond)
            left_table = left.split(".")[0]
            right_table = right.split(".")[0]

            if (left_table in used_tables) ^ (right_table in used_tables):
                new_table = right_table if left_table in used_tables else left_table
                join_clause += f" JOIN {new_table} ON {cond}"
                joins_built.append(new_table)
                used_tables.add(new_table)
                join_conditions.remove(cond)
                break
        else:
            break

    where_clause = ""
    if filter_conditions:
        where_clause = "WHERE " + " AND ".join(filter_conditions)

    prefix = query[:from_match.start()].strip()
    return f"{prefix} {join_clause} {where_clause} {rest_clause}".strip()

# === Step 1: Parse SQL and extract base and JOIN clauses ===
def extract_base_and_joins(sql):
    sql = sql.strip().rstrip(";")
    match = re.search(r"(FROM .+?)(WHERE|GROUP BY|ORDER BY|LIMIT|$)", sql, re.IGNORECASE | re.DOTALL)
    if not match:
        raise ValueError("Could not extract FROM and JOINs")

    full_from = match.group(1).strip()
    base_match = re.match(r"FROM\s+([^\s]+(?:\s+[a-z]\w*)?)", full_from, re.IGNORECASE)
    if not base_match:
        raise ValueError("Could not find base table in FROM clause")

    base = base_match.group(0)
    rest = full_from[len(base):].strip()

    joins = re.findall(r"(JOIN\s+.+?ON\s+.+?)(?=JOIN|$)", rest, re.IGNORECASE | re.DOTALL)
    return base, joins

# === Step 2: Generate permutations ===
def generate_join_variants(base, joins, max_variants=10):
    if len(joins) <= 5:
        join_orders = list(permutations(joins))
    else:
        join_orders = [random.sample(joins, len(joins)) for _ in range(max_variants)]

    candidates = []
    for join_order in join_orders:
        joined = " ".join(join_order)
        candidates.append(f"{base} {joined}")
    return list(set(candidates))

# === Step 3: Replace original FROM clause ===
def inject_from_clause(original_query, new_from_clause):
    pattern = r"(FROM[\s\S]+?)(WHERE|GROUP BY|ORDER BY|LIMIT|$)"
    match = re.search(pattern, original_query, re.IGNORECASE)
    if not match:
        return original_query

    start = match.start(1)
    end = match.end(1)
    return original_query[:start] + new_from_clause + " " + original_query[end:]

# === Step 4: Predict best variant ===
def predict_best_variant(original_query, max_variants=10, return_all=False, top_k=3):
    # Convert implicit joins
    query = convert_implicit_to_explicit_joins(original_query)

    try:
        base, joins = extract_base_and_joins(query)
    except ValueError as e:
        print(f"Failed to parse query: {e}")
        return original_query, float("inf"), []

    variants = generate_join_variants(base, joins, max_variants)
    full_queries = [inject_from_clause(query, variant) for variant in variants]

    encoder = get_encoder()
    model = get_model()

    features = encoder.encode(full_queries)
    predictions = model.predict(features)

    ranked = sorted(zip(full_queries, predictions), key=lambda x: x[1])
    best_query, best_score = ranked[0]

    if return_all:
        return best_query, best_score, ranked[:top_k]
    return best_query, best_score

# === Batch mode ===
def batch_optimize(input_csv, output_csv, max_variants=20, top_k=3):
    df = pd.read_csv(input_csv)
    results = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Optimizing Queries"):
        query_id = row["query_id"]
        raw_query = row["raw_query"]

        try:
            best_query, best_time, all_variants = predict_best_variant(
                raw_query,
                max_variants=max_variants,
                return_all=True,
                top_k=top_k
            )

            result = {
                "query_id": query_id,
                "original_query": raw_query,
                "optimized_query": best_query,
                "predicted_time": best_time,
            }

            for i, (q, score) in enumerate(all_variants):
                result[f"alt_{i+1}_query"] = q
                result[f"alt_{i+1}_score"] = score

        except Exception as e:
            print(f"[!] Failed to optimize {query_id}: {e}")
            result = {
                "query_id": query_id,
                "original_query": raw_query,
                "optimized_query": raw_query,
                "predicted_time": -1,
            }

        results.append(result)

    pd.DataFrame(results).to_csv(output_csv, index=False)
    print(f"Saved optimized queries to: {output_csv}")


# === CLI Entrypoint ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", type=str, help="Single SQL query to optimize")
    parser.add_argument("--input_csv", type=str, help="CSV file with `query_id` and `raw_query`")
    parser.add_argument("--output_csv", type=str, default="optimized_queries.csv")
    parser.add_argument("--max_variants", type=int, default=20)
    parser.add_argument("--top_k", type=int, default=3)
    args = parser.parse_args()

    if args.input_csv:
        batch_optimize(args.input_csv, args.output_csv, args.max_variants, args.top_k)

    elif args.query:
        best_query, best_time, all_variants = predict_best_variant(
            args.query, max_variants=args.max_variants, return_all=True, top_k=args.top_k
        )
        print("🔍 Optimized Query:")
        print(best_query)
        print(f"Predicted Time: {best_time:.4f} ms\n")

        print("Top Candidates:")
        for q, score in all_variants:
            print(f"{score:.4f} ms | {q.strip()}")

    else:
        query = """
        SELECT o.id, c.name, p.name, o.total
        FROM orders o, products p, customers c
        WHERE o.product_id = p.id AND o.customer_id = c.id AND o.total > 100
        ORDER BY o.id;
        """
        best_query, best_time, all_variants = predict_best_variant(query, max_variants=20, return_all=True)

        print("Optimized Query:")
        print(best_query)
        print(f"Predicted Time: {best_time:.4f} ms")

        print("\nAll Candidates:")
        for q, score in all_variants:
            print(f"{score:.4f} ms  |  {q.strip()}")

