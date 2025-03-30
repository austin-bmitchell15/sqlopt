import json
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import psycopg2

# === Feature Extraction ===
def extract_features(plan: dict):
    features = {
        "execution_time": plan.get("Execution Time", 0),
        "planning_time": plan.get("Planning Time", 0),
    }

    def walk(node, depth=1):
        stats = {
            "total_rows": node.get("Actual Rows", 0),
            "total_loops": node.get("Actual Loops", 0),
            "node_types": {node.get("Node Type", "Unknown"): 1},
            "max_depth": depth,
            "has_seq_scan": node.get("Node Type") == "Seq Scan",
            "has_sort": node.get("Node Type") == "Sort",
            "has_index_scan": node.get("Node Type") == "Index Scan",
            "workers_launched": node.get("Workers Launched", 0) if "Workers Launched" in node else 0
        }

        for subplan in node.get("Plans", []):
            substats = walk(subplan, depth + 1)

            stats["total_rows"] += substats["total_rows"]
            stats["total_loops"] += substats["total_loops"]
            stats["workers_launched"] += substats["workers_launched"]
            stats["max_depth"] = max(stats["max_depth"], substats["max_depth"])
            stats["has_seq_scan"] |= substats["has_seq_scan"]
            stats["has_sort"] |= substats["has_sort"]
            stats["has_index_scan"] |= substats["has_index_scan"]

            for k, v in substats["node_types"].items():
                stats["node_types"][k] = stats["node_types"].get(k, 0) + v

        return stats

    root = plan["Plan"]
    stats = walk(root)
    features.update(stats)
    return features


# === Execute Query ===
def run_query_task(args):
    qtype, query_file = args
    conn = psycopg2.connect(dbname='tpch', user='postgres', password='password')
    cursor = conn.cursor()

    try:
        if qtype == "view":
            query_id = query_file.stem.replace(".query", "")
            stem_base = query_file.name.replace(".query.sql", "")
            create_file = query_file.with_name(f"{stem_base}.create.sql")
            drop_file = query_file.with_name(f"{stem_base}.drop.sql")

            with conn:
                if create_file.exists():
                    cursor.execute(create_file.read_text())

                query = query_file.read_text()
                cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
                plan_json = cursor.fetchone()[0][0]

                if drop_file.exists():
                    cursor.execute(drop_file.read_text())

        else:
            query_id = query_file.stem
            query = query_file.read_text()
            with conn:
                cursor.execute(f"EXPLAIN (ANALYZE, FORMAT JSON) {query}")
                plan_json = cursor.fetchone()[0][0]

        return {
            "query_id": query_id,
            "query": query,
            "plan": plan_json
        }

    except Exception as e:
        return {"error": str(e), "query_file": str(query_file)}
    finally:
        cursor.close()
        conn.close()


# === Write to CSVs ===
def append_to_csv(features, metadata, features_file, metadata_file):
    pd.DataFrame(features).to_csv(features_file, mode='a', header=not Path(features_file).exists(), index=False)
    pd.DataFrame(metadata).to_csv(metadata_file, mode='a', header=not Path(metadata_file).exists(), index=False)

# === Main Runner ===
def run_in_batches(batch_size=10, max_workers=4, timeout_sec=60):
    base_dir = Path("./data/processed_queries")
    features_file = "./data/training_data.csv"
    metadata_file = "./data/query_metadata.csv"

    # === Load processed query_ids ===
    processed_ids = set()
    if Path(features_file).exists():
        try:
            df = pd.read_csv(features_file, usecols=["query_id"])
            processed_ids = set(df["query_id"].astype(str))
        except Exception as e:
            print(f"Warning: Failed to load existing training_data.csv: {e}")

    all_queries = []
    skipped_families = set()

    for query_dir in sorted(base_dir.iterdir()):
        if not query_dir.is_dir():
            continue

        for query_file in sorted(query_dir.glob("query*.query.sql")):
            query_id = query_file.stem.replace(".query", "")
            base_id = query_id.split(".")[0]
            if query_id not in processed_ids:
                all_queries.append(("view", query_file, query_id, base_id))

        for flat_query in query_dir.glob("query*.sql"):
            if any(suffix in flat_query.name for suffix in [".query.sql", ".create.sql", ".drop.sql"]):
                continue
            query_id = flat_query.stem
            base_id = query_id.split(".")[0]
            if query_id not in processed_ids:
                all_queries.append(("flat", flat_query, query_id, base_id))

    print(f"{len(processed_ids)} queries already processed.")
    print(f"{len(all_queries)} queries left to process.")

    if not all_queries:
        print("All queries already processed.")
        return

    # === Process in batches ===
    for i in range(0, len(all_queries), batch_size):
        batch = [
            (qtype, qfile)
            for qtype, qfile, query_id, base_id in all_queries[i:i + batch_size]
            if base_id not in skipped_families
        ]

        if not batch:
            continue

        features, metadata = [], []

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(run_query_task, args): args for args in batch
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Batch {i//batch_size+1}", unit="query"):
                args = futures[future]
                _, qfile = args
                query_id = qfile.stem.replace(".query", "")
                base_id = query_id.split(".")[0]

                try:
                    result = future.result(timeout=timeout_sec)

                    if "error" in result:
                        print(f"Error in {result['query_file']}: {result['error']}")
                        continue

                    f = extract_features(result["plan"])
                    f["query_id"] = result["query_id"]
                    features.append(f)

                    metadata.append({
                        "query_id": result["query_id"],
                        "raw_query": result["query"]
                    })

                except Exception as e:
                    print(f"Timeout or failure in {query_id} — skipping all of {base_id}")
                    skipped_families.add(base_id)

        append_to_csv(features, metadata, features_file, metadata_file)

    print("All remaining queries processed and written to CSV.")
    print(f"Skipped query families: {sorted(skipped_families)}")



# === Start the script ===
if __name__ == "__main__":
    run_in_batches(batch_size=128, max_workers=8)

