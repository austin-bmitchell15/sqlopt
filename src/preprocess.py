import re
from pathlib import Path
from tqdm import tqdm

src_dir = Path("./postgres-tpch/generated_queries")
dst_root = Path("./data/processed_queries")
dst_root.mkdir(exist_ok=True)

# List of all SQL files
files = list(src_dir.glob("query*.sql"))

for file in tqdm(files, desc="Preprocessing", unit="query"):
    filename = file.stem  # e.g., query15.2

    # Extract base query ID (e.g., query15 from query15.2)
    base_query = filename.split(".")[0]
    query_subdir = dst_root / base_query
    query_subdir.mkdir(exist_ok=True)

    # Read and preprocess text
    text = file.read_text()

    # Doing some preprocessing that might otherwise break PostgreSQL
    text = re.sub(r"interval '(\d+)' day\s*\(\d+\)", r"interval '\1 days'", text)
    text = re.sub(r"\bFIRST\s+-?\d+", "", text)
    text = text.replace("\r\n", "\n")

    # Check for view usage
    if "create view" in text.lower():
        create_view = re.search(r"(create\s+view.*?;)", text, re.IGNORECASE | re.DOTALL)
        drop_view = re.search(r"(drop\s+view.*?;)", text, re.IGNORECASE | re.DOTALL)

        # If there are views we need to split the query into 3 parts
        # create.sql, query.sql, and drop.sql
        if create_view and drop_view:
            create_sql = create_view.group(1).strip()
            drop_sql = drop_view.group(1).strip()
            between = text[text.index(create_sql) + len(create_sql): text.index(drop_sql)].strip()

            # Write parts
            (query_subdir / f"{filename}.create.sql").write_text(create_sql + "\n")
            (query_subdir / f"{filename}.query.sql").write_text(between + "\n")
            (query_subdir / f"{filename}.drop.sql").write_text(drop_sql + "\n")
        else:
            print(f"Skipped {filename}: View query could not be split cleanly.")
    else:
        # Save full cleaned query as-is
        (query_subdir / f"{filename}.sql").write_text(text)

