"""
Prints subject x length_bucket, subject x variant_type, and subject x data_source
crosstabs for train/test, and flags any cell with fewer than 5 examples so gaps
are known explicitly rather than discovered silently in a benchmark run.

Run after scripts/build_dataset_manifest.py:
    python scripts/data_quality_report.py
"""
import json
import os

import pandas as pd


def load_all(split):
    rows = []
    for fname in sorted(os.listdir(f"data/{split}")):
        if fname.endswith(".json"):
            with open(f"data/{split}/{fname}") as f:
                rows.append(json.load(f))
    return rows


def main():
    with open("data/raw/rubrics.json") as f:
        rubrics = {r["question_id"]: r for r in json.load(f)}

    train_rows = load_all("train")
    test_rows = load_all("test")
    for r in train_rows:
        r["subject"] = rubrics[r["question_id"]]["subject"]
    for r in test_rows:
        r["subject"] = rubrics[r["question_id"]]["subject"]

    train_df = pd.DataFrame(train_rows)
    test_df = pd.DataFrame(test_rows)

    print(f"Train examples: {len(train_df)} | Test examples: {len(test_df)}\n")

    print("=== TRAIN: subject x length_bucket ===")
    print(pd.crosstab(train_df["subject"], train_df["length_bucket"]))

    print("\n=== TRAIN: subject x data_source ===")
    print(pd.crosstab(train_df["subject"], train_df["data_source"]))

    print("\n=== TEST: subject x variant_type ===")
    print(pd.crosstab(test_df["subject"], test_df["variant_type"]))

    print("\n=== TEST: subject x length_bucket ===")
    print(pd.crosstab(test_df["subject"], test_df["length_bucket"]))

    print("\n=== TEST: subject x data_source ===")
    print(pd.crosstab(test_df["subject"], test_df["data_source"]))

    print("\n=== Under-populated cells (<5 examples) ===")
    checks = [
        ("train subject x length_bucket", train_df, ["subject", "length_bucket"]),
        ("test subject x variant_type", test_df, ["subject", "variant_type"]),
        ("test subject x length_bucket", test_df, ["subject", "length_bucket"]),
    ]
    for name, df, dims in checks:
        counts = df.groupby(dims).size()
        thin = counts[counts < 5]
        if len(thin):
            print(f"-- {name} --")
            print(thin)
        else:
            print(f"-- {name}: no cells under 5 --")


if __name__ == "__main__":
    main()
