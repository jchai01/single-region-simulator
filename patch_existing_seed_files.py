"""
One-time patch for scenario_details_seed{N}.parquet files generated
BEFORE the export_scenario_details() fix (missing "method" column,
and scenario_id was just an integer rank that collides across
different seeds' files once concatenated).

Run this ONCE against your existing 4 files -- no need to rerun the
actual NSGA-II search, which is the expensive part. Safe to run
multiple times (it's a no-op if a file is already patched).

Usage: python patch_existing_seed_files.py
(run from the same directory containing your data/ folder)
"""
import glob
import re
import pandas as pd

PATTERN = "data/scenario_details_seed*.parquet"


def infer_seed_number(path):
    match = re.search(r"seed(\d+)\.parquet$", path)
    if not match:
        raise ValueError(f"Could not infer seed number from filename: {path}")
    return int(match.group(1))


def patch_file(path):
    seed = infer_seed_number(path)
    df = pd.read_parquet(path)

    if "method" in df.columns and df["scenario_id"].astype(str).str.startswith("seed").all():
        print(f"{path}: already patched, skipping.")
        return

    df["method"] = "nsga2"
    # Original scenario_id was an integer rank (0, 1, 2...) -- collides
    # across different seeds' files once concatenated. Prefix with the
    # seed number, inferred from the filename, to make it unique.
    df["scenario_id"] = df["scenario_id"].apply(lambda rank: f"seed{seed}_{rank}")

    df.to_parquet(path, index=False)
    print(f"{path}: patched ({len(df)} rows, seed={seed}).")


if __name__ == "__main__":
    files = sorted(glob.glob(PATTERN))
    if not files:
        print(f"No files matched {PATTERN} -- check you're running this from "
              "the directory containing your data/ folder.")
    for f in files:
        patch_file(f)
