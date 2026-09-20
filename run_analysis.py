"""Run the Uber frequent-pins analysis and report cleaning counts."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.clustering import cluster_user_pins
from src.data_cleaning import add_temporal_features, clean_coordinates, load_uber_data


def summarize(results):
    """Create a small summary dataframe from per-user clustering results."""
    if not results:
        return pd.DataFrame(columns=["user_id", "n_pins", "n_clusters", "noise_points"])
    return pd.DataFrame(
        [
            {key: result[key] for key in ("user_id", "n_pins", "n_clusters", "noise_points")}
            for result in results
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="path to the Uber Peru CSV export")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    raw = load_uber_data(args.dataset)
    clean = add_temporal_features(clean_coordinates(raw))
    print(f"Rows loaded: {len(raw)}")
    print(f"Rows removed during cleaning: {len(raw) - len(clean)}")
    print(f"Users retained: {clean['user_id'].nunique()}")

    dbscan = summarize(cluster_user_pins(clean, method="dbscan"))
    kmeans = summarize(cluster_user_pins(clean, method="kmeans", k=5))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    dbscan.to_csv(args.output_dir / "dbscan_user_summary.csv", index=False)
    kmeans.to_csv(args.output_dir / "kmeans_user_summary.csv", index=False)

    print("DBSCAN summary:")
    print(dbscan.describe(include="all") if not dbscan.empty else "No users had enough pins")
    print("K-Means summary:")
    print(kmeans.describe(include="all") if not kmeans.empty else "No users had enough pins")
    print("No accuracy metric is reported: the dataset has no labels for correct frequent locations.")


if __name__ == "__main__":
    main()
