"""Focused tests for validation, cleaning, and per-user clustering."""

import numpy as np
import pandas as pd
import pytest

from src.clustering import cluster_user_pins
from src.data_cleaning import add_temporal_features, clean_coordinates


def frame(rows):
    return pd.DataFrame(rows)


def test_clean_coordinates_reports_only_valid_lima_rows():
    data = frame([
        {"user_id": "a", "start_at": "01/01/2010 10:00", "end_at": "01/01/2010 10:10", "start_lat": -12.0, "start_lon": -77.0, "end_lat": -12.01, "end_lon": -77.01},
        {"user_id": "b", "start_at": "not-a-date", "end_at": "01/01/2010 10:10", "start_lat": -12.0, "start_lon": -77.0, "end_lat": -12.01, "end_lon": -77.01},
        {"user_id": "c", "start_at": "01/01/2010 10:00", "end_at": "01/01/2010 10:10", "start_lat": 140.0, "start_lon": -77.0, "end_lat": -12.01, "end_lon": -77.01},
    ])
    cleaned = clean_coordinates(data)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["user_id"] == "a"


def test_missing_columns_are_named():
    with pytest.raises(ValueError, match="end_lon"):
        clean_coordinates(frame({"user_id": ["a"]}))


def test_empty_cleaned_frame_can_receive_temporal_features():
    columns = ["user_id", "start_at", "end_at", "start_lat", "start_lon", "end_lat", "end_lon"]
    empty = frame({column: pd.Series(dtype="object") for column in columns})
    result = add_temporal_features(clean_coordinates(empty))
    assert result.empty


def test_clustering_is_per_user_and_small_users_are_skipped():
    rows = []
    for user_id, lat in (("a", -12.0), ("b", -12.1)):
        for index in range(2):
            rows.append({"user_id": user_id, "start_lat": lat, "start_lon": -77.0, "end_lat": lat, "end_lon": -77.0})
    rows.append({"user_id": "too-small", "start_lat": -12.0, "start_lon": -77.0, "end_lat": np.nan, "end_lon": np.nan})
    results = cluster_user_pins(frame(rows), method="kmeans", k=5)
    assert {result["user_id"] for result in results} == {"a", "b"}
    assert all(result["n_pins"] == 4 for result in results)
