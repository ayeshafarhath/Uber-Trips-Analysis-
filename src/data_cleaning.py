"""Load and validate the Uber Peru trip export."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

LIMA_BBOX = {
    "lat_min": -12.5,
    "lat_max": -11.8,
    "lon_min": -77.2,
    "lon_max": -76.5,
}

REQUIRED_COLUMNS = {
    "user_id",
    "start_at",
    "end_at",
    "start_lat",
    "start_lon",
    "end_lat",
    "end_lon",
}


def load_uber_data(path: str | Path) -> pd.DataFrame:
    """Read the semicolon-delimited export and validate its required columns."""
    data_path = Path(path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Dataset does not exist: {data_path}")

    frame = pd.read_csv(data_path, sep=";", decimal=",", low_memory=False)
    frame.columns = [str(column).strip().strip("'") for column in frame.columns]
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")
    return frame


def clean_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Clean coordinates and timestamps while preserving a predictable schema.

    Rows with missing or invalid coordinates, or unparseable timestamps, are removed.
    The caller can compare lengths before and after this function to report removals.
    """
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError(f"Dataframe is missing required columns: {', '.join(missing)}")

    preferred = [
        "journey_id", "user_id", "start_at", "end_at", "start_lat", "start_lon",
        "end_lat", "end_lon", "end_state", "distance", "duration",
    ]
    columns = [column for column in preferred if column in df.columns]
    clean = df.loc[:, columns].copy()

    coordinate_columns = ["start_lat", "start_lon", "end_lat", "end_lon"]
    for column in coordinate_columns:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    clean["start_at"] = pd.to_datetime(clean["start_at"], dayfirst=True, errors="coerce")
    clean["end_at"] = pd.to_datetime(clean["end_at"], dayfirst=True, errors="coerce")

    clean = clean.dropna(subset=coordinate_columns + ["start_at", "end_at"])
    valid_range = (
        clean["start_lat"].between(-90, 90)
        & clean["start_lon"].between(-180, 180)
        & clean["end_lat"].between(-90, 90)
        & clean["end_lon"].between(-180, 180)
    )
    clean = clean.loc[valid_range]

    in_lima = (
        clean["start_lat"].between(LIMA_BBOX["lat_min"], LIMA_BBOX["lat_max"])
        & clean["start_lon"].between(LIMA_BBOX["lon_min"], LIMA_BBOX["lon_max"])
        & clean["end_lat"].between(LIMA_BBOX["lat_min"], LIMA_BBOX["lat_max"])
        & clean["end_lon"].between(LIMA_BBOX["lon_min"], LIMA_BBOX["lon_max"])
    )
    return clean.loc[in_lima].reset_index(drop=True)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple time features to a cleaned trip dataframe."""
    if "start_at" not in df or not pd.api.types.is_datetime64_any_dtype(df["start_at"]):
        raise ValueError("start_at must be parsed datetime values before adding features")
    result = df.copy()
    result["hour"] = result["start_at"].dt.hour
    result["weekday"] = result["start_at"].dt.day_name()
    result["is_weekend"] = result["start_at"].dt.weekday >= 5
    return result
