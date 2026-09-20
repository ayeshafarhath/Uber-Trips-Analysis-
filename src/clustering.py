"""Cluster frequent start and end locations for individual users."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import DBSCAN, KMeans

EARTH_RADIUS_M = 6_371_000
DEFAULT_EPS_METERS = 100
DEFAULT_MIN_SAMPLES = 2


def get_user_pins(rides_df):
    """Return user, latitude, and longitude columns for all trip endpoints."""
    required = {"user_id", "start_lat", "start_lon", "end_lat", "end_lon"}
    missing = sorted(required - set(rides_df.columns))
    if missing:
        raise ValueError(f"Dataframe is missing required columns: {', '.join(missing)}")
    starts = rides_df[["user_id", "start_lat", "start_lon"]].rename(
        columns={"start_lat": "lat", "start_lon": "lon"}
    )
    ends = rides_df[["user_id", "end_lat", "end_lon"]].rename(
        columns={"end_lat": "lat", "end_lon": "lon"}
    )
    return np.concatenate([starts.to_numpy(), ends.to_numpy()], axis=0)


def cluster_pins_dbscan(
    pins_deg: np.ndarray,
    *,
    eps_meters: float = DEFAULT_EPS_METERS,
    min_samples: int = DEFAULT_MIN_SAMPLES,
):
    """Cluster latitude/longitude points with DBSCAN and haversine distance."""
    points = np.asarray(pins_deg, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pins_deg must have shape (n_points, 2)")
    if eps_meters <= 0 or min_samples < 1:
        raise ValueError("eps_meters must be positive and min_samples must be at least one")
    if len(points) == 0:
        return {"labels": np.array([], dtype=int), "centroids": np.empty((0, 2))}

    model = DBSCAN(
        eps=eps_meters / EARTH_RADIUS_M,
        min_samples=min_samples,
        metric="haversine",
    )
    labels = model.fit_predict(np.radians(points))
    centroids = np.array(
        [points[labels == label].mean(axis=0) for label in sorted(set(labels)) if label != -1]
    )
    if centroids.size == 0:
        centroids = np.empty((0, 2))
    return {"labels": labels, "centroids": centroids}


def cluster_pins_kmeans(pins_deg: np.ndarray, *, k: int = 5):
    """Run K-Means as a deliberately simple fixed-k comparison baseline."""
    points = np.asarray(pins_deg, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pins_deg must have shape (n_points, 2)")
    if len(points) == 0:
        return {"labels": np.array([], dtype=int), "centroids": np.empty((0, 2))}
    if k < 1:
        raise ValueError("k must be positive")
    k = min(k, len(points))
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(np.radians(points))
    return {"labels": labels, "centroids": np.degrees(model.cluster_centers_)}


def cluster_user_pins(rides_df, *, method="dbscan", eps_meters=DEFAULT_EPS_METERS, min_samples=DEFAULT_MIN_SAMPLES, k=5):
    """Return one clustering result per user, skipping users with no usable pins."""
    results = []
    for user_id, user_rides in rides_df.groupby("user_id", sort=True):
        pins = get_user_pins(user_rides)[:, 1:].astype(float)
        if len(pins) < 2:
            continue
        if method == "dbscan":
            result = cluster_pins_dbscan(pins, eps_meters=eps_meters, min_samples=min_samples)
        elif method == "kmeans":
            result = cluster_pins_kmeans(pins, k=k)
        else:
            raise ValueError("method must be 'dbscan' or 'kmeans'")
        result["user_id"] = user_id
        result["n_pins"] = len(pins)
        result["n_clusters"] = len(result["centroids"])
        result["noise_points"] = int(np.sum(result["labels"] == -1))
        results.append(result)
    return results
