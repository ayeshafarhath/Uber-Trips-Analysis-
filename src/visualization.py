"""Create Plotly maps from one user's clustering result."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px


def plot_user_pins(
    pins: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    *,
    title: str = "Frequent Pins",
):
    """Plot pins and cluster centroids for one user.

    ``labels`` must align with ``pins``. DBSCAN noise labels of ``-1`` are kept
    in the input so that they remain visible as their own plotted category.
    Empty inputs return an empty, valid Plotly figure.
    """
    points = np.asarray(pins, dtype=float)
    cluster_labels = np.asarray(labels)
    cluster_centroids = np.asarray(centroids, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("pins must have shape (n_points, 2)")
    if cluster_labels.ndim != 1 or len(cluster_labels) != len(points):
        raise ValueError("labels must contain one value per pin")
    if cluster_centroids.size == 0:
        cluster_centroids = np.empty((0, 2), dtype=float)
    if cluster_centroids.ndim != 2 or cluster_centroids.shape[1] != 2:
        raise ValueError("centroids must have shape (n_clusters, 2)")

    frame = pd.DataFrame(points, columns=["lat", "lon"])
    frame["cluster"] = cluster_labels.astype(str)
    figure = px.scatter_map(
        frame,
        lat="lat",
        lon="lon",
        color="cluster",
        zoom=12,
        height=600,
        title=title,
        map_style="carto-positron",
    )
    if len(cluster_centroids):
        centroid_frame = pd.DataFrame(cluster_centroids, columns=["lat", "lon"])
        figure.add_scattermap(
            lat=centroid_frame["lat"],
            lon=centroid_frame["lon"],
            mode="markers",
            marker={"size": 14, "color": "black", "symbol": "star"},
            name="centroid",
        )
    return figure
