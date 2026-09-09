"""
clustering.py - Frequent pins clustering
DBSCAN is primary (density-based, handles noise, auto k)
K-Means is baseline comparison
"""
import numpy as np
from sklearn.cluster import DBSCAN, KMeans

EARTH_RADIUS_M = 6371000
EPS_METERS = 100  # 100m radius as per Snap research
EPS_RAD = EPS_METERS / EARTH_RADIUS_M

def get_user_pins(rides_df):
    """Vectorized stacking of start + end pins"""
    start = rides_df[['start_lat','start_lon']].to_numpy()
    end = rides_df[['end_lat','end_lon']].to_numpy()
    return np.vstack([start, end])  # shape (2*n_rides, 2)

def cluster_pins_dbscan(pins_deg: np.ndarray):
    """pins in degrees"""
    if len(pins_deg) < 2:
        return {"labels": np.array([-1]*len(pins_deg)), "centroids": pins_deg}
    pins_rad = np.radians(pins_deg)
    db = DBSCAN(eps=EPS_RAD, min_samples=2, metric='haversine')
    labels = db.fit_predict(pins_rad)
    centroids = []
    for lbl in set(labels):
        if lbl == -1: 
            continue
        centroids.append(pins_deg[labels==lbl].mean(axis=0))
    return {"labels": labels, "centroids": np.array(centroids) if centroids else np.empty((0,2))}

def cluster_pins_kmeans(pins_deg: np.ndarray, k=5):
    if len(pins_deg) == 0:
        return {"labels": np.array([]), "centroids": np.empty((0,2))}
    k = min(k, len(pins_deg))
    pins_rad = np.radians(pins_deg)
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = km.fit_predict(pins_rad)
    centroids_rad = km.cluster_centers_
    centroids_deg = np.degrees(centroids_rad)
    return {"labels": labels, "centroids": centroids_deg}
