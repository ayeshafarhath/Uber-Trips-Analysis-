# Uber Trips Analysis - Frequent Pins Recommendation

> Clustering 23,111 Uber trips from Lima, Peru (2010) to identify frequent locations. Compares DBSCAN (density-based) vs K-Means for location recommendation.

**Problem:** Riders type the same addresses repeatedly. Can we auto-detect 2-4 frequent pins (home, office, gym) from GPS history to reduce typing time?

**Solution:** Stack start + end pins per user, cluster with DBSCAN (100m radius, haversine metric). DBSCAN is primary because it handles noise (one-off trips) and doesn't require pre-defining k. K-Means is used as baseline.

### Dataset
- **Source:** Uber Peru 2010 (Lima)
- **Size:** 23,111 trips, 1,390 users (avg 16.6 rides/user)
- **Format quirks:** `;` delimiter, `,` decimal (European) - handled in `src/data_cleaning.py`
- **Fields:** journey_id, user_id, start_at/end_at, start/end lat/lon, distance, duration, price, source
- Place `uber_peru_2010.csv` in `data/` locally (not committed)

### Tech Stack
Python, Pandas (vectorized, no apply), NumPy, Scikit-learn (DBSCAN, KMeans), Plotly (mapbox), tqdm

### Project Workflow
1. Load with correct delimiter/decimal handling
2. Clean: vectorized lat/lon validation + Lima bbox filter (-12.5 to -11.8, -77.2 to -76.5) removes GPS outliers
3. EDA: temporal patterns by hour/weekday
4. Clustering: per-user pins -> DBSCAN (eps=100m / earth_radius, min_samples=2, haversine)
5. Comparison: K-Means k=5 forced vs DBSCAN auto clusters
6. Visualization: scatter_mapbox with centroids

### Key Findings
- DBSCAN finds 2-4 frequent locations per user vs K-Means forcing 5
- Noise handling: DBSCAN marks 10-15% one-off trips as -1, K-Means cannot
- Most frequent pins align with commuter pattern (home/office)
- Peak demand observed in evening hours (see notebook)

### Clustering Methodology

**Why DBSCAN:**
- Density-based, no k needed
- eps=100m chosen from Snap's frequent pins research - walkable distance for same place
- min_samples=2 = at least one return trip
- haversine metric for GPS distance
- Handles irregular street shapes

**Why K-Means as baseline:**
- Centroid-based, requires k
- k=5 arbitrary, fails when user has <5 frequent places, forces outliers into clusters
- Used to show why DBSCAN is better for this use case

**No GMM / PCA claimed** - Not needed for GPS pins. GMM would add probabilistic soft assignment but with 2D lat/lon and clear density, DBSCAN is more interpretable.

### Project Structure

## Project Structure

```text
Uber-Trips-Analysis-/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md
├── notebooks/
│   └── 01_frequent_pins_analysis.ipynb
├── src/
│   ├── data_cleaning.py
│   ├── clustering.py
│   └── visualization.py
└── outputs/
    └── figures/
```

### Installation & Usage
```bash
git clone https://github.com/ayeshafarhath/uber-trips-analysis.git
cd uber-trips-analysis
pip install -r requirements.txt
# Place uber_peru_2010.csv in data/
jupyter notebook notebooks/01_frequent_pins_analysis.ipynb
```

## Limitations

- **Single city & old data:** 23k trips only from Lima (2010) — not generalizable to current global Uber usage
- **No ground truth:** No labeled "home/office" data, so evaluation is visual & distribution-based, not accuracy-based
- **GPS noise:** Up to ~100m error in start/end pins, can merge nearby locations
- **Feature scope:** Price/distance/time deliberately excluded to keep clustering purely geospatial (for interview clarity)

## Future Work

- **Empirical eps tuning:** Add k-distance elbow plot to justify eps=100m statistically, not just research-based
- **Quantitative comparison:** Silhouette score & Davies-Bouldin Index for DBSCAN vs K-Means (k=5)
- **Temporal layer:** Separate weekday vs weekend pins to detect work vs home patterns
- **Business extension:** Use price + distance features for trip segmentation (short commutes vs airport trips)
- **Scale:** Test on multi-city dataset & deploy as Streamlit app with interactive map

Author

Ayesha Farhath- B.Tech CSE (AI & ML) - Data Analyst / Junior Data Scientist
