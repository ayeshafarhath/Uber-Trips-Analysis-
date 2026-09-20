# Uber Trips Analysis

A geospatial clustering project exploring how repeated trip endpoints can be grouped into frequent user locations using DBSCAN and a K-Means baseline.

This repository focuses on per-user location clustering from ride data and is designed as a practical data-analysis project rather than a production recommendation system.

## Problem

Riders often repeat the same places over time. This project tests whether historical trip endpoints can be grouped into recurring locations without needing to predefine the number of clusters.

## Approach

The pipeline performs the following steps:

1. Load trip data and normalize column names
2. Validate numeric coordinates and timestamps
3. Remove invalid or out-of-range rows
4. Restrict analysis to the Lima bounding box
5. Stack start and end coordinates for each user
6. Run DBSCAN with haversine distance
7. Run a fixed-k K-Means baseline for comparison
8. Output per-user clustering summaries

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- Plotly
- Jupyter-style exploratory workflow

## Project Structure

```text
src/
  data_cleaning.py
  clustering.py
  visualization.py
run_analysis.py
requirements.txt
tests/
  test_pipeline.py
data/
  sample_raw.csv
```

## Running the project

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the sample pipeline:

```bash
python run_analysis.py --input data/sample_raw.csv --output-dir outputs/
```

Run tests:

```bash
pytest -q
```

## Notes

- This repository does not include the original Uber Peru dataset.
- No ground-truth labels are provided for “correct” locations.
- DBSCAN is used as the primary method because it handles noise and does not require a predetermined cluster count.
- K-Means is included only as a comparison baseline and is not treated as the more correct method.

## Limitations

- The project is limited to a single city dataset.
- 100m clustering can merge nearby places or split noisy points.
- A cluster is a repeated location pattern, not automatically a named place such as home or office.
- This project is exploratory and not a production recommendation service.
