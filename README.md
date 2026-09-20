# Uber Trips Analysis: frequent locations per user

This project explores whether repeated start and end locations in an Uber trip history can be grouped into useful frequent-location candidates for each user.

It is an exploratory geospatial analysis, not a production recommendation system. The repository does not include the original dataset, so the analysis cannot be reproduced until the data is obtained from its source.

## Problem

A rider may repeatedly use the same locations. The analysis tests whether simple clustering of historical trip endpoints can identify those repeated locations without assigning labels such as `home` or `office`.

## Dataset

The code expects the Uber Peru 2010 export as a semicolon-delimited CSV with European decimal commas. Obtain the dataset from its permitted source and place it somewhere outside version control, for example:

```text
data/uber_peru_2010.csv
```

The required columns are:

- `user_id`
- `start_at`, `end_at`
- `start_lat`, `start_lon`
- `end_lat`, `end_lon`

Other columns are retained when present. Do not commit the dataset unless its license permits redistribution.

## Approach

1. Read the export using its delimiter and decimal conventions.
2. Validate required columns and numeric coordinate values.
3. Remove rows with missing timestamps or endpoint coordinates.
4. Remove coordinates outside valid latitude/longitude ranges.
5. Restrict the analysis to the Lima bounding box used by this dataset.
6. Stack start and end points for each user.
7. Run DBSCAN separately for each user using haversine distance.
8. Run fixed-`k=5` K-Means as a simple comparison baseline, not as an objectively correct choice.
9. Write per-user summaries to `outputs/`.

## Clustering methodology

DBSCAN is useful here because it can mark isolated points as noise and does not require the number of locations to be known in advance. Coordinates are converted to radians and compared with the haversine metric. The default `eps` is 100 metres and `min_samples` is 2; these are explicit heuristics, not experimentally proven optimal values.

K-Means is included only as a fixed-`k` baseline. It forces every point into a cluster and is therefore not expected to represent irregular or noisy location histories as naturally as DBSCAN.

## Results

Results are generated locally rather than written in advance. Run the analysis and inspect:

- `outputs/dbscan_user_summary.csv`
- `outputs/kmeans_user_summary.csv`

The script prints the number of loaded and removed rows and summary statistics. This repository does not claim a particular cluster count, noise percentage, commuting pattern, or demand pattern without a committed run output that supports it.

There is no ground-truth label for the correct frequent locations, so this project does not report classification accuracy. Any future clustering comparison should define a valid user-level evaluation before reporting a single aggregate metric.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_analysis.py data/uber_peru_2010.csv
```

On Windows, activate the environment with `.venv\\Scripts\\activate`.

## Limitations

- The analysis is limited to one historical city dataset.
- A 100-metre radius can merge nearby places or split the same place when GPS noise is high.
- A cluster is only a repeated geographic location; it is not automatically a meaningful place label.
- Users with too few endpoints are skipped.
- The current script writes summaries but does not claim a validated recommendation-quality measure.

## Project structure

```text
src/
  data_cleaning.py
  clustering.py
  visualization.py
run_analysis.py
requirements.txt
```
