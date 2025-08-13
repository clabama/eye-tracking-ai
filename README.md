# Eye Tracking Processing Notebooks

This repository contains Jupyter notebooks for processing and analysing eye-tracking data. The notebooks form a pipeline from raw data exports to various visualisations.

## Notebook overview

- **`data_processed.ipynb`** – Reads raw CSV files, converts timestamps to milliseconds, filters and interpolates gaze data, normalises pupil size and detects fixations using the I‑DT algorithm. The cleaned files are stored as `*_processed.csv`; detected fixations are saved under `processed/fixations` as `*_fixations.csv`.
- **`heatmaps.ipynb`** – Builds heatmaps from the fixation files (`x`, `y` columns), smoothing them with a Gaussian kernel. Resulting images are saved to `processed/heatmaps`. The notebook also flattens all heatmaps, reduces them via PCA and performs hierarchical clustering to visualise clusters and their average heatmaps.
- **`scanpaths.ipynb`** – Loads each `*_fixations.csv`, sorts fixations by start time and plots the temporal sequence of gaze positions. The generated `*_scanpath.png` images are written to `processed/scanpaths`.
- **`pupilsize_exploration.ipynb`** – Aggregates normalised pupil sizes from the fixation files by image category and outputs statistical summaries and boxplots (visualisation only, no files written).
- **`names_fixed.ipynb`** – Scans the `raw` directory for files matching `ProbandXX_idYY`, renaming them to `PXXX_idYY_kategorie` while avoiding number collisions. Trailing underscore files are removed.
- **`renaming.ipynb`** – Currently empty; reserved for future renaming utilities.

## Usage
Each notebook can be executed independently to perform its respective step. Outputs are written to the `processed/` subdirectories as described above.

