Eye Tracking Processing Notebooks
This repository contains Jupyter notebooks and helper files for processing and analysing eye‑tracking data—from raw exports to exploratory statistics and gaze visualisations.

Repository structure
labelling.ipynb – Lists raw files and their renamed counterparts, recording the renaming status for each.

labels_per_id.csv – CSV mapping image IDs to hand‑assigned categories and numeric weights.

img/ – Stimulus images. Each filename encodes the image ID and category, e.g. id001_meme_ncc.jpg.

data_analysis/
pupilsize_exploration.ipynb – Aggregates normalised pupil sizes from the fixation files by image category and outputs statistical summaries and boxplots (visualisation only, no files written).

preprocessing/
data_processed.ipynb – Reads raw CSV files, converts timestamps to milliseconds, filters and interpolates gaze data, normalises pupil size and detects fixations using the I‑DT algorithm. The cleaned files are stored as *_processed.csv; detected fixations are saved under processed/fixations as *_fixations.csv.

names_fixed.ipynb – Scans the raw directory for files matching ProbandXX_idYY, renaming them to PXXX_idYY_kategorie while avoiding number collisions. Trailing underscore files are removed.

renaming.ipynb – Currently empty; reserved for future renaming utilities.

visualization/
heatmaps.ipynb – Builds heatmaps from the fixation files (x, y columns), smoothing them with a Gaussian kernel. Resulting images are saved to processed/heatmaps. The notebook also flattens all heatmaps, reduces them via PCA and performs hierarchical clustering to visualise clusters and their average heatmaps.

scanpaths.ipynb – Loads each *_fixations.csv, sorts fixations by start time and plots the temporal sequence of gaze positions. The generated *_scanpath.png images are written to processed/scanpaths.

heatmaps/ – Directory of generated heatmap images.

scanpaths/ – Directory of generated scanpath visualisations.

Usage
Each notebook can be executed independently to perform its step in the pipeline. Run them within Jupyter Notebook or JupyterLab; outputs are written to the respective processed/ subdirectories noted above.
