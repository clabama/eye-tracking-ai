# Eye Tracking Processing Notebooks

This repository contains Jupyter notebooks and helper files for **processing and analyzing eye-tracking data** — from raw exports to exploratory statistics and gaze visualizations.

---

## Repository Structure

### Main files

- **labelling.ipynb** – Lists raw files and their renamed counterparts, recording the renaming status for each.
- **labels_per_id.csv** – CSV mapping image IDs to hand-assigned categories and numeric weights.
- **img/** – Stimulus images. Each filename encodes the image ID and category, e.g. `id001_meme_ncc.jpg`.

---

### `data_analysis/`

- **pupilsize_exploration.ipynb**  
  Aggregates normalized pupil sizes from the fixation files by image category and outputs statistical summaries and boxplots.  
  _(Visualization only, no files written)._

---

### `preprocessing/`

- **data_processed.ipynb**

  - Reads raw CSV files
  - Converts timestamps to milliseconds
  - Filters and interpolates gaze data
  - Normalizes pupil size
  - Detects fixations using the I-DT algorithm
  - Saves results:
    - Cleaned files as `*_processed.csv`
    - Fixations under `processed/fixations/` as `*_fixations.csv`

- **names_fixed.ipynb** – Scans the raw directory for files matching `ProbandXX_idYY`, renames them to `PXXX_idYY_category` while avoiding number collisions. Trailing underscore files are removed.
- **renaming.ipynb** – Currently empty; reserved for future renaming utilities.

---

### `visualization/`

- **heatmaps.ipynb**  
  Builds heatmaps from fixation files (`x`, `y` columns), smoothed with a Gaussian kernel.

  - Outputs to `processed/heatmaps/`
  - Flattens all heatmaps, reduces them via PCA, and performs hierarchical clustering to visualize clusters and their average heatmaps.

- **scanpaths.ipynb**  
  Loads each `*_fixations.csv`, sorts fixations by start time, and plots the temporal sequence of gaze positions.

  - Outputs to `processed/scanpaths/` as `*_scanpath.png`

- **heatmaps/** – Directory of generated heatmap images.
- **scanpaths/** – Directory of generated scanpath visualizations.

---

## Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Installation

1. **Clone the repository:**

   ```bash
   git clone <repository-url>
   cd eye-tracking-ai
   ```

2. **Create a virtual environment:**

   ```bash
   python -m venv .venv
   ```

3. **Activate the virtual environment:**

   - On Windows:
     ```bash
     .venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

5. **Launch Jupyter:**
   ```bash
   jupyter notebook
   ```
   or
   ```bash
   jupyter lab
   ```

---

## Usage

Each notebook can be executed independently to perform its step in the pipeline.  
Run them within **Jupyter Notebook** or **JupyterLab**; outputs are written to the respective `processed/` subdirectories.
