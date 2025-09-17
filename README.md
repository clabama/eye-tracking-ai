# Eye Tracking Processing Notebooks

This repository contains Jupyter notebooks and helper files for **processing and analyzing eye-tracking data** — from raw exports to exploratory statistics and gaze visualizations.

---

## Repository Structure

### Main files

- **labelling.ipynb** – Lists raw files and their renamed counterparts, recording the renaming status for each.
- **labels_per_id.csv** – CSV mapping image IDs to hand-assigned categories and numeric weights.
- **img/** – Stimulus images. Each filename encodes the image ID and category, e.g. `id001_meme_ncc.jpg`.

---

### `analysis_pipeline/`

Modular Python package that powers the confirmatory analysis workflow:

- **data_loading.py** – builds participant-level metric tables from the processed fixation CSVs and merges them with label metadata.
- **metrics.py** – reusable metric computation helpers (fixation counts, scanpath length, pupil metrics, etc.).
- **hierarchy.py** – aggregates metrics hierarchically (all images → single labels → label combinations) with baseline deltas normalised by the 49 participants.
- **clustering.py** – PCA + k-means helpers for discovering label clusters in the metric space.
- **statistics.py** – confirmatory statistical tests (ANOVA, Kruskal-Wallis, pairwise post-hoc comparisons, effect sizes).
- **visualization.py** – Plotly-based plotting utilities for metric deltas, distributions, and clustering views.
- **gui.py** – Streamlit application that exposes the full exploratory-to-confirmatory workflow interactively.

### Example notebooks

- **notebooks/confirmatory_analysis_example.ipynb** – step-by-step demonstration of loading metrics, building hierarchical summaries, running statistical tests, and exploring clustering.

### Legacy notebooks

- The previous exploratory notebooks are still available under `data_analysis/`, `preprocessing/`, and `visualization/` for reference.

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

---

## Confirmatory analysis workflow

1. **Generate metrics** – Ensure the processed fixation CSVs live under `fixations/` and labels are recorded in `labels_per_id.csv`. The new `analysis_pipeline` package will normalise per-participant metrics automatically.
2. **Interactive GUI** – Launch the Streamlit interface:
   ```bash
   streamlit run analysis_pipeline/gui.py
   ```
   The app starts from the full dataset baseline, then lets you drill into single labels and label combinations, visualises metric deltas, displays PCA/k-means clustering, and runs confirmatory statistics with effect sizes.
3. **Notebook workflow** – Open `notebooks/confirmatory_analysis_example.ipynb` to see the same steps scripted in Python for reproducible reports.

The reusable functions allow you to compose additional scripts or dashboards without rewriting metric calculations, making it straightforward to test hypotheses on the existing dataset and future data collections.
