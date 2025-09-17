"""Metric computation utilities for fixation-level data."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FixationMetricConfig:
    """Configuration for metric computation."""

    min_required_fixations: int = 1
    coordinate_columns: Iterable[str] = ("x", "y")
    duration_column: str = "duration"
    pupil_size_column: str = "avg_pupil_size"
    pupil_norm_column: str = "pupil_size_norm"


def _scanpath_length(values: pd.DataFrame, coordinate_columns: Iterable[str]) -> float:
    coords = values.loc[:, list(coordinate_columns)].to_numpy(dtype=float, copy=True)
    if len(coords) < 2:
        return 0.0
    deltas = np.diff(coords, axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    return float(distances.sum())


def compute_fixation_metrics(
    fixations: pd.DataFrame,
    config: Optional[FixationMetricConfig] = None,
) -> Dict[str, float]:
    """Compute a suite of metrics for a fixation table."""

    if config is None:
        config = FixationMetricConfig()

    if fixations.empty:
        return {
            "fixation_count": 0.0,
            "total_fixation_duration": 0.0,
            "mean_fixation_duration": np.nan,
            "scanpath_length": 0.0,
            "mean_saccade_length": np.nan,
            "avg_pupil_size": np.nan,
            "avg_norm_pupil_size": np.nan,
            "std_norm_pupil_size": np.nan,
        }

    fixation_count = float(len(fixations))
    durations = fixations.get(config.duration_column)
    total_duration = float(durations.sum()) if durations is not None else np.nan
    mean_duration = float(durations.mean()) if durations is not None else np.nan

    scan_length = _scanpath_length(fixations, config.coordinate_columns)
    mean_saccade = float(scan_length / max(fixation_count - 1.0, 1.0))

    pupil = fixations.get(config.pupil_size_column)
    pupil_norm = fixations.get(config.pupil_norm_column)

    metrics: Dict[str, float] = {
        "fixation_count": fixation_count,
        "total_fixation_duration": total_duration,
        "mean_fixation_duration": mean_duration,
        "scanpath_length": scan_length,
        "mean_saccade_length": mean_saccade,
        "avg_pupil_size": float(pupil.mean()) if pupil is not None else np.nan,
        "avg_norm_pupil_size": float(pupil_norm.mean()) if pupil_norm is not None else np.nan,
        "std_norm_pupil_size": float(pupil_norm.std(ddof=1)) if pupil_norm is not None else np.nan,
    }

    return metrics


def load_fixation_table(filepath: Path) -> pd.DataFrame:
    """Load a fixation CSV produced by the preprocessing step."""

    df = pd.read_csv(filepath)
    df.columns = [column.strip() for column in df.columns]
    return df
