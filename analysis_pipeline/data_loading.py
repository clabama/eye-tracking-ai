"""Data ingestion helpers for confirmatory analysis."""
from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .metrics import FixationMetricConfig, compute_fixation_metrics, load_fixation_table

_LABEL_COLUMNS: List[str] = ["meme", "person", "politik", "ort", "text"]

_FILENAME_PATTERN = re.compile(
    r"^P(?P<participant>\d+)_id(?P<image>\d+)(?:_(?P<label>[A-Za-z0-9-]+))?_(?P<weights>[0-9]{5})?\.csv$"
)


@dataclass(frozen=True)
class FixationRecord:
    """Metadata extracted from a fixation file name."""

    participant_id: str
    image_id: int
    label_hint: Optional[str]
    weight_code: Optional[str]

    @property
    def label_from_filename(self) -> Optional[str]:
        if self.label_hint:
            return self.label_hint.replace("-", " ")
        return None


def parse_fixation_filename(filename: str) -> FixationRecord:
    """Parse metadata encoded in a processed fixation file name."""

    match = _FILENAME_PATTERN.match(Path(filename).name)
    if not match:
        raise ValueError(f"Cannot parse fixation filename: {filename}")

    participant = match.group("participant")
    image = match.group("image")
    label = match.group("label")
    weights = match.group("weights")

    return FixationRecord(
        participant_id=f"P{int(participant):03d}",
        image_id=int(image),
        label_hint=label,
        weight_code=weights,
    )


def load_labels(labels_csv: Path) -> pd.DataFrame:
    """Load the label assignment table and derive combination columns."""

    labels = pd.read_csv(labels_csv)
    labels["image_id"] = labels["image_id"].astype(int)
    labels[_LABEL_COLUMNS] = labels[_LABEL_COLUMNS].fillna(0).astype(int)
    labels["label_count"] = labels[_LABEL_COLUMNS].sum(axis=1)

    def _combo(row: pd.Series) -> str:
        active = [column for column in _LABEL_COLUMNS if row[column] > 0]
        return " & ".join(active) if active else "unlabeled"

    labels["label_combo"] = labels.apply(_combo, axis=1)
    return labels


def get_participant_ids(fixations_dir: Path) -> List[str]:
    """Return sorted participant identifiers present in the fixation directory."""

    participants = set()
    for filepath in Path(fixations_dir).glob("*.csv"):
        try:
            record = parse_fixation_filename(filepath.name)
        except ValueError:
            continue
        participants.add(record.participant_id)
    return sorted(participants)


def build_metrics_table(
    fixations_dir: Path,
    labels_csv: Path,
    *,
    metric_config: Optional[FixationMetricConfig] = None,
) -> pd.DataFrame:
    """Build a participant-level metric table joined with label metadata."""

    labels = load_labels(labels_csv)
    labels = labels.set_index("image_id")
    records: List[Dict[str, float]] = []

    for filepath in Path(fixations_dir).glob("*.csv"):
        try:
            record = parse_fixation_filename(filepath.name)
        except ValueError:
            continue

        fixation_df = load_fixation_table(filepath)
        metrics = compute_fixation_metrics(fixation_df, config=metric_config)

        label_row = labels.loc[record.image_id] if record.image_id in labels.index else None
        combined: Dict[str, float] = {
            "filename": filepath.name,
            "participant_id": record.participant_id,
            "image_id": record.image_id,
            "label_hint": record.label_from_filename,
        }
        combined.update(metrics)

        if label_row is not None:
            for column in label_row.index:
                combined[column] = label_row[column]

        records.append(combined)

    metrics_table = pd.DataFrame.from_records(records)
    if not metrics_table.empty:
        metrics_table["participant_id"] = metrics_table["participant_id"].astype("category")
        metrics_table["label_combo"] = metrics_table["label_combo"].astype("category")
    return metrics_table
