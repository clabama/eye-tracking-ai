"""Hierarchical summaries of metrics across label levels."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import numpy as np
import pandas as pd

_LABEL_COLUMNS: List[str] = ["meme", "person", "politik", "ort", "text"]
_DEFAULT_METRIC_COLUMNS: List[str] = [
    "fixation_count",
    "total_fixation_duration",
    "mean_fixation_duration",
    "scanpath_length",
    "mean_saccade_length",
    "avg_pupil_size",
    "avg_norm_pupil_size",
]


@dataclass(frozen=True)
class SummaryResult:
    """Container bundling summary statistics for a label grouping."""

    summary: pd.DataFrame
    metrics: List[str]

    def difference_columns(self) -> List[str]:
        return [column for column in self.summary.columns if column.endswith("_delta")]


def _participant_count(metrics_table: pd.DataFrame) -> int:
    return int(metrics_table["participant_id"].nunique())


def _aggregate_subset(
    subset: pd.DataFrame,
    metrics: Iterable[str],
    participant_total: int,
) -> Dict[str, float]:
    if subset.empty:
        result: Dict[str, float] = {f"{metric}_mean": np.nan for metric in metrics}
        result.update({f"{metric}_std": np.nan for metric in metrics})
        result.update({f"{metric}_norm": np.nan for metric in metrics})
        result["participants_observed"] = 0
        result["images_observed"] = 0
        return result

    participant_group = (
        subset.groupby("participant_id", observed=True)[list(metrics)].mean()
    )

    result: Dict[str, float] = {}
    for metric in metrics:
        values = participant_group[metric]
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
        result[f"{metric}_norm"] = float(values.sum() / participant_total)
    result["participants_observed"] = int(participant_group.shape[0])
    result["images_observed"] = int(subset["image_id"].nunique())
    return result


def compute_baseline(
    metrics_table: pd.DataFrame,
    *,
    metrics: Optional[Iterable[str]] = None,
) -> pd.Series:
    metrics = list(metrics or _DEFAULT_METRIC_COLUMNS)
    participant_total = _participant_count(metrics_table)
    summary = _aggregate_subset(metrics_table, metrics, participant_total)
    return pd.Series(summary)


def _attach_differences(
    summary_df: pd.DataFrame,
    baseline: Mapping[str, float],
    metrics: Iterable[str],
) -> pd.DataFrame:
    df = summary_df.copy()
    for metric in metrics:
        baseline_value = baseline.get(f"{metric}_norm", np.nan)
        if np.isnan(baseline_value) or baseline_value == 0:
            df[f"{metric}_delta"] = np.nan
            df[f"{metric}_delta_pct"] = np.nan
            continue
        df[f"{metric}_delta"] = df[f"{metric}_norm"] - baseline_value
        df[f"{metric}_delta_pct"] = df[f"{metric}_delta"] / baseline_value
    return df


def compute_label_comparison(
    metrics_table: pd.DataFrame,
    *,
    group_column: str,
    metrics: Optional[Iterable[str]] = None,
    baseline: Optional[Mapping[str, float]] = None,
    min_count: int = 1,
) -> SummaryResult:
    metrics = list(metrics or _DEFAULT_METRIC_COLUMNS)
    participant_total = _participant_count(metrics_table)

    summaries: List[Dict[str, float]] = []
    labels: List[str] = []

    for label_value, subset in metrics_table.groupby(group_column, observed=True):
        subset_stats = _aggregate_subset(subset, metrics, participant_total)
        if subset_stats["participants_observed"] < min_count:
            continue
        subset_stats[group_column] = label_value
        summaries.append(subset_stats)
        labels.append(label_value)

    summary_df = pd.DataFrame(summaries)
    if summary_df.empty:
        return SummaryResult(summary=pd.DataFrame(), metrics=metrics)

    if baseline is not None:
        summary_df = _attach_differences(summary_df, baseline, metrics)

    summary_df = summary_df.set_index(group_column).sort_index()
    return SummaryResult(summary=summary_df, metrics=metrics)


def _single_label_table(
    metrics_table: pd.DataFrame,
    metrics: Iterable[str],
    baseline: Optional[Mapping[str, float]],
) -> SummaryResult:
    participant_total = _participant_count(metrics_table)
    rows: List[Dict[str, float]] = []
    for label in _LABEL_COLUMNS:
        subset = metrics_table.loc[metrics_table[label] == 1]
        stats = _aggregate_subset(subset, metrics, participant_total)
        stats["label"] = label
        rows.append(stats)
    df = pd.DataFrame(rows)
    if baseline is not None:
        df = _attach_differences(df, baseline, metrics)
    df = df.set_index("label")
    return SummaryResult(summary=df, metrics=list(metrics))


def build_hierarchical_summary(
    metrics_table: pd.DataFrame,
    *,
    metrics: Optional[Iterable[str]] = None,
) -> Dict[str, SummaryResult]:
    metrics = list(metrics or _DEFAULT_METRIC_COLUMNS)
    baseline = compute_baseline(metrics_table, metrics=metrics)
    single_labels = _single_label_table(metrics_table, metrics, baseline)
    combinations = compute_label_comparison(
        metrics_table,
        group_column="label_combo",
        metrics=metrics,
        baseline=baseline,
        min_count=1,
    )
    baseline_df = baseline.to_frame().T
    baseline_df.index = ["All"]
    return {
        "baseline": SummaryResult(summary=baseline_df, metrics=metrics),
        "single_labels": single_labels,
        "label_combinations": combinations,
    }
