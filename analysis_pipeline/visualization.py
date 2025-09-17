"""Visualization primitives for the confirmatory analysis workflow."""
from __future__ import annotations

from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .hierarchy import SummaryResult


def metric_summary_frame(summary: SummaryResult, metric: str) -> pd.DataFrame:
    """Return a tidy dataframe with statistics for a single metric."""

    df = summary.summary.copy()
    columns = [
        f"{metric}_mean",
        f"{metric}_std",
        f"{metric}_norm",
        f"{metric}_delta",
        f"{metric}_delta_pct",
    ]
    available = [column for column in columns if column in df.columns]
    tidy = df[available].rename(
        columns={
            f"{metric}_mean": "mean",
            f"{metric}_std": "std",
            f"{metric}_norm": "normalized",
            f"{metric}_delta": "delta",
            f"{metric}_delta_pct": "delta_pct",
        }
    )
    tidy["group"] = tidy.index
    return tidy.reset_index(drop=True)


def plot_metric_normalized(summary: SummaryResult, metric: str, *, title: Optional[str] = None) -> go.Figure:
    data = metric_summary_frame(summary, metric)
    fig = px.bar(
        data,
        x="group",
        y="normalized",
        error_y="std" if "std" in data else None,
        title=title or f"Normalized {metric} by group",
        labels={"normalized": "Normalized value", "group": "Group"},
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_metric_delta(summary: SummaryResult, metric: str, *, title: Optional[str] = None) -> go.Figure:
    data = metric_summary_frame(summary, metric)
    if "delta" not in data:
        raise ValueError("Summary is missing delta information; provide a baseline when building it.")
    fig = px.bar(
        data,
        x="group",
        y="delta",
        title=title or f"Deviation from baseline for {metric}",
        labels={"delta": "Delta (absolute)", "group": "Group"},
        color="delta",
        color_continuous_scale="RdBu",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def plot_pca_scatter(
    pca_components: pd.DataFrame,
    *,
    assignments: Optional[pd.Series] = None,
    title: str = "PCA of metric profiles",
) -> go.Figure:
    df = pca_components.copy()
    if assignments is not None:
        df["cluster"] = assignments
    fig = px.scatter(
        df,
        x="PC1",
        y="PC2",
        color="cluster" if "cluster" in df else None,
        text=df.index,
        title=title,
    )
    fig.update_traces(textposition="top center")
    return fig


def plot_metric_distribution(
    metrics_table: pd.DataFrame,
    *,
    metric: str,
    group_column: str,
    title: Optional[str] = None,
) -> go.Figure:
    fig = px.box(
        metrics_table,
        x=group_column,
        y=metric,
        points="all",
        title=title or f"Distribution of {metric} by {group_column}",
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig
