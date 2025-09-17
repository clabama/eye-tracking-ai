"""Clustering helpers for label combinations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .hierarchy import compute_label_comparison


@dataclass
class ClusterOutput:
    features: pd.DataFrame
    scaled_features: np.ndarray
    pca_model: Optional[PCA]
    pca_components: Optional[pd.DataFrame]
    assignments: Optional[pd.Series]
    kmeans_model: Optional[KMeans]


def run_kmeans_clustering(
    features: pd.DataFrame,
    *,
    n_clusters: int,
    random_state: int = 42,
) -> KMeans:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
    model.fit(scaled)
    return model


def compute_cluster_profiles(
    metrics_table: pd.DataFrame,
    *,
    metrics: Optional[Iterable[str]] = None,
    n_components: int = 2,
    n_clusters: Optional[int] = None,
    random_state: int = 42,
) -> ClusterOutput:
    summary = compute_label_comparison(
        metrics_table,
        group_column="label_combo",
        metrics=metrics,
        baseline=None,
        min_count=1,
    )
    if summary.summary.empty:
        return ClusterOutput(
            features=pd.DataFrame(),
            scaled_features=np.empty((0, 0)),
            pca_model=None,
            pca_components=None,
            assignments=None,
            kmeans_model=None,
        )

    feature_columns = [column for column in summary.summary.columns if column.endswith("_norm")]
    features = summary.summary[feature_columns]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    pca_model: Optional[PCA] = None
    pca_components_df: Optional[pd.DataFrame] = None
    if features.shape[1] >= n_components and n_components > 0:
        pca_model = PCA(n_components=n_components, random_state=random_state)
        components = pca_model.fit_transform(scaled)
        pca_components_df = pd.DataFrame(
            components,
            index=features.index,
            columns=[f"PC{i+1}" for i in range(components.shape[1])],
        )

    assignments: Optional[pd.Series] = None
    kmeans_model: Optional[KMeans] = None
    if n_clusters is not None and n_clusters > 1:
        kmeans_model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=random_state)
        kmeans_model.fit(scaled)
        assignments = pd.Series(kmeans_model.labels_, index=features.index, name="cluster")

    return ClusterOutput(
        features=features,
        scaled_features=scaled,
        pca_model=pca_model,
        pca_components=pca_components_df,
        assignments=assignments,
        kmeans_model=kmeans_model,
    )
