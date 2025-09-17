"""High-level utilities for confirmatory eye-tracking analysis."""

from .data_loading import (
    build_metrics_table,
    get_participant_ids,
    load_labels,
)
from .hierarchy import (
    build_hierarchical_summary,
    compute_baseline,
    compute_label_comparison,
)
from .clustering import (
    compute_cluster_profiles,
    run_kmeans_clustering,
)
from .statistics import (
    run_confirmatory_tests,
    summarize_pairwise_results,
)

__all__ = [
    "build_metrics_table",
    "get_participant_ids",
    "load_labels",
    "build_hierarchical_summary",
    "compute_baseline",
    "compute_label_comparison",
    "compute_cluster_profiles",
    "run_kmeans_clustering",
    "run_confirmatory_tests",
    "summarize_pairwise_results",
]
