"""Streamlit application enabling interactive confirmatory analysis."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from .clustering import compute_cluster_profiles
from .data_loading import build_metrics_table
from .hierarchy import SummaryResult, build_hierarchical_summary
from .statistics import run_confirmatory_tests, summarize_pairwise_results
from .visualization import (
    metric_summary_frame,
    plot_metric_delta,
    plot_metric_distribution,
    plot_metric_normalized,
    plot_pca_scatter,
)


@st.cache_data(show_spinner="Loading metrics table...")
def _load_metrics_cached(fixations_dir: str, labels_csv: str) -> pd.DataFrame:
    return build_metrics_table(Path(fixations_dir), Path(labels_csv))


def _show_summary(summary: SummaryResult, metric: str, *, show_delta: bool = True) -> None:
    st.write(f"### {metric} summary")
    st.dataframe(metric_summary_frame(summary, metric), use_container_width=True)
    st.plotly_chart(
        plot_metric_normalized(summary, metric),
        use_container_width=True,
    )
    if show_delta:
        try:
            st.plotly_chart(
                plot_metric_delta(summary, metric),
                use_container_width=True,
            )
        except ValueError:
            st.info("Delta plot requires a baseline; ensure the summary was built with one.")


def _run_gui(metrics_table: pd.DataFrame) -> None:
    st.title("Eye-tracking confirmatory analysis")

    if metrics_table.empty:
        st.warning("No fixation metrics available. Check your paths and preprocessing outputs.")
        return

    st.sidebar.header("Analysis options")
    metric_columns = [
        column
        for column in [
            "fixation_count",
            "total_fixation_duration",
            "mean_fixation_duration",
            "scanpath_length",
            "mean_saccade_length",
            "avg_pupil_size",
            "avg_norm_pupil_size",
        ]
        if column in metrics_table.columns
    ]
    selected_metric = st.sidebar.selectbox("Metric", metric_columns)

    summary = build_hierarchical_summary(metrics_table, metrics=metric_columns)

    st.write("## Baseline (All images)")
    st.dataframe(summary["baseline"].summary, use_container_width=True)

    level = st.sidebar.radio("Hierarchy level", ("Single labels", "Label combinations"))
    if level == "Single labels":
        summary_result = summary["single_labels"]
    else:
        summary_result = summary["label_combinations"]
        available = summary_result.summary.index.tolist()
        selected_labels = st.sidebar.multiselect(
            "Filter label combinations",
            options=available,
            default=available,
        )
        if selected_labels:
            summary_result = SummaryResult(
                summary=summary_result.summary.loc[selected_labels],
                metrics=summary_result.metrics,
            )

    _show_summary(summary_result, selected_metric, show_delta=True)

    st.write("## Distribution by group")
    group_column = "label_combo" if level == "Label combinations" else "labels_txt"
    st.plotly_chart(
        plot_metric_distribution(
            metrics_table,
            metric=selected_metric,
            group_column=group_column,
            title=f"Distribution of {selected_metric} by {group_column}",
        ),
        use_container_width=True,
    )

    st.write("## Clustering of label combinations")
    cluster_k = st.sidebar.slider("Number of clusters", min_value=2, max_value=8, value=3)
    cluster_output = compute_cluster_profiles(
        metrics_table,
        metrics=metric_columns,
        n_components=2,
        n_clusters=cluster_k,
    )
    if cluster_output.pca_components is not None:
        st.plotly_chart(
            plot_pca_scatter(cluster_output.pca_components, assignments=cluster_output.assignments),
            use_container_width=True,
        )
        st.dataframe(cluster_output.features, use_container_width=True)
    else:
        st.info("PCA components could not be computed for the selected metrics.")

    st.write("## Confirmatory statistics")
    test_group_column = "label_combo" if level == "Label combinations" else "labels_txt"
    stats_results = run_confirmatory_tests(
        metrics_table,
        group_column=test_group_column,
        metrics=metric_columns,
    )
    if not stats_results:
        st.info("Not enough observations per group for statistical tests.")
    else:
        for metric_name, result in stats_results.items():
            st.subheader(metric_name)
            anova_res = result.get("anova")
            if anova_res:
                eta_string = (
                    f"{anova_res.effect_size:.3f}" if anova_res.effect_size is not None else "NA"
                )
                st.markdown(
                    f"ANOVA: F={anova_res.statistic:.3f} (df={anova_res.df_between}, {anova_res.df_within}), "
                    f"p={anova_res.pvalue:.3g}, eta²={eta_string}"
                )
            kruskal_res = result.get("kruskal")
            if kruskal_res:
                st.markdown(
                    f"Kruskal-Wallis: H={kruskal_res.statistic:.3f}, p={kruskal_res.pvalue:.3g}"
                )
            pairwise_param = summarize_pairwise_results(result.get("pairwise_parametric", []))
            if not pairwise_param.empty:
                st.write("Parametric post-hoc tests")
                st.dataframe(pairwise_param, use_container_width=True)
            pairwise_nonparam = summarize_pairwise_results(result.get("pairwise_nonparametric", []))
            if not pairwise_nonparam.empty:
                st.write("Non-parametric post-hoc tests")
                st.dataframe(pairwise_nonparam, use_container_width=True)


def main(
    *,
    fixations_dir: str = "fixations",
    labels_csv: str = "labels_per_id.csv",
) -> None:
    metrics_table = _load_metrics_cached(fixations_dir, labels_csv)
    _run_gui(metrics_table)


if __name__ == "__main__":
    main()
