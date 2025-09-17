"""Statistical testing utilities for confirmatory analyses."""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

_DEFAULT_METRICS: List[str] = [
    "fixation_count",
    "total_fixation_duration",
    "scanpath_length",
    "mean_fixation_duration",
    "avg_pupil_size",
    "avg_norm_pupil_size",
]


@dataclass(frozen=True)
class TestResult:
    statistic: float
    pvalue: float
    effect_size: Optional[float]
    test_type: str
    df_between: Optional[int] = None
    df_within: Optional[int] = None


@dataclass(frozen=True)
class PairwiseResult:
    group_a: str
    group_b: str
    statistic: float
    pvalue_raw: float
    pvalue_adj: float
    effect_size: Optional[float]
    method: str


def _cohen_d(sample_a: np.ndarray, sample_b: np.ndarray) -> Optional[float]:
    if len(sample_a) < 2 or len(sample_b) < 2:
        return None
    mean_a, mean_b = sample_a.mean(), sample_b.mean()
    var_a, var_b = sample_a.var(ddof=1), sample_b.var(ddof=1)
    pooled = ((len(sample_a) - 1) * var_a + (len(sample_b) - 1) * var_b) / (
        len(sample_a) + len(sample_b) - 2
    )
    if pooled <= 0:
        return None
    return float((mean_a - mean_b) / np.sqrt(pooled))


def _rank_biserial(u_stat: float, n_a: int, n_b: int) -> float:
    return 1 - (2 * u_stat) / (n_a * n_b)


def _prepare_samples(
    metrics_table: pd.DataFrame,
    group_column: str,
    metric: str,
) -> Dict[str, np.ndarray]:
    samples: Dict[str, np.ndarray] = {}
    for group_value, subset in metrics_table.groupby(group_column, observed=True):
        values = subset[metric].dropna().to_numpy()
        if len(values) > 0:
            samples[str(group_value)] = values
    return samples


def _anova(samples: Mapping[str, np.ndarray]) -> Optional[TestResult]:
    if len(samples) < 2:
        return None
    arrays = list(samples.values())
    statistic, pvalue = stats.f_oneway(*arrays)
    n_total = sum(len(sample) for sample in arrays)
    df_between = len(arrays) - 1
    df_within = n_total - len(arrays)
    eta_sq = (statistic * df_between) / (statistic * df_between + df_within) if df_within > 0 else None
    return TestResult(
        statistic=float(statistic),
        pvalue=float(pvalue),
        effect_size=float(eta_sq) if eta_sq is not None else None,
        test_type="ANOVA",
        df_between=df_between,
        df_within=df_within,
    )


def _kruskal(samples: Mapping[str, np.ndarray]) -> Optional[TestResult]:
    if len(samples) < 2:
        return None
    statistic, pvalue = stats.kruskal(*samples.values())
    return TestResult(
        statistic=float(statistic),
        pvalue=float(pvalue),
        effect_size=None,
        test_type="Kruskal-Wallis",
    )


def _bonferroni(pvalues: List[float]) -> List[float]:
    correction = len(pvalues)
    return [min(p * correction, 1.0) for p in pvalues]


def _pairwise_tests(
    samples: Mapping[str, np.ndarray],
    *,
    method: str,
    correction: str = "bonferroni",
) -> List[PairwiseResult]:
    pairs = list(itertools.combinations(samples.items(), 2))
    raw_pvalues: List[float] = []
    base_results: List[Tuple[str, str, float, float, Optional[float]]] = []

    for (label_a, sample_a), (label_b, sample_b) in pairs:
        if method == "parametric":
            statistic, pvalue = stats.ttest_ind(sample_a, sample_b, equal_var=False)
            effect = _cohen_d(sample_a, sample_b)
        else:
            statistic, pvalue = stats.mannwhitneyu(sample_a, sample_b, alternative="two-sided")
            effect = _rank_biserial(statistic, len(sample_a), len(sample_b))
        raw_pvalues.append(float(pvalue))
        base_results.append((str(label_a), str(label_b), float(statistic), float(pvalue), effect))

    if not base_results:
        return []

    if correction.lower() == "bonferroni":
        adjusted = _bonferroni(raw_pvalues)
    else:
        adjusted = raw_pvalues

    pairwise_results: List[PairwiseResult] = []
    for (label_a, label_b, statistic, pvalue, effect), adj in zip(base_results, adjusted):
        pairwise_results.append(
            PairwiseResult(
                group_a=label_a,
                group_b=label_b,
                statistic=statistic,
                pvalue_raw=pvalue,
                pvalue_adj=adj,
                effect_size=effect,
                method="parametric" if method == "parametric" else "nonparametric",
            )
        )
    return pairwise_results


def run_confirmatory_tests(
    metrics_table: pd.DataFrame,
    *,
    group_column: str,
    metrics: Optional[Iterable[str]] = None,
    alpha: float = 0.05,
    correction: str = "bonferroni",
) -> Dict[str, Dict[str, object]]:
    metrics = list(metrics or _DEFAULT_METRICS)
    results: Dict[str, Dict[str, object]] = {}

    for metric in metrics:
        samples = _prepare_samples(metrics_table, group_column, metric)
        if len(samples) < 2:
            continue
        anova_res = _anova(samples)
        kruskal_res = _kruskal(samples)
        parametric_pairs = _pairwise_tests(samples, method="parametric", correction=correction)
        nonparam_pairs = _pairwise_tests(samples, method="nonparametric", correction=correction)

        results[metric] = {
            "anova": anova_res,
            "kruskal": kruskal_res,
            "pairwise_parametric": parametric_pairs,
            "pairwise_nonparametric": nonparam_pairs,
            "descriptives": {
                label: {
                    "n": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)) if len(values) > 1 else np.nan,
                }
                for label, values in samples.items()
            },
        }
    return results


def summarize_pairwise_results(
    pairwise: Iterable[PairwiseResult],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for result in pairwise:
        rows.append(
            {
                "group_a": result.group_a,
                "group_b": result.group_b,
                "statistic": result.statistic,
                "pvalue_raw": result.pvalue_raw,
                "pvalue_adj": result.pvalue_adj,
                "effect_size": result.effect_size,
                "method": result.method,
                "significant": result.pvalue_adj < alpha,
            }
        )
    return pd.DataFrame(rows)
