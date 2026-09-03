"""Cumulative forgetting-rate scoring over integration steps.

The prior AnnData must contain ``step_key`` in ``.obs``. The query/input latent is
tagged as the final step (inferred as the next step after the last prior step,
unless ``final_step`` is given). Scores use cumulative training across all steps
and evaluate on a fixed held-out ``eval_source`` test set.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

from ._cl import (
    DEFAULT_CLASSIFIER,
    DEFAULT_KNN_NEIGHBORS,
    CLMetricsResult,
    compute_cl_metrics,
)


DEFAULT_SOURCE_ORDER = ("reference", "query_fetal", "preterm", "CRC")
DEFAULT_STEP_KEY = "integration_step"
_STEP_NUM_RE = re.compile(r"^(?:step)?(\d+)$", re.IGNORECASE)


def _as_anndata(obj: ad.AnnData | str | Path, *, label: str) -> ad.AnnData:
    if isinstance(obj, ad.AnnData):
        return obj
    path = Path(obj)
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    print(f"Loading {label} from {path}", flush=True)
    return sc.read(path)


def _step_sort_key(name: str) -> tuple:
    text = str(name)
    match = _STEP_NUM_RE.match(text.strip())
    if match:
        return (0, int(match.group(1)), text)
    return (1, text)


def infer_step_order(
    prior: ad.AnnData,
    *,
    step_key: str,
    final_step: str | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...], str]:
    """Return ``(full_order, prior_steps, final_step)`` from prior ``obs[step_key]``.

    Examples
    --------
    >>> from cscanvi.metrics import infer_step_order
    >>> full_order, prior_steps, final_step = infer_step_order(
    ...     prior, step_key="integration_step"
    ... )
    >>> full_order, prior_steps, final_step
    (('step0', 'step1', 'step2'), ('step0', 'step1'), 'step2')
    """
    if step_key not in prior.obs:
        raise KeyError(
            f"Prior adata is missing obs[{step_key!r}]. "
            f"Present columns sample={list(prior.obs.columns)[:20]}"
        )
    prior_steps = tuple(sorted(prior.obs[step_key].astype(str).unique(), key=_step_sort_key))
    if not prior_steps:
        raise ValueError(f"No values found in prior.obs[{step_key!r}].")

    if final_step is None:
        last = prior_steps[-1]
        match = _STEP_NUM_RE.match(last.strip())
        if match:
            final_step = f"step{int(match.group(1)) + 1}"
        else:
            final_step = f"{last}_next"
    if final_step in prior_steps:
        raise ValueError(
            f"final_step={final_step!r} already present in prior steps {list(prior_steps)}."
        )
    full_order = prior_steps + (final_step,)
    return full_order, prior_steps, final_step


def compute_cumulative_fr_scores(
    input_h5ad: ad.AnnData | str | Path,
    prior_h5ad: ad.AnnData | str | Path,
    *,
    step_key: str = DEFAULT_STEP_KEY,
    final_step: str | None = None,
    integration_step_order: Sequence[str] | None = None,
    prior_integration_steps: Sequence[str] | None = None,
    source_col: str = "cell_source",
    source_order: Sequence[str] = DEFAULT_SOURCE_ORDER,
    label_col: str = "level_2_annot",
    eval_source: str = "reference",
    seed: int = 42,
    cumulative_training: bool = True,
    embedding_key: str | None = None,
    classifier: str = DEFAULT_CLASSIFIER,
    knn_neighbors: int = DEFAULT_KNN_NEIGHBORS,
    missing_source_fill: str | None = "case-control CRC",
) -> CLMetricsResult:
    """
    Compute cumulative accuracy / forgetting over prior steps plus one final latent.

    Parameters
    ----------
    input_h5ad
        Final-step latent AnnData or path. Tagged as ``final_step``.
    prior_h5ad
        Prior latents AnnData or path. Must contain ``step_key`` in ``.obs``.
    step_key
        obs column naming integration steps in the prior (default: ``integration_step``).
    final_step
        Name for the input latent step. If omitted, inferred as the next step after
        the last prior step (e.g. prior ends at step3 -> final_step=step4).
    integration_step_order / prior_integration_steps
        Optional overrides. By default inferred from prior ``step_key`` values.

    Examples
    --------
    >>> from cscanvi.metrics import compute_cumulative_fr_scores, write_fr_outputs
    >>> result = compute_cumulative_fr_scores(
    ...     "latents/X_scanvi_my_run.h5ad",
    ...     "adata_ready/prior_for_FR.h5ad",
    ...     eval_source="reference",
    ...     label_col="level_2_annot",
    ...     classifier="knn",
    ... )
    >>> result.summary["final_average_accuracy"]
    >>> result.summary["final_maximum_forgetting"]
    >>> write_fr_outputs(result, "results/fr_metrics")
    """
    input_path = str(input_h5ad) if not isinstance(input_h5ad, ad.AnnData) else None
    prior_path = str(prior_h5ad) if not isinstance(prior_h5ad, ad.AnnData) else None

    adata = _as_anndata(input_h5ad, label="input latent").copy()
    prior = _as_anndata(prior_h5ad, label="prior latents").copy()

    if integration_step_order is None or prior_integration_steps is None or final_step is None:
        inferred_order, inferred_prior, inferred_final = infer_step_order(
            prior, step_key=step_key, final_step=final_step
        )
        if integration_step_order is None:
            integration_step_order = inferred_order
        if prior_integration_steps is None:
            prior_integration_steps = inferred_prior
        if final_step is None:
            final_step = inferred_final

    integration_step_order = tuple(integration_step_order)
    prior_integration_steps = tuple(prior_integration_steps)
    if final_step not in integration_step_order:
        raise ValueError(
            f"final_step={final_step!r} must appear in integration_step_order="
            f"{list(integration_step_order)}"
        )
    missing = [s for s in prior_integration_steps if s not in integration_step_order]
    if missing:
        raise ValueError(
            f"prior_integration_steps has values not in integration_step_order: {missing}"
        )
    if final_step in prior_integration_steps:
        raise ValueError("final_step must not be listed in prior_integration_steps.")

    if step_key != "integration_step":
        prior.obs["integration_step"] = prior.obs[step_key].astype(str)
    else:
        prior.obs["integration_step"] = prior.obs["integration_step"].astype(str)

    present_prior = set(prior.obs["integration_step"].astype(str))
    missing_prior = [s for s in prior_integration_steps if s not in present_prior]
    if missing_prior:
        raise ValueError(
            f"Prior is missing required steps {missing_prior}. "
            f"Present: {sorted(present_prior)}"
        )

    adata.obs["integration_step"] = final_step

    print(
        f"Cumulative FR | prior_steps={list(prior_integration_steps)} | "
        f"final_step={final_step!r} | order={list(integration_step_order)} | "
        f"eval_source={eval_source!r} | cumulative_training={cumulative_training}",
        flush=True,
    )
    print(
        "Prior crosstab:\n"
        f"{pd.crosstab(prior.obs['integration_step'], prior.obs[source_col])}",
        flush=True,
    )

    return compute_cl_metrics(
        adata,
        input_h5ad=input_path,
        prior_steps_adata=prior,
        prior_steps_h5ad=prior_path,
        seed=seed,
        source_col=source_col,
        label_col=label_col,
        source_order=tuple(source_order),
        integration_step_col="integration_step",
        integration_step_order=integration_step_order,
        prior_integration_steps=prior_integration_steps,
        default_integration_step=final_step,
        eval_source=eval_source,
        cumulative_training=cumulative_training,
        embedding_key=embedding_key,
        classifier=classifier,
        knn_neighbors=knn_neighbors,
        missing_source_fill=missing_source_fill,
    )


def write_fr_outputs(
    result: CLMetricsResult,
    output_dir: str | Path,
    *,
    stem: str | None = None,
) -> Path:
    """Write summary CSV, metric matrices, and metrics JSON. Returns summary path.

    Examples
    --------
    >>> from cscanvi.metrics import compute_cumulative_fr_scores, write_fr_outputs
    >>> result = compute_cumulative_fr_scores(query_latent, prior_latent)
    >>> write_fr_outputs(result, "results/fr_metrics", stem="my_run")
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if stem is None:
        stem = Path(result.input_h5ad).stem if result.input_h5ad else "fr_scores"

    row = {
        "input_h5ad": result.input_h5ad,
        "prior_steps_h5ad": result.prior_steps_h5ad,
        "split_seed": result.split_seed,
        **result.summary,
        "final_macro_f1": result.metrics_by_measure["macro_f1"]["final_average"],
        "macro_f1_forgetting": result.metrics_by_measure["macro_f1"]["final_maximum_forgetting"],
        "task_order": ",".join(result.task_order),
        "n_tasks": len(result.task_order),
    }
    summary_path = output_dir / f"{stem}_summary.csv"
    pd.DataFrame([row]).to_csv(summary_path, index=False)

    prefix = output_dir / stem
    result.accuracy_matrix.to_csv(f"{prefix}_accuracy_matrix.csv")
    result.balanced_accuracy_matrix.to_csv(f"{prefix}_balanced_accuracy_matrix.csv")
    result.macro_f1_matrix.to_csv(f"{prefix}_macro_f1_matrix.csv")
    result.per_class_forgetting.to_csv(f"{prefix}_per_class_forgetting.csv", index=False)
    with open(f"{prefix}_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "input_h5ad": result.input_h5ad,
                "prior_steps_h5ad": result.prior_steps_h5ad,
                "task_order": result.task_order,
                "summary": result.summary,
                "metrics_by_measure": result.metrics_by_measure,
            },
            handle,
            indent=2,
        )
    print(f"Wrote summary row: {summary_path}", flush=True)
    return summary_path


def merge_fr_summaries(
    input_dir: str | Path,
    output_csv: str | Path,
    *,
    pattern: str = "*_summary.csv",
    sort_by: str = "final_maximum_forgetting",
    metric_version: str | None = None,
) -> pd.DataFrame:
    """Concatenate per-run ``*_summary.csv`` files into one table.

    Examples
    --------
    >>> from cscanvi.metrics import merge_fr_summaries
    >>> df = merge_fr_summaries(
    ...     "results/fr_metrics",
    ...     "forgetting_metrics.csv",
    ...     pattern="*_summary.csv",
    ...     sort_by="final_maximum_forgetting",
    ... )
    """
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {input_dir / pattern}")

    summary_df = pd.concat([pd.read_csv(path) for path in files], ignore_index=True)
    if metric_version:
        summary_df.insert(0, "fr_metric_version", metric_version)
    if sort_by in summary_df.columns:
        summary_df = summary_df.sort_values(sort_by)

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)
    print(f"Merged {len(files)} rows -> {output_csv} | shape={summary_df.shape}", flush=True)
    return summary_df


def build_prior_latent(
    adata: ad.AnnData | str | Path,
    *,
    embedding_key: str = "X_scANVI",
    step_label: str = "step0",
    step_key: str = DEFAULT_STEP_KEY,
    keep_obs_cols: Sequence[str] | None = None,
    source_col: str = "cell_source",
    default_source: str = "reference",
) -> ad.AnnData:
    """Extract an embedding into ``.X`` and tag cells with an integration step.

    Used to build the prior latent consumed by :func:`compute_cumulative_fr_scores`.

    Examples
    --------
    >>> from cscanvi.metrics import build_prior_latent
    >>> prior = build_prior_latent(
    ...     "adata_ready/oliver2024_panGI_ref.h5ad",
    ...     embedding_key="X_scANVI",
    ...     step_label="step0",
    ... )
    >>> prior.write("adata_ready/prior_for_FR.h5ad")
    """
    src = _as_anndata(adata, label="reference for prior")
    if embedding_key not in src.obsm:
        raise KeyError(
            f"Missing obsm[{embedding_key!r}]; available={list(src.obsm.keys())}"
        )

    default_cols = (
        "cell_source",
        "level_1_annot",
        "level_2_annot",
        "level_3_annot",
        "fine_annot",
        "donorID_unified",
        "study",
        "log1p_n_counts",
        "percent_mito",
    )
    cols = list(keep_obs_cols) if keep_obs_cols is not None else list(default_cols)
    obs = pd.DataFrame(index=src.obs_names.copy())
    for col in cols:
        if col in src.obs.columns:
            obs[col] = src.obs[col].astype(str).to_numpy()
    if source_col not in obs.columns:
        obs[source_col] = default_source
    obs[step_key] = step_label
    return ad.AnnData(
        X=np.asarray(src.obsm[embedding_key], dtype="float32"),
        obs=obs,
    )
