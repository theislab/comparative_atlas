"""Continual-learning / forgetting-rate metrics for sequential SCANVI integration.

Paper metrics follow Chaudhry et al. (arXiv:2002.08165 / GEM Eqs. 2–3). Matrix
entry ``R[i, j]`` is accuracy on a fixed held-out ``eval_source`` test set using
step-``j`` embeddings after cumulative training through step ``i``.

- ``A_T = (1/T) sum_j R[T-1, j]``
- ``F_T = (1/(T-1)) sum_{j=0}^{T-2} (max_{l<T} a_{l,j} - a_{T,j})``

For the reference task (``j=0``), ``a_{l,0}`` uses step-``l`` embeddings of the
same cells (``R[l, l]``), and ``a_{T,0} = R[T-1, T-1]``. Using ``R[l, 0]`` for
all ``l`` wrongly freezes step-0 embeddings and drives ``F_T`` to ~0 when
``T=2``. For ``j>0``, ``a_{l,j} = R[l, j]`` as usual.

Call :func:`compute_cl_metrics` with a final-step latent and optional prior-step
latents, or :func:`cscanvi.metrics.compute_cumulative_fr_scores` when the prior
already carries an integration-step column.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


DEFAULT_SOURCE_ORDER = ("reference", "query_fetal", "case-control CRC")
DEFAULT_INTEGRATION_STEP_ORDER = ("step0", "step1", "step2")
DEFAULT_LEGACY_INTEGRATION_STEP = "step2"
DEFAULT_PRIOR_INTEGRATION_STEPS = ("step0", "step1")
DEFAULT_CLASSIFIER = "knn"
DEFAULT_KNN_NEIGHBORS = 15


@dataclass
class MetricsConfig:
    """Configuration for continual-learning metric computation."""

    source_col: str = "cell_source"
    source_order: Tuple[str, ...] = DEFAULT_SOURCE_ORDER
    integration_step_col: str = "integration_step"
    integration_step_order: Tuple[str, ...] = DEFAULT_INTEGRATION_STEP_ORDER
    prior_integration_steps: Tuple[str, ...] = DEFAULT_PRIOR_INTEGRATION_STEPS
    default_integration_step: str = DEFAULT_LEGACY_INTEGRATION_STEP
    prior_steps_h5ad: str | None = None
    label_col: str = "level_2_annot"
    embedding_key: str | None = None
    test_size: float = 0.2
    seed: int = 42
    max_iter: int = 1000
    class_weight: str | None = None
    bootstrap_iters: int = 1000
    eval_source: str | None = None
    cumulative_training: bool = True
    fixed_test_obs_names: Tuple[str, ...] | None = None
    classifier: str = DEFAULT_CLASSIFIER
    knn_neighbors: int = DEFAULT_KNN_NEIGHBORS
    missing_source_fill: str | None = "case-control CRC"


@dataclass
class CLMetricsResult:
    """Notebook-friendly container returned by :func:`compute_cl_metrics`."""

    task_order: List[str]
    label_vocab: List[str]
    accuracy_matrix: pd.DataFrame
    balanced_accuracy_matrix: pd.DataFrame
    macro_f1_matrix: pd.DataFrame
    per_class_forgetting: pd.DataFrame
    metrics_by_measure: Dict[str, Dict[str, Any]]
    summary: Dict[str, float]
    eval_source: str | None = None
    eval_test_obs_names: List[str] | None = None
    split_seed: int | None = None
    input_h5ad: str | None = None
    prior_steps_h5ad: str | None = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_h5ad": self.input_h5ad,
            "prior_steps_h5ad": self.prior_steps_h5ad,
            "task_order": self.task_order,
            "label_vocab": self.label_vocab,
            "accuracy_matrix": self.accuracy_matrix,
            "balanced_accuracy_matrix": self.balanced_accuracy_matrix,
            "macro_f1_matrix": self.macro_f1_matrix,
            "per_class_forgetting": self.per_class_forgetting,
            "metrics_by_measure": self.metrics_by_measure,
            "summary": self.summary,
            "eval_source": self.eval_source,
            "eval_test_obs_names": self.eval_test_obs_names,
            "split_seed": self.split_seed,
        }


@dataclass
class TaskData:
    task_name: str
    X: np.ndarray
    y: np.ndarray
    train_idx: np.ndarray
    test_idx: np.ndarray
    obs_names: np.ndarray


def parse_source_order(source_order: str) -> List[str]:
    order = [part.strip() for part in source_order.split(",") if part.strip()]
    if not order:
        raise ValueError("source_order must contain at least one label.")
    return order


def parse_integration_step_order(integration_step_order: str) -> List[str]:
    order = [part.strip() for part in integration_step_order.split(",") if part.strip()]
    if not order:
        raise ValueError("integration_step_order must contain at least one step label.")
    return order


def ensure_integration_step(
    adata: sc.AnnData,
    *,
    integration_step_col: str,
    default_integration_step: str,
    inplace: bool = False,
    require_existing: bool = False,
) -> sc.AnnData:
    """Assign or validate the integration-step column."""
    target = adata if inplace else adata.copy()
    if integration_step_col not in target.obs:
        if require_existing:
            raise KeyError(
                f"Column {integration_step_col!r} not found in adata.obs; "
                "prior-steps latent files must include integration_step."
            )
        print(
            f"Column {integration_step_col!r} not found; assigning "
            f"{default_integration_step!r} to all cells.",
            flush=True,
        )
        target.obs[integration_step_col] = default_integration_step
    else:
        target.obs[integration_step_col] = target.obs[integration_step_col].astype(str)
    return target


def _prepare_step2_adata(
    adata: sc.AnnData,
    *,
    source_col: str,
    integration_step_col: str,
    default_integration_step: str,
    label_col: str,
    missing_source_fill: str | None = "case-control CRC",
) -> sc.AnnData:
    adata = adata.copy()
    adata = ensure_integration_step(
        adata,
        integration_step_col=integration_step_col,
        default_integration_step=default_integration_step,
        inplace=True,
        require_existing=False,
    )
    adata.obs[source_col] = normalize_cell_source(
        adata.obs[source_col], missing_fill=missing_source_fill
    )
    return adata


def _prepare_prior_steps_adata(
    adata: sc.AnnData,
    *,
    source_col: str,
    integration_step_col: str,
    prior_integration_steps: Sequence[str],
    label_col: str,
    missing_source_fill: str | None = "case-control CRC",
) -> sc.AnnData:
    adata = adata.copy()
    adata = ensure_integration_step(
        adata,
        integration_step_col=integration_step_col,
        default_integration_step="",
        inplace=True,
        require_existing=True,
    )
    adata.obs[source_col] = normalize_cell_source(
        adata.obs[source_col], missing_fill=missing_source_fill
    )
    keep = adata.obs[integration_step_col].isin(list(prior_integration_steps))
    if not keep.any():
        raise ValueError(
            f"No cells with integration_step in {list(prior_integration_steps)} "
            f"found in prior-steps adata."
        )
    dropped = adata[~keep].copy()
    if dropped.n_obs > 0:
        print(
            f"Dropping {dropped.n_obs:,} prior-steps cells outside "
            f"{list(prior_integration_steps)}",
            flush=True,
        )
    return adata[keep].copy()


def _filter_labeled_cells(adata: sc.AnnData, label_col: str) -> sc.AnnData:
    label_mask = adata.obs[label_col].notna() & (
        adata.obs[label_col].astype(str).str.strip() != ""
    )
    if not label_mask.all():
        n_drop = int((~label_mask).sum())
        print(f"Dropping {n_drop:,} cells with missing {label_col}", flush=True)
        adata = adata[label_mask].copy()
    return adata


def resolve_eval_source(source_order: Sequence[str], eval_source: str | None) -> str:
    resolved = source_order[0] if eval_source is None else eval_source
    if resolved not in source_order:
        raise ValueError(
            f"eval_source {resolved!r} is not in source_order: {list(source_order)}"
        )
    return resolved


def _score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float, float]:
    """Score predictions using only labels present in y_true (avoids sklearn warnings)."""
    eval_labels = np.unique(y_true)
    recalls = []
    for label in eval_labels:
        mask = y_true == label
        if np.any(mask):
            recalls.append(float(np.mean(y_pred[mask] == label)))
    bal_acc = float(np.mean(recalls)) if recalls else float("nan")

    try:
        macro_f1 = float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                labels=eval_labels,
                zero_division=0,
            )
        )
    except TypeError:
        macro_f1 = float(
            f1_score(y_true, y_pred, average="macro", labels=eval_labels)
        )

    return float(accuracy_score(y_true, y_pred)), bal_acc, macro_f1


def make_classifier(
    *,
    classifier: str,
    max_iter: int,
    class_weight: str | None,
    knn_neighbors: int,
):
    name = classifier.strip().lower()
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=knn_neighbors, n_jobs=-1)
    if name in ("logistic", "logistic_regression", "lr"):
        return LogisticRegression(
            max_iter=max_iter,
            solver="lbfgs",
            multi_class="multinomial",
            class_weight=class_weight,
        )
    raise ValueError(
        f"Unknown classifier {classifier!r}; expected 'knn' or 'logistic'."
    )


def _load_fixed_test_obs_names(path: str | Path) -> Tuple[str, ...]:
    names: List[str] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            name = line.strip()
            if name:
                names.append(name)
    if not names:
        raise ValueError(f"No obs names found in {path}")
    return tuple(canonical_obs_name(name) for name in names)


def canonical_obs_name(name: str) -> str:
    """
    Strip anndata ``concatenate`` batch suffixes (-0, -1) for cross-step matching.

    Combined step0/step1 latent files assign ``-0`` to step0 cells and ``-1`` to
    step1 cells with the same underlying barcode. Step-2 latents typically keep
    the unsuffixed barcode. Eval test cells are tracked by this canonical ID.
    """
    return re.sub(r"-[01]$", "", str(name))


def _normalize_adata_obs_names(adata: sc.AnnData, *, index_col: str = "_index") -> sc.AnnData:
    """
    Rewrite obs index to canonical barcodes.

    Step-2 sweep latents often keep integer obs_names (0, 1, ...) with the real
    barcode in ``obs['_index']``. Prior-step latents typically already use barcodes
    as obs_names; when ``index_col`` is absent, only batch suffixes are stripped.
    """
    adata = adata.copy()
    if index_col in adata.obs:
        adata.obs_names = pd.Index(adata.obs[index_col].astype(str), dtype=str)
    adata.obs_names = pd.Index(
        [canonical_obs_name(name) for name in adata.obs_names.astype(str)],
        dtype=str,
    )
    return adata


def _obs_name_lookup(obs_names: np.ndarray) -> Dict[str, int]:
    """Map raw or canonical obs names to positional indices."""
    lookup: Dict[str, int] = {}
    for idx, name in enumerate(obs_names):
        raw = str(name)
        lookup[raw] = idx
        lookup[canonical_obs_name(raw)] = idx
    return lookup


def reference_train_test_split(
    obs_names: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float,
    seed: int,
    fixed_test_obs_names: Sequence[str] | None = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Deterministic reference split shared across latent h5ad files.

    When ``fixed_test_obs_names`` is None, cells are sorted by ``obs_names`` before
    ``train_test_split(..., random_state=seed)`` so file order does not change which
    barcodes land in the test set.
    """
    n = len(y)
    if n == 0:
        raise ValueError("Cannot split an empty reference task.")

    if fixed_test_obs_names is not None:
        fixed_canonical = [canonical_obs_name(name) for name in fixed_test_obs_names]
        fixed_set = set(fixed_canonical)
        name_to_idx = _obs_name_lookup(obs_names)
        missing = [name for name in fixed_canonical if name not in name_to_idx]
        if missing:
            raise KeyError(
                f"{len(missing)} fixed test obs names are absent from this adata. "
                f"Examples: {missing[:5]}"
            )
        test_idx = np.asarray([name_to_idx[name] for name in fixed_canonical], dtype=np.int64)
        train_idx = np.asarray(
            [
                idx
                for idx, name in enumerate(obs_names)
                if canonical_obs_name(name) not in fixed_set
            ],
            dtype=np.int64,
        )
        return train_idx, test_idx, fixed_canonical

    canonical_names = np.array([canonical_obs_name(name) for name in obs_names])
    sort_order = np.argsort(canonical_names, kind="mergesort")
    y_sorted = y[sort_order]
    idx_sorted = np.arange(n, dtype=np.int64)
    train_sorted, test_sorted = train_test_split(
        idx_sorted,
        test_size=test_size,
        random_state=seed,
        stratify=y_sorted,
    )
    train_idx = sort_order[np.asarray(train_sorted, dtype=np.int64)]
    test_idx = sort_order[np.asarray(test_sorted, dtype=np.int64)]
    test_names = [canonical_obs_name(name) for name in obs_names[test_idx]]
    return train_idx, test_idx, test_names


def normalize_cell_source(
    series: pd.Series, *, missing_fill: str | None = "case-control CRC"
) -> pd.Series:
    """Cast source labels to str, optionally filling missing values."""
    normalized = series.copy()
    if missing_fill is not None:
        normalized = normalized.where(~normalized.isna(), missing_fill)
    return normalized.astype(str)


def prepare_epi_crc_cell_source(
    adata: sc.AnnData,
    *,
    source_col: str = "cell_source",
    index_col: str = "_index",
    inplace: bool = False,
    missing_fill: str | None = "case-control CRC",
) -> sc.AnnData:
    """
    Apply notebook-style cell_source cleanup.

    Maps missing ``cell_source`` values to ``missing_fill``. If ``index_col``
    is present, assignment follows the notebook pattern via that index column.
    """
    target = adata if inplace else adata.copy()
    if source_col not in target.obs:
        raise KeyError(f"Column '{source_col}' not found in adata.obs.")

    source = target.obs[source_col]
    if index_col in target.obs:
        mapping = dict(zip(target.obs[index_col].astype(str), source))
        mapping = {
            key: missing_fill if missing_fill is not None and pd.isna(value) else value
            for key, value in mapping.items()
        }
        target.obs[source_col] = target.obs[index_col].astype(str).map(mapping).values
    else:
        target.obs[source_col] = normalize_cell_source(
            source, missing_fill=missing_fill
        ).values
    return target


def _resolve_config(
    config: MetricsConfig | None,
    overrides: Dict[str, Any],
) -> MetricsConfig:
    base = config or MetricsConfig()
    if not overrides:
        return base
    if "fixed_test_obs_names" in overrides and overrides["fixed_test_obs_names"] is not None:
        overrides = dict(overrides)
        overrides["fixed_test_obs_names"] = tuple(
            canonical_obs_name(name) for name in overrides["fixed_test_obs_names"]
        )
    return replace(base, **overrides)


def _metric_matrices_to_frames(
    metric_matrices: Dict[str, np.ndarray],
    task_order: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    index = list(task_order)
    columns = list(task_order)
    return (
        pd.DataFrame(metric_matrices["accuracy"], index=index, columns=columns),
        pd.DataFrame(metric_matrices["balanced_accuracy"], index=index, columns=columns),
        pd.DataFrame(metric_matrices["macro_f1"], index=index, columns=columns),
    )


def _build_cl_metrics_result(
    *,
    tasks: Sequence[TaskData],
    label_vocab: Sequence[str],
    metric_matrices: Dict[str, np.ndarray],
    per_class_forgetting_df: pd.DataFrame,
    metrics_by_measure: Dict[str, Dict[str, Any]],
    input_h5ad: str | None = None,
    prior_steps_h5ad: str | None = None,
    eval_source: str | None = None,
    eval_test_obs_names: List[str] | None = None,
    split_seed: int | None = None,
) -> CLMetricsResult:
    task_order = [task.task_name for task in tasks]
    accuracy_df, balanced_df, macro_f1_df = _metric_matrices_to_frames(
        metric_matrices, task_order
    )
    accuracy_metrics = metrics_by_measure["accuracy"]
    summary = {
        "final_average_accuracy": float(accuracy_metrics["final_average"]),
        "final_maximum_forgetting": float(accuracy_metrics["final_maximum_forgetting"]),
        "backward_transfer": float(accuracy_metrics["backward_transfer"]),
        "mean_per_class_final_maximum_forgetting": float(
            per_class_forgetting_df["final_maximum_forgetting"].mean(skipna=True)
        ),
    }
    return CLMetricsResult(
        input_h5ad=input_h5ad,
        prior_steps_h5ad=prior_steps_h5ad,
        eval_source=eval_source,
        eval_test_obs_names=eval_test_obs_names,
        split_seed=split_seed,
        task_order=task_order,
        label_vocab=list(label_vocab),
        accuracy_matrix=accuracy_df,
        balanced_accuracy_matrix=balanced_df,
        macro_f1_matrix=macro_f1_df,
        per_class_forgetting=per_class_forgetting_df,
        metrics_by_measure=metrics_by_measure,
        summary=summary,
    )


def extract_embeddings(adata: sc.AnnData, embedding_key: str | None) -> np.ndarray:
    if embedding_key is not None:
        if embedding_key not in adata.obsm:
            raise KeyError(f"Embedding key '{embedding_key}' not found in obsm.")
        return np.asarray(adata.obsm[embedding_key], dtype=np.float32)
    return np.asarray(adata.X, dtype=np.float32)


def build_tasks_from_latent_inputs(
    adata_step2: sc.AnnData,
    adata_prior_steps: sc.AnnData | None,
    *,
    source_col: str,
    integration_step_col: str,
    integration_step_order: Sequence[str],
    prior_integration_steps: Sequence[str],
    default_integration_step: str,
    label_col: str,
    embedding_key: str | None,
    eval_source: str,
    test_size: float,
    seed: int,
    fixed_test_obs_names: Sequence[str] | None = None,
    missing_source_fill: str | None = "case-control CRC",
) -> Tuple[List[TaskData], List[str], List[str]]:
    if source_col not in adata_step2.obs:
        raise KeyError(f"Source column '{source_col}' not found in step-2 adata.obs.")
    if label_col not in adata_step2.obs:
        raise KeyError(f"Label column '{label_col}' not found in step-2 adata.obs.")

    step2 = _filter_labeled_cells(
        _prepare_step2_adata(
            adata_step2,
            source_col=source_col,
            integration_step_col=integration_step_col,
            default_integration_step=default_integration_step,
            label_col=label_col,
            missing_source_fill=missing_source_fill,
        ),
        label_col,
    )

    step_slices: List[Tuple[str, sc.AnnData]] = []
    if adata_prior_steps is not None:
        if label_col not in adata_prior_steps.obs:
            raise KeyError(
                f"Label column '{label_col}' not found in prior-steps adata.obs."
            )
        prior = _filter_labeled_cells(
            _prepare_prior_steps_adata(
                adata_prior_steps,
                source_col=source_col,
                integration_step_col=integration_step_col,
                prior_integration_steps=prior_integration_steps,
                label_col=label_col,
                missing_source_fill=missing_source_fill,
            ),
            label_col,
        )
        for step_name in prior_integration_steps:
            mask = prior.obs[integration_step_col].astype(str) == step_name
            if mask.any():
                step_slices.append(
                    (step_name, _normalize_adata_obs_names(prior[mask].copy()))
                )

    final_step = default_integration_step
    if final_step not in integration_step_order:
        raise ValueError(
            f"default_integration_step {final_step!r} must appear in "
            f"integration_step_order."
        )
    step_slices.append((final_step, _normalize_adata_obs_names(step2)))

    task_steps = [name for name in integration_step_order if any(name == s for s, _ in step_slices)]
    if not task_steps:
        raise ValueError("No integration steps available after merging latent inputs.")

    labels_all = []
    for _, adata_step in step_slices:
        labels_all.extend(adata_step.obs[label_col].astype(str).tolist())
    label_vocab = sorted(set(labels_all))
    label_to_idx = {label: idx for idx, label in enumerate(label_vocab)}

    sources_union = set(step2.obs[source_col].astype(str).tolist())
    if adata_prior_steps is not None:
        sources_union.update(prior.obs[source_col].astype(str).tolist())
    if eval_source not in sources_union:
        raise ValueError(
            f"eval_source {eval_source!r} not found in {source_col}. "
            f"Present: {sorted(sources_union)}"
        )

    split_step = prior_integration_steps[0] if adata_prior_steps is not None else final_step
    split_adata = next((adata for name, adata in step_slices if name == split_step), None)
    if split_adata is None:
        raise ValueError(f"No cells available for eval split at {split_step!r}.")

    split_sources = split_adata.obs[source_col].astype(str).to_numpy()
    split_labels = split_adata.obs[label_col].astype(str).to_numpy()
    split_obs_names = split_adata.obs_names.astype(str)
    split_mask = split_sources == eval_source
    if not split_mask.any():
        raise ValueError(
            f"No {eval_source!r} cells found at integration step {split_step!r} "
            f"for the fixed eval split."
        )

    split_y = np.array(
        [label_to_idx[label] for label in split_labels[split_mask]],
        dtype=np.int64,
    )
    if len(np.unique(split_y)) < 2:
        raise ValueError(
            f"Eval source {eval_source!r} at {split_step!r} has fewer than 2 label "
            "classes; cannot stratify train/test split."
        )
    _, _, eval_test_obs_names = reference_train_test_split(
        split_obs_names[split_mask],
        split_y,
        test_size=test_size,
        seed=seed,
        fixed_test_obs_names=fixed_test_obs_names,
    )
    print(
        f"Fixed eval split on {eval_source!r} at {split_step!r}: "
        f"test={len(eval_test_obs_names):,}, seed={seed} "
        f"(same test barcodes reused at each integration step)",
        flush=True,
    )

    slice_by_step = {name: adata for name, adata in step_slices}
    tasks: List[TaskData] = []
    for step_name in task_steps:
        adata_step = slice_by_step[step_name]
        X = extract_embeddings(adata_step, embedding_key)
        obs_names = adata_step.obs_names.astype(str)
        labels = adata_step.obs[label_col].astype(str).to_numpy()
        step_sources = adata_step.obs[source_col].astype(str).to_numpy()
        y = np.array([label_to_idx[label] for label in labels], dtype=np.int64)
        idx = np.arange(len(y), dtype=np.int64)

        eval_mask = step_sources == eval_source
        if not eval_mask.any():
            train_idx = idx
            test_idx = np.array([], dtype=np.int64)
        else:
            eval_obs_names = obs_names[eval_mask]
            eval_local_idx = np.where(eval_mask)[0]
            name_to_local: Dict[str, int] = {}
            for name, local_idx in zip(eval_obs_names, eval_local_idx):
                raw = str(name)
                name_to_local[raw] = local_idx
                name_to_local[canonical_obs_name(raw)] = local_idx
            missing_test = [
                name for name in eval_test_obs_names if name not in name_to_local
            ]
            if missing_test:
                raise KeyError(
                    f"{len(missing_test)} fixed test obs names are absent from "
                    f"{eval_source!r} cells at integration step {step_name!r}. "
                    f"Examples: {missing_test[:5]}"
                )
            test_idx = np.asarray(
                [name_to_local[name] for name in eval_test_obs_names],
                dtype=np.int64,
            )
            test_name_set = set(eval_test_obs_names)
            ref_train_idx = np.asarray(
                [
                    local_idx
                    for name, local_idx in zip(eval_obs_names, eval_local_idx)
                    if canonical_obs_name(name) not in test_name_set
                ],
                dtype=np.int64,
            )
            non_eval_idx = np.where(~eval_mask)[0]
            train_idx = (
                np.concatenate([non_eval_idx, ref_train_idx])
                if ref_train_idx.size > 0
                else non_eval_idx
            )

        tasks.append(
            TaskData(
                task_name=step_name,
                X=X,
                y=y,
                train_idx=train_idx,
                test_idx=test_idx,
                obs_names=obs_names,
            )
        )
        print(
            f"Task {step_name!r}: {len(y):,} cells, "
            f"{len(np.unique(y))} classes, "
            f"train={len(train_idx):,}, test={len(test_idx):,}",
            flush=True,
        )

    return tasks, label_vocab, eval_test_obs_names


def build_metric_matrices(
    tasks: Sequence[TaskData],
    n_classes: int,
    max_iter: int,
    class_weight: str | None,
    *,
    eval_source: str,
    eval_col_idx: int | None = None,
    cumulative_training: bool,
    classifier: str = DEFAULT_CLASSIFIER,
    knn_neighbors: int = DEFAULT_KNN_NEIGHBORS,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Build R[i, j] = a_{i,j} matrices (HAL / Lopez-Paz convention).

    After integration step ``i``, a classifier is trained (cumulatively on steps
    ``0..i`` by default) and evaluated on the fixed held-out ``eval_source`` test
    barcodes at **every** integration step ``j`` (same cells, embeddings from
    step ``j``). Thus column ``j`` is task ``j`` in the paper metrics.
    """
    del eval_col_idx  # retained for call-site compatibility; full matrix is used
    T = len(tasks)
    metric_matrices = {
        "accuracy": np.full((T, T), np.nan, dtype=np.float64),
        "balanced_accuracy": np.full((T, T), np.nan, dtype=np.float64),
        "macro_f1": np.full((T, T), np.nan, dtype=np.float64),
    }
    per_class_recall = np.full((n_classes, T, T), np.nan, dtype=np.float64)

    for j, task_j in enumerate(tasks):
        if np.asarray(task_j.test_idx, dtype=np.int64).size == 0:
            raise ValueError(
                f"Integration step {task_j.task_name!r} has no held-out "
                f"{eval_source!r} test cells for evaluation."
            )

    for i in range(T):
        if cumulative_training:
            X_train = np.vstack([tasks[k].X[tasks[k].train_idx] for k in range(i + 1)])
            y_train = np.concatenate([tasks[k].y[tasks[k].train_idx] for k in range(i + 1)])
        else:
            X_train = tasks[i].X[tasks[i].train_idx]
            y_train = tasks[i].y[tasks[i].train_idx]

        clf = make_classifier(
            classifier=classifier,
            max_iter=max_iter,
            class_weight=class_weight,
            knn_neighbors=knn_neighbors,
        )
        clf.fit(X_train, y_train)

        train_steps = ", ".join(task.task_name for task in tasks[: i + 1])
        row_acc: List[float] = []
        row_bal: List[float] = []
        row_f1: List[float] = []

        for j in range(T):
            eval_step_task = tasks[j]
            eval_test_idx = np.asarray(eval_step_task.test_idx, dtype=np.int64)
            y_true = eval_step_task.y[eval_test_idx]
            y_pred = clf.predict(eval_step_task.X[eval_test_idx])
            acc, bal_acc, macro_f1 = _score_predictions(y_true, y_pred)

            metric_matrices["accuracy"][i, j] = acc
            metric_matrices["balanced_accuracy"][i, j] = bal_acc
            metric_matrices["macro_f1"][i, j] = macro_f1
            row_acc.append(acc)
            row_bal.append(bal_acc)
            row_f1.append(macro_f1)

            for c in range(n_classes):
                class_mask = y_true == c
                if np.any(class_mask):
                    per_class_recall[c, i, j] = float(
                        np.mean(y_pred[class_mask] == c)
                    )

        print(
            f"Step {i} ({tasks[i].task_name!r}): trained on [{train_steps}], "
            f"evaluated on {eval_source!r} test at all {T} steps | "
            f"diag_acc={row_acc[i]:.4f}, mean_acc={float(np.mean(row_acc)):.4f}, "
            f"mean_bal_acc={float(np.mean(row_bal)):.4f}, "
            f"mean_macro_f1={float(np.mean(row_f1)):.4f}",
            flush=True,
        )

    return metric_matrices, per_class_recall


def _bootstrap_mean_ci(
    values: np.ndarray, bootstrap_iters: int, seed: int, alpha: float = 0.05
) -> Tuple[float, float]:
    finite_vals = values[np.isfinite(values)]
    if finite_vals.size == 0:
        return float("nan"), float("nan")
    if finite_vals.size == 1:
        one = float(finite_vals[0])
        return one, one

    rng = np.random.default_rng(seed)
    sample_means = np.empty(bootstrap_iters, dtype=np.float64)
    n = finite_vals.size

    for b in range(bootstrap_iters):
        sample = rng.choice(finite_vals, size=n, replace=True)
        sample_means[b] = np.mean(sample)

    lower = float(np.quantile(sample_means, alpha / 2.0))
    upper = float(np.quantile(sample_means, 1.0 - alpha / 2.0))
    return lower, upper


def _paper_forgetting_per_task(R: np.ndarray) -> np.ndarray:
    """Per-task forgetting terms for Chaudhry Eq. 3 (0-indexed task columns).

    For the reference / first task (``j=0``), compare peak diagonal retention
    ``R[l, l]`` against final integrated embeddings ``R[T-1, T-1]``.
    """
    T = R.shape[0]
    per_task_forgetting = np.empty(T - 1, dtype=np.float64)
    for j in range(T - 1):
        if j == 0:
            past_scores = np.asarray([R[l, l] for l in range(T - 1)], dtype=np.float64)
            final_perf = float(R[T - 1, T - 1])
        else:
            past_scores = R[: T - 1, j]
            final_perf = float(R[T - 1, j])
        best_before_final = float(np.nanmax(past_scores))
        per_task_forgetting[j] = best_before_final - final_perf
    return per_task_forgetting


def _paper_backward_transfer_per_task(R: np.ndarray) -> np.ndarray:
    """Chaudhry-style BWT: average (a_{T,j} - a_{j,j}) over past tasks."""
    T = R.shape[0]
    bwt_vals = np.empty(T - 1, dtype=np.float64)
    for j in range(T - 1):
        if j == 0:
            bwt_vals[j] = float(R[T - 1, T - 1] - R[0, 0])
        else:
            bwt_vals[j] = float(R[T - 1, j] - R[j, j])
    return bwt_vals


def summarize_metric_matrix(
    R: np.ndarray,
    *,
    bootstrap_iters: int,
    seed: int,
    eval_col_idx: int | None = None,
) -> Dict[str, float | List[float]]:
    """Summarize R[i, j] with Chaudhry/GEM Eqs. 2–3.

    ``A_T = mean_j R[T-1, j]``
    ``F_T = mean_j (max_{l<T} a_{l,j} - a_{T,j})`` with reference-task alignment
    described in :func:`_paper_forgetting_per_task`.
    """
    del eval_col_idx  # retained for call-site compatibility; full matrix is used
    if R.ndim != 2 or R.shape[0] != R.shape[1]:
        raise ValueError(f"Expected square metric matrix, got shape {R.shape}.")
    T = R.shape[0]
    if T == 0:
        raise ValueError("Metric matrix is empty.")

    final_row = R[T - 1, :T]
    if not np.isfinite(final_row).any():
        raise ValueError(
            "Final metric row is all NaN. Check eval_source / task construction."
        )

    # Eq. 2: Accuracy = (1/T) sum_j a_{T,j}
    final_average = float(np.nanmean(final_row))
    final_average_ci = _bootstrap_mean_ci(
        final_row[np.isfinite(final_row)], bootstrap_iters, seed
    )

    if T == 1:
        final_max_forgetting = 0.0
        final_max_forgetting_ci = (0.0, 0.0)
        worst_task_forgetting = 0.0
        backward_transfer = 0.0
    else:
        per_task_forgetting = _paper_forgetting_per_task(R)
        final_max_forgetting = float(np.nanmean(per_task_forgetting))
        worst_task_forgetting = float(np.nanmax(per_task_forgetting))
        final_max_forgetting_ci = _bootstrap_mean_ci(
            per_task_forgetting[np.isfinite(per_task_forgetting)],
            bootstrap_iters,
            seed + 17,
        )
        backward_transfer = float(np.nanmean(_paper_backward_transfer_per_task(R)))

    return {
        "final_average": final_average,
        "final_average_ci95": [float(final_average_ci[0]), float(final_average_ci[1])],
        "final_maximum_forgetting": float(final_max_forgetting),
        "final_maximum_forgetting_ci95": [
            float(final_max_forgetting_ci[0]),
            float(final_max_forgetting_ci[1]),
        ],
        "worst_task_forgetting": float(worst_task_forgetting),
        "backward_transfer": float(backward_transfer),
    }


def compute_per_class_forgetting(
    per_class_recall: np.ndarray,
    label_vocab: Sequence[str],
    *,
    eval_col_idx: int | None = None,
    eval_class_indices: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Per-class A_T / F_T using the same paper averages over integration steps."""
    del eval_col_idx  # retained for call-site compatibility; full matrix is used
    n_classes, T, n_tasks = per_class_recall.shape
    if n_tasks != T:
        raise ValueError(
            f"Expected square per-class recall [C,T,T], got shape {per_class_recall.shape}."
        )
    class_indices = (
        list(eval_class_indices)
        if eval_class_indices is not None
        else list(range(n_classes))
    )
    rows: List[Dict[str, float | str]] = []

    for c in class_indices:
        C = per_class_recall[c, :T, :T]
        if not np.isfinite(C).any():
            continue

        final_row = C[T - 1, :T]
        final_mean_recall = (
            float(np.nanmean(final_row)) if np.isfinite(final_row).any() else float("nan")
        )

        if T == 1:
            class_forgetting = 0.0
            worst_task_forgetting = 0.0
        else:
            per_task_forgetting = _paper_forgetting_per_task(C)
            if np.isfinite(per_task_forgetting).any():
                class_forgetting = float(np.nanmean(per_task_forgetting))
                worst_task_forgetting = float(np.nanmax(per_task_forgetting))
            else:
                class_forgetting = float("nan")
                worst_task_forgetting = float("nan")

        rows.append(
            {
                "class_idx": c,
                "class_label": label_vocab[c],
                "final_mean_recall": final_mean_recall,
                "final_maximum_forgetting": class_forgetting,
                "worst_task_forgetting": worst_task_forgetting,
            }
        )
    return pd.DataFrame(rows)


def compute_cl_metrics(
    adata: sc.AnnData,
    *,
    prior_steps_adata: sc.AnnData | None = None,
    config: MetricsConfig | None = None,
    input_h5ad: str | None = None,
    prior_steps_h5ad: str | None = None,
    **kwargs: Any,
) -> CLMetricsResult:
    """
    Compute continual-learning metrics from a final-step latent and optional prior steps.

    Examples
    --------
    >>> import scanpy as sc
    >>> from cscanvi.metrics import compute_cl_metrics
    >>> adata_final = sc.read("latents/X_scanvi_my_run.h5ad")
    >>> adata_prior = sc.read("prior_steps_X_scANVI.h5ad")
    >>> result = compute_cl_metrics(
    ...     adata_final,
    ...     prior_steps_adata=adata_prior,
    ...     source_col="cell_source",
    ...     label_col="level_2_annot",
    ...     eval_source="reference",
    ...     classifier="knn",
    ...     knn_neighbors=15,
    ... )
    >>> result.summary["final_average_accuracy"]
    >>> result.summary["final_maximum_forgetting"]
    """
    cfg = _resolve_config(config, kwargs)
    if prior_steps_adata is None and cfg.prior_steps_h5ad:
        print(f"Loading prior-steps latent from {cfg.prior_steps_h5ad}", flush=True)
        prior_steps_adata = sc.read(cfg.prior_steps_h5ad)
    if prior_steps_h5ad is None:
        prior_steps_h5ad = cfg.prior_steps_h5ad

    eval_source = resolve_eval_source(cfg.source_order, cfg.eval_source)
    eval_col_idx = None
    print(
        f"Evaluating {cfg.label_col!r} on fixed {eval_source!r} test cells at every "
        f"integration step j (paper A_T/F_T over tasks=steps; "
        f"classifier={cfg.classifier!r}"
        + (f", knn_neighbors={cfg.knn_neighbors}" if cfg.classifier == "knn" else "")
        + f"; integration_step_order={list(cfg.integration_step_order)}, "
        f"prior_steps={'yes' if prior_steps_adata is not None else 'no'}, "
        f"cumulative_training={cfg.cumulative_training})",
        flush=True,
    )

    tasks, label_vocab, eval_test_obs_names = build_tasks_from_latent_inputs(
        adata_step2=adata,
        adata_prior_steps=prior_steps_adata,
        source_col=cfg.source_col,
        integration_step_col=cfg.integration_step_col,
        integration_step_order=cfg.integration_step_order,
        prior_integration_steps=cfg.prior_integration_steps,
        default_integration_step=cfg.default_integration_step,
        label_col=cfg.label_col,
        embedding_key=cfg.embedding_key,
        eval_source=eval_source,
        test_size=cfg.test_size,
        seed=cfg.seed,
        fixed_test_obs_names=cfg.fixed_test_obs_names,
        missing_source_fill=cfg.missing_source_fill,
    )
    eval_class_indices = sorted(
        {
            label
            for task in tasks
            for label in task.y[task.test_idx].tolist()
        }
    )

    metric_matrices, per_class_recall = build_metric_matrices(
        tasks=tasks,
        n_classes=len(label_vocab),
        max_iter=cfg.max_iter,
        class_weight=cfg.class_weight,
        eval_source=eval_source,
        eval_col_idx=eval_col_idx,
        cumulative_training=cfg.cumulative_training,
        classifier=cfg.classifier,
        knn_neighbors=cfg.knn_neighbors,
    )

    metrics_by_measure = {}
    for metric_name, matrix in metric_matrices.items():
        metrics_by_measure[metric_name] = summarize_metric_matrix(
            matrix,
            bootstrap_iters=cfg.bootstrap_iters,
            seed=cfg.seed,
            eval_col_idx=eval_col_idx,
        )

    per_class_forgetting_df = compute_per_class_forgetting(
        per_class_recall,
        label_vocab,
        eval_col_idx=eval_col_idx,
        eval_class_indices=eval_class_indices,
    )
    return _build_cl_metrics_result(
        tasks=tasks,
        label_vocab=label_vocab,
        metric_matrices=metric_matrices,
        per_class_forgetting_df=per_class_forgetting_df,
        metrics_by_measure=metrics_by_measure,
        input_h5ad=input_h5ad,
        prior_steps_h5ad=prior_steps_h5ad,
        eval_source=eval_source,
        eval_test_obs_names=eval_test_obs_names,
        split_seed=cfg.seed,
    )


def compute_cl_metrics_from_path(
    input_path: str | Path,
    *,
    prior_steps_h5ad: str | Path | None = None,
    config: MetricsConfig | None = None,
    **kwargs: Any,
) -> CLMetricsResult:
    """Load a final-step latent (and optional prior-steps h5ad) and compute metrics.

    Examples
    --------
    >>> from cscanvi.metrics import compute_cl_metrics_from_path
    >>> result = compute_cl_metrics_from_path(
    ...     "latents/X_scanvi_my_run.h5ad",
    ...     prior_steps_h5ad="prior_steps_X_scANVI.h5ad",
    ...     source_col="cell_source",
    ...     label_col="level_2_annot",
    ...     eval_source="reference",
    ... )
    >>> result.summary["final_maximum_forgetting"]
    """
    input_path = str(input_path)
    print(f"Loading step-2 latent from {input_path}", flush=True)
    adata = sc.read(input_path)

    prior_path = prior_steps_h5ad
    if prior_path is None and config is not None and config.prior_steps_h5ad:
        prior_path = config.prior_steps_h5ad
    if prior_path is None and "prior_steps_h5ad" in kwargs and kwargs["prior_steps_h5ad"]:
        prior_path = kwargs["prior_steps_h5ad"]

    prior_adata = None
    prior_path_str = None
    if prior_path is not None:
        prior_path_str = str(prior_path)
        print(f"Loading prior-steps latent from {prior_path_str}", flush=True)
        prior_adata = sc.read(prior_path_str)

    return compute_cl_metrics(
        adata,
        prior_steps_adata=prior_adata,
        config=config,
        input_h5ad=input_path,
        prior_steps_h5ad=prior_path_str,
        **kwargs,
    )


def cl_metrics_to_payload(result: CLMetricsResult, *, label_col: str) -> Dict[str, Any]:
    return {
        "input_h5ad": result.input_h5ad,
        "prior_steps_h5ad": result.prior_steps_h5ad,
        "n_tasks": len(result.task_order),
        "n_classes": len(result.label_vocab),
        "task_order": result.task_order,
        "label_col": label_col,
        "eval_source": result.eval_source,
        "split_seed": result.split_seed,
        "n_eval_test_cells": (
            len(result.eval_test_obs_names) if result.eval_test_obs_names else 0
        ),
        "metrics_by_measure": result.metrics_by_measure,
        "mean_per_class_final_maximum_forgetting": result.summary[
            "mean_per_class_final_maximum_forgetting"
        ],
        "final_average_accuracy": result.summary["final_average_accuracy"],
        "final_maximum_forgetting": result.summary["final_maximum_forgetting"],
        "backward_transfer": result.summary["backward_transfer"],
        "notes": {
            "R[i,j]": "Metric on task j after training classifier on task i (0-based).",
            "integration_step_order": result.task_order,
            "eval_source": result.eval_source,
            "label_col": label_col,
        },
    }


def save_cl_metrics(result: CLMetricsResult, output_prefix: Path, *, label_col: str) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    prefix = str(output_prefix)

    result.accuracy_matrix.to_csv(f"{prefix}_accuracy_matrix.csv")
    result.balanced_accuracy_matrix.to_csv(f"{prefix}_balanced_accuracy_matrix.csv")
    result.macro_f1_matrix.to_csv(f"{prefix}_macro_f1_matrix.csv")
    result.per_class_forgetting.to_csv(f"{prefix}_per_class_forgetting.csv", index=False)

    metrics_payload = cl_metrics_to_payload(result, label_col=label_col)
    metrics_path = f"{prefix}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"Saved metric matrices with prefix: {prefix}")
    print(f"Saved per-class forgetting: {prefix}_per_class_forgetting.csv")
    print(f"Saved metrics: {metrics_path}")

    m = result.metrics_by_measure["accuracy"]
    print(
        f"[accuracy] A_T={m['final_average']:.6f} "
        f"F_T={m['final_maximum_forgetting']:.6f} "
        f"BWT={m['backward_transfer']:.6f}"
    )
