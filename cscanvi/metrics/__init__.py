"""Continual-learning and forgetting-rate metrics.

Examples
--------
>>> from cscanvi.metrics import compute_cumulative_fr_scores, write_fr_outputs
>>> result = compute_cumulative_fr_scores(
...     "latents/X_scanvi_my_run.h5ad",
...     "adata_ready/prior_for_FR.h5ad",
...     eval_source="reference",
...     label_col="level_2_annot",
... )
>>> result.summary["final_maximum_forgetting"]
>>> result.summary["final_average_accuracy"]
>>> write_fr_outputs(result, "results/fr_metrics")

>>> from cscanvi.metrics import compute_cl_metrics
>>> result = compute_cl_metrics(
...     adata_final,
...     prior_steps_adata=adata_prior,
...     source_col="cell_source",
...     label_col="level_2_annot",
...     eval_source="reference",
... )
"""

from ._cl import (
    DEFAULT_CLASSIFIER,
    DEFAULT_INTEGRATION_STEP_ORDER,
    DEFAULT_KNN_NEIGHBORS,
    DEFAULT_LEGACY_INTEGRATION_STEP,
    DEFAULT_PRIOR_INTEGRATION_STEPS,
    DEFAULT_SOURCE_ORDER,
    CLMetricsResult,
    MetricsConfig,
    TaskData,
    build_metric_matrices,
    build_tasks_from_latent_inputs,
    cl_metrics_to_payload,
    compute_cl_metrics,
    compute_cl_metrics_from_path,
    compute_per_class_forgetting,
    ensure_integration_step,
    extract_embeddings,
    make_classifier,
    normalize_cell_source,
    parse_integration_step_order,
    parse_source_order,
    prepare_epi_crc_cell_source,
    resolve_eval_source,
    save_cl_metrics,
    summarize_metric_matrix,
)
from ._cumulative import (
    build_prior_latent,
    compute_cumulative_fr_scores,
    infer_step_order,
    merge_fr_summaries,
    write_fr_outputs,
)

__all__ = [
    "CLMetricsResult",
    "DEFAULT_CLASSIFIER",
    "DEFAULT_INTEGRATION_STEP_ORDER",
    "DEFAULT_KNN_NEIGHBORS",
    "DEFAULT_LEGACY_INTEGRATION_STEP",
    "DEFAULT_PRIOR_INTEGRATION_STEPS",
    "DEFAULT_SOURCE_ORDER",
    "MetricsConfig",
    "TaskData",
    "build_metric_matrices",
    "build_prior_latent",
    "build_tasks_from_latent_inputs",
    "cl_metrics_to_payload",
    "compute_cl_metrics",
    "compute_cl_metrics_from_path",
    "compute_cumulative_fr_scores",
    "compute_per_class_forgetting",
    "ensure_integration_step",
    "extract_embeddings",
    "infer_step_order",
    "make_classifier",
    "merge_fr_summaries",
    "normalize_cell_source",
    "parse_integration_step_order",
    "parse_source_order",
    "prepare_epi_crc_cell_source",
    "resolve_eval_source",
    "save_cl_metrics",
    "summarize_metric_matrix",
    "write_fr_outputs",
]
