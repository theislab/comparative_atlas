"""Fisher-scored dataset-order search (MCTS, beam, exact DP).

Examples
--------
>>> from cscanvi.order_search import search_dataset_orders
>>> result = search_dataset_orders(
...     "fisher_cache.npz",
...     output_dir="results/order_search",
...     lams=(0.0, 1.0),
...     methods=("mcts", "beam", "exact"),
... )
>>> result["results"][0]["recommended_order"]

>>> from cscanvi.order_search import run_dataset_order_search
>>> result = run_dataset_order_search(
...     "results/order_search",
...     query_h5ad="query.h5ad",
...     reference_model="models/ref",
...     reference_h5ad="ref.h5ad",
...     lams=(0.0, 0.5, 1.0, 2.0),
... )
"""

from ._search import (
    DEFAULT_CONTROL_VALUES,
    DEFAULT_DISEASE_KEY,
    DEFAULT_REQUIRED_OBS,
    DEFAULT_STUDY_KEY,
    TINY,
    CostModel,
    FisherProfiles,
    MCTSNode,
    MinMaxStats,
    Ordering,
    beam_search,
    candidate_studies,
    control_fisher_for_study,
    exact_search,
    extract_control_fisher,
    load_fisher_cache,
    mcts_search,
    objective,
    orderings_frame,
    read_cells,
    read_obs,
    run_dataset_order_search,
    scanpy_subsample_indices,
    search_dataset_orders,
)

__all__ = [
    "DEFAULT_CONTROL_VALUES",
    "DEFAULT_DISEASE_KEY",
    "DEFAULT_REQUIRED_OBS",
    "DEFAULT_STUDY_KEY",
    "TINY",
    "CostModel",
    "FisherProfiles",
    "MCTSNode",
    "MinMaxStats",
    "Ordering",
    "beam_search",
    "candidate_studies",
    "control_fisher_for_study",
    "exact_search",
    "extract_control_fisher",
    "load_fisher_cache",
    "mcts_search",
    "objective",
    "orderings_frame",
    "read_cells",
    "read_obs",
    "run_dataset_order_search",
    "scanpy_subsample_indices",
    "search_dataset_orders",
]
