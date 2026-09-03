"""Monte Carlo tree search over dataset integration orders, scored by control Fisher.

Stage ``fisher`` evaluates the reference model's diagonal Fisher on each candidate
dataset's control cells via ``SCANVI.load_query_data_with_replay`` and caches the
per-parameter tensors to npz. This is the only stage that touches the model.

Large query h5ad files are never read whole: only sampled control rows are pulled
off disk.

Stage ``search`` treats orderings as root-to-leaf paths in a prefix tree and
explores it with UCT Monte Carlo tree search. The cost of appending dataset
``d`` to prefix ``S`` is

    cost(d | S) = log10( sum_i F_d,i / (1 + lam * Fhat_S,i) + tail_d )

where ``Fhat_S`` is the mean-normalised Fisher profile accumulated over ``S``.
At ``lam=0`` this is ``log10 Tr(F_d)`` and every ordering ties. For ``lam>0``,
curvature that ``d`` places on directions the model has already hardened around
is discounted (``interaction='redundancy'``) or amplified
(``interaction='conflict'``).

When ``decay == 1`` the cost of a step depends on the prefix only as a set, so
the cost model can be tabulated over ``2^n`` subsets and an exact Held-Karp
optimum is available as an oracle.
"""

from __future__ import annotations

import gc
import json
import math
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

TINY = 1e-300
DEFAULT_STUDY_KEY = "study"
DEFAULT_DISEASE_KEY = "disease_unified"
DEFAULT_CONTROL_VALUES = ("Normal",)
DEFAULT_REQUIRED_OBS = ("fine_annot", "donorID_unified", "log1p_n_counts", "percent_mito")


def _as_path(path: str | Path) -> Path:
    return Path(path)


def parse_use_gpu(value: str | int | bool | None):
    """Normalise a GPU selector to scvi's ``use_gpu`` argument."""
    if value is None or isinstance(value, (bool, int)):
        return value
    lowered = str(value).strip().lower()
    if lowered in ("auto", "none", ""):
        return None
    if lowered in ("true", "yes", "1"):
        return True
    if lowered in ("false", "no", "0"):
        return False
    return int(value)


# --------------------------------------------------------------------------------------
# Stage 1: targeted h5ad reads
# --------------------------------------------------------------------------------------


def _sparse_dataset(node):
    try:
        from anndata._core.sparse_dataset import SparseDataset
    except ImportError:  # anndata >= 0.10
        from anndata.experimental import sparse_dataset as SparseDataset
    return SparseDataset(node)


def read_obs(path: Path) -> pd.DataFrame:
    import h5py
    from anndata._io.specs import read_elem

    with h5py.File(path, "r") as fh:
        obs = read_elem(fh["obs"])
    return obs.loc[:, ~obs.columns.duplicated()]


def read_cells(path: Path, row_idx: np.ndarray, obs: pd.DataFrame | None = None):
    """Materialise only ``row_idx`` from an h5ad, preserving the requested row order.

    ``row_idx`` is read in sorted order because the on-disk CSR reader requires it, then
    permuted back: minibatch composition depends on cell order, so the order has to match
    what an in-memory ``sc.pp.subsample`` would have produced.
    """
    import anndata as ad
    import h5py
    from anndata._io.specs import read_elem

    row_idx = np.asarray(row_idx)
    order = np.argsort(row_idx, kind="stable")
    sorted_idx = row_idx[order]
    restore = np.argsort(order, kind="stable")

    with h5py.File(path, "r") as fh:
        if obs is None:
            obs = read_elem(fh["obs"])
            obs = obs.loc[:, ~obs.columns.duplicated()]
        var = read_elem(fh["var"])
        source = "layers/counts" if ("layers" in fh and "counts" in fh["layers"]) else "X"
        node = fh[source]
        if isinstance(node, h5py.Group) and "data" in node:
            matrix = _sparse_dataset(node)[sorted_idx]
        else:
            matrix = node[sorted_idx, :]
        matrix = matrix.astype(np.float32)[restore]
        adata = ad.AnnData(X=matrix, obs=obs.iloc[sorted_idx].iloc[restore].copy(), var=var)

    adata.obs_names_make_unique()
    return adata


def scanpy_subsample_indices(
    n: int, *, fraction: float | None = None, n_obs: int | None = None, seed: int = 0
) -> np.ndarray:
    """Index draw identical to ``sc.pp.subsample``, so cached Fisher values stay
    comparable with the numbers already produced in the notebook."""
    np.random.seed(seed)
    if n_obs is not None:
        size = min(int(n_obs), n)
    elif fraction is not None:
        size = int(fraction * n)
    else:
        raise ValueError("Provide either fraction or n_obs.")
    return np.random.choice(n, size=size, replace=False)


def candidate_studies(
    obs: pd.DataFrame,
    exclude: Sequence[str],
    min_ctrl_cells: int,
    *,
    study_key: str = DEFAULT_STUDY_KEY,
    disease_key: str = DEFAULT_DISEASE_KEY,
    control_values: Sequence[str] = DEFAULT_CONTROL_VALUES,
) -> list[str]:
    study = obs[study_key].astype(str)
    is_ctrl = obs[disease_key].astype(str).isin(tuple(control_values))
    n_ctrl = is_ctrl.groupby(study).sum()

    kept, dropped = [], {}
    exclude_set = set(exclude)
    for name in sorted(n_ctrl.index):
        if name in exclude_set:
            dropped[name] = "excluded"
        elif int(n_ctrl[name]) < min_ctrl_cells:
            dropped[name] = f"only {int(n_ctrl[name])} control cells"
        else:
            kept.append(name)
    print(f"Candidate studies ({len(kept)}): {kept}", flush=True)
    if dropped:
        print(f"Dropped: {json.dumps(dropped, indent=2)}", flush=True)
    if len(kept) < 2:
        raise ValueError("Need at least two candidate studies to search over an ordering.")
    return kept


def control_fisher_for_study(
    study_name: str,
    ctrl_adata,
    replay_adata,
    ref_model: str,
    use_gpu,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Diagonal Fisher of the reference model on one study's control cells.

    ``load_query_data_with_replay`` extends the batch categories with this study's donor
    and computes ``ctrl_importances`` against the batch-extended reference weights. The
    adata passed in carries only the control cells rather than the whole study: since
    donor is 1:1 with study here and all labels are 'Unknown', the registry it produces
    (n_batch, n_labels, n_vars) is identical to the full study's, and the Fisher reads
    only ``uns['ctrl_query']`` anyway.

    The incoming donor's batch-embedding columns do not exist in the checkpoint and are
    left at their random initialisation, so Tr(F) depends on the torch RNG state. Seeding
    per study keeps the numbers reproducible and puts every study on the same footing.
    """
    import torch

    from cscanvi._scanvi import SCANVI

    torch.manual_seed(seed)
    np.random.seed(seed)

    test_dl = ctrl_adata.copy()
    test_dl.uns["ctrl_query"] = ctrl_adata.copy()
    test_dl.uns["replay_adata"] = replay_adata.copy()

    model = SCANVI.load_query_data_with_replay(
        test_dl,
        reference_model=ref_model,
        unfrozen=True,
        control_uns_key="ctrl_query",
        replay_uns_key="replay_adata",
        use_gpu=use_gpu,
    )
    fisher = {
        name: tensor.detach().cpu().numpy().astype(np.float32)
        for name, tensor in model.module.ctrl_importances
    }

    del model, test_dl
    gc.collect()

    bad = [name for name, arr in fisher.items() if not np.isfinite(arr).all()]
    if bad:
        raise FloatingPointError(f"{study_name}: non-finite Fisher entries in {bad[:5]}")
    return fisher


def extract_control_fisher(
    query_h5ad: str | Path,
    reference_model: str | Path,
    reference_h5ad: str | Path,
    output_dir: str | Path,
    *,
    study_key: str = DEFAULT_STUDY_KEY,
    disease_key: str = DEFAULT_DISEASE_KEY,
    control_values: Sequence[str] = DEFAULT_CONTROL_VALUES,
    required_obs: Sequence[str] = DEFAULT_REQUIRED_OBS,
    exclude_studies: Sequence[str] = (),
    ctrl_cells: int = 2000,
    ctrl_frac: float = 0.2,
    min_ctrl_cells: int = 64,
    replay_cells: int = 512,
    seed: int = 0,
    fisher_repeats: int = 3,
    use_gpu: str | int | bool | None = "auto",
    overwrite: bool = False,
) -> Path:
    """Compute per-dataset control Fisher and write ``fisher_cache.npz``.

    Returns the cache path. Reuses an existing cache unless ``overwrite`` is True.

    Examples
    --------
    >>> from cscanvi.order_search import extract_control_fisher
    >>> cache = extract_control_fisher(
    ...     query_h5ad="adata_ready/ten_crc_query.h5ad",
    ...     reference_model="models/preterm_ord_step_ewc1em2_ep50_R2",
    ...     reference_h5ad="07072026_fetal_oliver2024_panGI_EWC2.0_replay0.2_100epoch_emb_counts.h5ad",
    ...     output_dir="results/dataset_order_search/my_run",
    ...     ctrl_cells=2000,
    ...     fisher_repeats=3,
    ... )
    """
    import scvi

    out_dir = _as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = out_dir / "fisher_cache.npz"
    if cache_path.exists() and not overwrite:
        print(f"Reusing Fisher cache at {cache_path} (pass overwrite=True to recompute).", flush=True)
        return cache_path

    ref_model = str(_as_path(reference_model))
    query_path = _as_path(query_h5ad)
    ref_path = _as_path(reference_h5ad)
    gpu = parse_use_gpu(use_gpu)
    control_values = tuple(control_values)

    print(f"Reading query obs from {query_path}", flush=True)
    obs = read_obs(query_path)
    missing = [c for c in (study_key, disease_key, *required_obs) if c not in obs.columns]
    if missing:
        raise KeyError(f"Query obs is missing required columns: {missing}")
    studies = candidate_studies(
        obs,
        list(exclude_studies),
        min_ctrl_cells,
        study_key=study_key,
        disease_key=disease_key,
        control_values=control_values,
    )

    print(f"Reading {replay_cells} replay cells from {ref_path}", flush=True)
    ref_obs = read_obs(ref_path)
    replay_idx = scanpy_subsample_indices(len(ref_obs), n_obs=replay_cells, seed=seed)
    replay_adata = read_cells(ref_path, replay_idx, obs=ref_obs)
    scvi.model.SCANVI.prepare_query_anndata(replay_adata, ref_model)
    del ref_obs
    gc.collect()

    study_col = obs[study_key].astype(str).to_numpy()
    ctrl_mask = obs[disease_key].astype(str).isin(control_values).to_numpy()

    payload: dict[str, np.ndarray] = {}
    rows = []
    for study_name in studies:
        ctrl_rows = np.flatnonzero((study_col == study_name) & ctrl_mask)
        if ctrl_cells > 0:
            pick = scanpy_subsample_indices(len(ctrl_rows), n_obs=ctrl_cells, seed=seed)
        else:
            pick = scanpy_subsample_indices(len(ctrl_rows), fraction=ctrl_frac, seed=seed)
        selected = ctrl_rows[pick]
        if len(selected) < min_ctrl_cells:
            raise ValueError(
                f"{study_name}: only {len(selected)} control cells after subsampling "
                f"(pool={len(ctrl_rows)}); raise ctrl_cells or lower min_ctrl_cells."
            )

        print(f"[{study_name}] reading {len(selected):,} control cells ...", flush=True)
        ctrl_adata = read_cells(query_path, selected, obs=obs)
        scvi.model.SCANVI.prepare_query_anndata(ctrl_adata, ref_model)

        accumulated: dict[str, np.ndarray] | None = None
        seed_traces = []
        n_repeats = max(fisher_repeats, 1)
        for repeat in range(n_repeats):
            fisher = control_fisher_for_study(
                study_name, ctrl_adata, replay_adata, ref_model, gpu, seed=seed + repeat
            )
            seed_traces.append(
                float(np.log10(max(sum(a.sum(dtype=np.float64) for a in fisher.values()), TINY)))
            )
            if accumulated is None:
                accumulated = {k: v.astype(np.float64) for k, v in fisher.items()}
            else:
                for key, value in fisher.items():
                    accumulated[key] += value
            del fisher
            gc.collect()

        assert accumulated is not None
        averaged = {k: (v / n_repeats).astype(np.float32) for k, v in accumulated.items()}
        trace = float(sum(arr.sum(dtype=np.float64) for arr in averaged.values()))
        for param, arr in averaged.items():
            payload[f"{study_name}::{param}"] = arr
        rows.append(
            {
                "dataset": study_name,
                "n_ctrl_pool": int(len(ctrl_rows)),
                "n_ctrl_cells": int(len(selected)),
                "n_params": int(sum(arr.size for arr in averaged.values())),
                "n_repeats": n_repeats,
                "trace_F": trace,
                "log10_trace_F": float(np.log10(max(trace, TINY))),
                "log10_trace_seed_min": min(seed_traces),
                "log10_trace_seed_max": max(seed_traces),
                "log10_trace_seed_spread": max(seed_traces) - min(seed_traces),
            }
        )
        print(
            f"  study {study_name}: log10 Tr(F) = {np.log10(max(trace, TINY)):.6f} "
            f"(seed spread {max(seed_traces) - min(seed_traces):.4f} over {n_repeats} seeds)",
            flush=True,
        )

        del accumulated, averaged, ctrl_adata
        gc.collect()

    np.savez_compressed(cache_path, **payload)
    summary = pd.DataFrame(rows).sort_values("log10_trace_F").reset_index(drop=True)
    summary.to_csv(out_dir / "fisher_summary.csv", index=False)

    meta = {
        "reference_model": ref_model,
        "query_h5ad": str(query_path),
        "reference_h5ad": str(ref_path),
        "study_key": study_key,
        "disease_key": disease_key,
        "control_values": list(control_values),
        "ctrl_cells": ctrl_cells,
        "ctrl_frac": ctrl_frac,
        "replay_cells": replay_cells,
        "seed": seed,
        "fisher_repeats": fisher_repeats,
        "datasets": studies,
    }
    with open(out_dir / "fisher_config.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    print(f"\nWrote Fisher cache to {cache_path}", flush=True)
    print(summary.to_string(index=False), flush=True)

    del payload, replay_adata, obs
    gc.collect()
    return cache_path


# --------------------------------------------------------------------------------------
# Stage 2: Fisher profiles and the ordering cost
# --------------------------------------------------------------------------------------


def load_fisher_cache(cache_path: str | Path) -> dict[str, dict[str, np.ndarray]]:
    """Read the npz cache back into ``{dataset: {param_name: array}}``."""
    per_dataset: dict[str, dict[str, np.ndarray]] = {}
    with np.load(cache_path) as handle:
        for key in handle.files:
            dataset, param = key.split("::", 1)
            per_dataset.setdefault(dataset, {})[param] = handle[key]
    return per_dataset


class FisherProfiles:
    """Per-dataset Fisher traces plus a cross-dataset comparable coordinate system.

    Query-time category extension grows the last axis of the batch-facing weights by a
    different amount per dataset, so those coordinates are comparable only up to the
    shared reference prefix. Parameters are truncated to the common last-axis length for
    the interaction term; whatever is truncated away is carried in ``tail`` and added
    unweighted, so ``lam=0`` still reproduces the full ``Tr(F)`` exactly.
    """

    def __init__(self, per_dataset: dict[str, dict[str, np.ndarray]]):
        self.datasets = sorted(per_dataset)
        self.trace = {
            d: float(sum(arr.sum(dtype=np.float64) for arr in per_dataset[d].values()))
            for d in self.datasets
        }

        shared = set(per_dataset[self.datasets[0]])
        for d in self.datasets[1:]:
            shared &= set(per_dataset[d])

        blocks: dict[str, list[np.ndarray]] = {d: [] for d in self.datasets}
        self.aligned_params: list[str] = []
        self.truncated_params: dict[str, tuple[int, int]] = {}
        for param in sorted(shared):
            shapes = [per_dataset[d][param].shape for d in self.datasets]
            if len({s[:-1] for s in shapes}) > 1:
                continue
            keep = min(s[-1] for s in shapes)
            widest = max(s[-1] for s in shapes)
            if keep != widest:
                self.truncated_params[param] = (keep, widest)
            for d in self.datasets:
                blocks[d].append(np.asarray(per_dataset[d][param][..., :keep], dtype=np.float64).ravel())
            self.aligned_params.append(param)

        self.vec = {d: np.concatenate(blocks[d]) for d in self.datasets}
        self.mean = {d: float(max(self.vec[d].mean(), TINY)) for d in self.datasets}
        self.tail = {d: self.trace[d] - float(self.vec[d].sum()) for d in self.datasets}
        self.n_aligned = int(self.vec[self.datasets[0]].size)

    def accumulated_profile(self, path: tuple[str, ...], decay: float) -> np.ndarray | None:
        """Mean-normalised Fisher profile of everything already integrated.

        Each dataset is scaled to unit mean before accumulating so the profile encodes
        *which* directions are loaded rather than the raw magnitude, which would let a
        single high-trace study dominate the accumulator purely by scale.
        """
        if not path:
            return None
        acc = np.zeros(self.n_aligned, dtype=np.float64)
        for age, dataset in enumerate(reversed(path)):
            acc += (decay**age) * (self.vec[dataset] / self.mean[dataset])
        scale = float(acc.mean())
        return acc / scale if scale > 0 else None

    def step_cost(self, dataset: str, acc: np.ndarray | None, lam: float, interaction: str) -> float:
        """log10 of the (re-weighted) control Fisher trace for adding ``dataset``."""
        if acc is None or lam == 0.0:
            total = self.trace[dataset]
        else:
            weight = 1.0 + lam * acc
            if interaction == "redundancy":
                core = float((self.vec[dataset] / weight).sum())
            else:
                core = float((self.vec[dataset] * weight).sum())
            total = core + self.tail[dataset]
        return float(np.log10(max(total, TINY)))

    def similarity_matrix(self) -> pd.DataFrame:
        """Cosine similarity between dataset Fisher profiles -- what drives the search."""
        names = self.datasets
        mat = np.zeros((len(names), len(names)))
        norms = {d: float(np.linalg.norm(self.vec[d])) for d in names}
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                denom = norms[a] * norms[b]
                mat[i, j] = float(self.vec[a] @ self.vec[b]) / denom if denom > 0 else np.nan
        return pd.DataFrame(mat, index=names, columns=names)


class CostModel:
    """Step costs, tabulated over subsets when possible.

    With ``decay == 1`` the accumulated profile is a mean over the prefix and therefore
    permutation invariant, so a step cost is a function of (prefix set, candidate). For
    n datasets that is n*2^(n-1) distinct values -- 1024 at n=8 -- which can be computed
    once. Every search then runs on scalar lookups instead of 3.3M-element array passes.
    """

    def __init__(
        self,
        profiles: FisherProfiles,
        *,
        lam: float,
        interaction: str,
        decay: float,
        tabulate_max: int = 14,
        profile_cache_size: int = 4,
    ):
        self.profiles = profiles
        self.lam = lam
        self.interaction = interaction
        self.decay = decay
        self.datasets = profiles.datasets
        self.set_indexed = decay == 1.0
        self.table: dict[tuple[frozenset, str], float] | None = None
        self._profile_cache: OrderedDict[tuple[str, ...], np.ndarray | None] = OrderedDict()
        self._profile_cache_size = profile_cache_size
        self.n_evaluations = 0

        if self.set_indexed and len(self.datasets) <= tabulate_max:
            self._tabulate()

    def _profile(self, path: tuple[str, ...]) -> np.ndarray | None:
        key = tuple(sorted(path)) if self.set_indexed else path
        if key in self._profile_cache:
            self._profile_cache.move_to_end(key)
            return self._profile_cache[key]
        value = self.profiles.accumulated_profile(path, self.decay)
        self._profile_cache[key] = value
        if len(self._profile_cache) > self._profile_cache_size:
            self._profile_cache.popitem(last=False)
        return value

    def _tabulate(self) -> None:
        table: dict[tuple[frozenset, str], float] = {}
        n = len(self.datasets)
        for size in range(n):
            for members in combinations(self.datasets, size):
                acc = self.profiles.accumulated_profile(members, 1.0)
                key = frozenset(members)
                for candidate in self.datasets:
                    if candidate in key:
                        continue
                    table[(key, candidate)] = self.profiles.step_cost(
                        candidate, acc, self.lam, self.interaction
                    )
                    self.n_evaluations += 1
        self.table = table

    def cost(self, prefix: tuple[str, ...], candidate: str) -> float:
        if self.table is not None:
            return self.table[(frozenset(prefix), candidate)]
        self.n_evaluations += 1
        return self.profiles.step_cost(candidate, self._profile(prefix), self.lam, self.interaction)

    def path_costs(self, path: tuple[str, ...]) -> tuple[float, ...]:
        return tuple(self.cost(path[:k], path[k]) for k in range(len(path)))


@dataclass(frozen=True)
class Ordering:
    path: tuple[str, ...]
    costs: tuple[float, ...]

    @property
    def total(self) -> float:
        return float(sum(self.costs))

    @property
    def worst(self) -> float:
        return float(max(self.costs)) if self.costs else 0.0

    def key(self, aggregate: str) -> float:
        """Scalar objective, for reporting and for search-vs-oracle gaps."""
        return self.total if aggregate == "sum" else self.worst

    def sort_key(self, aggregate: str) -> tuple[float, float]:
        """Under minimax most prefixes tie on the worst step, so the total breaks ties;
        without it a beam has nothing to steer on."""
        return (self.worst, self.total) if aggregate == "max" else (self.total, self.worst)


def objective(costs: tuple[float, ...], aggregate: str) -> float:
    return float(sum(costs)) if aggregate == "sum" else float(max(costs))


# --------------------------------------------------------------------------------------
# Monte Carlo tree search
# --------------------------------------------------------------------------------------


class MinMaxStats:
    """Running bounds used to map costs onto [0, 1] at selection time.

    Normalising during selection rather than at backup keeps every stored statistic on
    the raw cost scale, so bounds discovered late do not invalidate early backups.
    """

    def __init__(self) -> None:
        self.minimum = math.inf
        self.maximum = -math.inf

    def update(self, value: float) -> None:
        self.minimum = min(self.minimum, value)
        self.maximum = max(self.maximum, value)

    def normalise(self, value: float) -> float:
        """Map a cost to [0, 1] with 1 = cheapest seen, since we minimise."""
        if self.maximum > self.minimum:
            return (self.maximum - value) / (self.maximum - self.minimum)
        return 0.5


@dataclass
class MCTSNode:
    prefix: tuple[str, ...]
    untried: list[str]
    children: dict[str, "MCTSNode"] = field(default_factory=dict)
    visits: int = 0
    cost_sum: float = 0.0
    best_cost: float = math.inf

    @property
    def mean_cost(self) -> float:
        return self.cost_sum / self.visits if self.visits else math.inf


def _remaining(datasets: list[str], prefix: tuple[str, ...]) -> list[str]:
    used = set(prefix)
    return [d for d in datasets if d not in used]


def _ordered_untried(cost_model: CostModel, prefix: tuple[str, ...]) -> list[str]:
    """Expand cheapest-first: a weak prior that costs nothing once costs are tabulated."""
    remaining = _remaining(cost_model.datasets, prefix)
    return sorted(remaining, key=lambda d: cost_model.cost(prefix, d))


def _rollout(
    cost_model: CostModel,
    prefix: tuple[str, ...],
    costs: tuple[float, ...],
    policy: str,
    eps: float,
    rng: np.random.Generator,
) -> Ordering:
    path, path_costs = prefix, list(costs)
    while True:
        remaining = _remaining(cost_model.datasets, path)
        if not remaining:
            break
        if policy == "random" or (policy == "eps-greedy" and rng.random() < eps):
            choice = remaining[int(rng.integers(len(remaining)))]
        else:
            choice = min(remaining, key=lambda d: cost_model.cost(path, d))
        path_costs.append(cost_model.cost(path, choice))
        path = path + (choice,)
    return Ordering(path=path, costs=tuple(path_costs))


def _uct_select(node: MCTSNode, c_uct: float, stats: MinMaxStats, backup: str) -> MCTSNode:
    log_parent = math.log(max(node.visits, 1))
    best_score, best_child = -math.inf, None
    for child in node.children.values():
        if backup == "mean":
            value = child.mean_cost
        elif backup == "best":
            value = child.best_cost
        else:
            value = 0.5 * (child.mean_cost + child.best_cost)
        exploit = stats.normalise(value)
        explore = c_uct * math.sqrt(log_parent / child.visits)
        score = exploit + explore
        if score > best_score:
            best_score, best_child = score, child
    assert best_child is not None
    return best_child


def mcts_search(
    cost_model: CostModel,
    *,
    aggregate: str,
    n_simulations: int,
    c_uct: float,
    backup: str,
    rollout: str,
    rollout_eps: float,
    seed: int,
    restarts: int = 1,
) -> tuple[Ordering, pd.DataFrame, pd.DataFrame]:
    """UCT search over the ordering prefix tree.

    Returns the best ordering found, the root action statistics, and the
    best-so-far convergence trace.

    Examples
    --------
    >>> from cscanvi.order_search import CostModel, FisherProfiles, load_fisher_cache, mcts_search
    >>> profiles = FisherProfiles(load_fisher_cache("fisher_cache.npz"))
    >>> cost_model = CostModel(profiles, lam=1.0, interaction="redundancy", decay=1.0)
    >>> best, roots, convergence = mcts_search(
    ...     cost_model,
    ...     aggregate="sum",
    ...     n_simulations=20000,
    ...     c_uct=1.0,
    ...     backup="mixed",
    ...     rollout="eps-greedy",
    ...     rollout_eps=0.25,
    ...     seed=0,
    ... )
    >>> best.path, best.total
    """
    datasets = cost_model.datasets
    overall_best: Ordering | None = None
    convergence_rows: list[dict] = []
    root_rows: list[dict] = []

    for restart in range(restarts):
        rng = np.random.default_rng(seed + restart)
        stats = MinMaxStats()
        root = MCTSNode(prefix=(), untried=_ordered_untried(cost_model, ()))
        best: Ordering | None = None

        for sim in range(1, n_simulations + 1):
            node = root
            visited = [root]

            # Selection: descend fully expanded nodes.
            while not node.untried and node.children:
                node = _uct_select(node, c_uct, stats, backup)
                visited.append(node)

            # Expansion: take one untried action, cheapest first.
            if node.untried:
                action = node.untried.pop(0)
                prefix = node.prefix + (action,)
                child = MCTSNode(prefix=prefix, untried=_ordered_untried(cost_model, prefix))
                node.children[action] = child
                visited.append(child)
                node = child

            # Simulation and backup.
            costs = cost_model.path_costs(node.prefix)
            result = _rollout(cost_model, node.prefix, costs, rollout, rollout_eps, rng)
            value = objective(result.costs, aggregate)
            stats.update(value)
            for visited_node in visited:
                visited_node.visits += 1
                visited_node.cost_sum += value
                visited_node.best_cost = min(visited_node.best_cost, value)

            if best is None or value < best.key(aggregate):
                best = result
                convergence_rows.append(
                    {"restart": restart, "simulation": sim, "best_objective": value,
                     "order": " > ".join(result.path)}
                )

        assert best is not None
        for action, child in sorted(root.children.items(), key=lambda kv: kv[1].best_cost):
            root_rows.append(
                {
                    "restart": restart,
                    "first_dataset": action,
                    "visits": child.visits,
                    "visit_share": child.visits / max(root.visits, 1),
                    "mean_objective": child.mean_cost,
                    "best_objective": child.best_cost,
                }
            )
        if overall_best is None or best.key(aggregate) < overall_best.key(aggregate):
            overall_best = best

    assert overall_best is not None
    return overall_best, pd.DataFrame(root_rows), pd.DataFrame(convergence_rows)


# --------------------------------------------------------------------------------------
# Reference searches
# --------------------------------------------------------------------------------------


def beam_search(
    cost_model: CostModel, *, aggregate: str, beam_width: int
) -> tuple[list[Ordering], pd.DataFrame]:
    """Keep the ``beam_width`` cheapest prefixes at each depth.

    Examples
    --------
    >>> from cscanvi.order_search import CostModel, FisherProfiles, load_fisher_cache, beam_search
    >>> profiles = FisherProfiles(load_fisher_cache("fisher_cache.npz"))
    >>> cost_model = CostModel(profiles, lam=1.0, interaction="redundancy", decay=1.0)
    >>> beam, tree = beam_search(cost_model, aggregate="sum", beam_width=8)
    >>> beam[0].path, beam[0].total
    """
    datasets = cost_model.datasets
    beam = [Ordering(path=(), costs=())]
    trace_rows = []

    for depth in range(len(datasets)):
        children: list[Ordering] = []
        for node in beam:
            for candidate in _remaining(datasets, node.path):
                cost = cost_model.cost(node.path, candidate)
                children.append(
                    Ordering(path=node.path + (candidate,), costs=node.costs + (cost,))
                )
        children.sort(key=lambda n: n.sort_key(aggregate))
        kept = children[:beam_width]
        kept_paths = {n.path for n in kept}
        for child in children:
            trace_rows.append(
                {
                    "depth": depth + 1,
                    "prefix": " > ".join(child.path[:-1]),
                    "candidate": child.path[-1],
                    "step_cost": child.costs[-1],
                    "cum_cost": child.total,
                    "worst_step": child.worst,
                    "kept_in_beam": child.path in kept_paths,
                }
            )
        beam = kept

    beam.sort(key=lambda n: n.sort_key(aggregate))
    return beam, pd.DataFrame(trace_rows)


def exact_search(cost_model: CostModel, *, aggregate: str) -> Ordering:
    """Held-Karp optimum over orderings.

    Valid only when the step cost sees the prefix through its set (``decay == 1``), which
    collapses the n! orderings onto 2^n states. Under ``aggregate='max'`` the minimax
    value is exact because max is monotone; the total is only a tie-break among prefixes
    that already share that value.

    Examples
    --------
    >>> from cscanvi.order_search import CostModel, FisherProfiles, exact_search, load_fisher_cache
    >>> profiles = FisherProfiles(load_fisher_cache("fisher_cache.npz"))
    >>> cost_model = CostModel(profiles, lam=1.0, interaction="redundancy", decay=1.0)
    >>> optimum = exact_search(cost_model, aggregate="sum")
    >>> optimum.path, optimum.total
    """
    datasets = cost_model.datasets
    n = len(datasets)

    def score_of(costs: tuple[float, ...]) -> tuple[float, float]:
        return (max(costs), sum(costs)) if aggregate == "max" else (sum(costs), max(costs))

    best: dict[int, tuple[tuple[float, float], tuple[str, ...], tuple[float, ...]]] = {
        0: ((0.0, 0.0), (), ())
    }
    for size in range(n):
        for members in combinations(range(n), size):
            mask = 0
            for bit in members:
                mask |= 1 << bit
            if mask not in best:
                continue
            _, path, costs = best[mask]
            for bit in range(n):
                if mask & (1 << bit):
                    continue
                new_costs = costs + (cost_model.cost(path, datasets[bit]),)
                new_score = score_of(new_costs)
                new_mask = mask | (1 << bit)
                if new_mask not in best or new_score < best[new_mask][0]:
                    best[new_mask] = (new_score, path + (datasets[bit],), new_costs)

    _, path, costs = best[(1 << n) - 1]
    return Ordering(path=path, costs=costs)


def orderings_frame(nodes: list[Ordering], lam: float, source: str) -> pd.DataFrame:
    rows = []
    for rank, node in enumerate(nodes, start=1):
        row = {
            "lam": lam,
            "source": source,
            "rank": rank,
            "order": " > ".join(node.path),
            "total_cost": node.total,
            "mean_cost": node.total / len(node.costs),
            "worst_step": node.worst,
            "first_dataset": node.path[0],
            "last_dataset": node.path[-1],
        }
        for step, (dataset, cost) in enumerate(zip(node.path, node.costs), start=1):
            row[f"step{step}_dataset"] = dataset
            row[f"step{step}_cost"] = cost
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# High-level search API
# --------------------------------------------------------------------------------------


def search_dataset_orders(
    fisher_cache: str | Path,
    output_dir: str | Path | None = None,
    *,
    lams: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    interaction: str = "redundancy",
    decay: float = 1.0,
    aggregate: str = "sum",
    methods: Sequence[str] = ("mcts", "beam", "exact"),
    mcts_simulations: int = 20000,
    mcts_c: float = 1.0,
    mcts_backup: str = "mixed",
    mcts_rollout: str = "eps-greedy",
    mcts_rollout_eps: float = 0.25,
    mcts_restarts: int = 1,
    beam_width: int = 8,
    top_k: int = 20,
    exact_max_datasets: int = 14,
    tabulate_max_datasets: int = 14,
    seed: int = 0,
) -> dict:
    """Search dataset orderings from a Fisher cache.

    Parameters
    ----------
    fisher_cache
        Path to ``fisher_cache.npz`` from :func:`extract_control_fisher`.
    output_dir
        If given, write CSVs/JSON artifacts here. Search still runs if ``None``.
    lams
        Interaction strengths. ``0`` reproduces the static ``log10 Tr(F)`` ranking.
    methods
        Subset of ``{"mcts", "beam", "exact"}``. ``exact`` is skipped when
        ``decay != 1`` or there are more than ``exact_max_datasets`` studies.

    Examples
    --------
    >>> from cscanvi.order_search import search_dataset_orders
    >>> result = search_dataset_orders(
    ...     "results/dataset_order_search/my_run/fisher_cache.npz",
    ...     "results/dataset_order_search/my_run",
    ...     lams=(0.0, 0.5, 1.0, 2.0),
    ...     methods=("mcts", "beam", "exact"),
    ...     mcts_simulations=20000,
    ... )
    >>> result["results"][0]["recommended_order"]
    """
    cache_path = _as_path(fisher_cache)
    out_dir = _as_path(output_dir) if output_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    profiles = FisherProfiles(load_fisher_cache(cache_path))
    gc.collect()

    print(
        f"Loaded {len(profiles.datasets)} datasets | aligned coordinates={profiles.n_aligned:,} | "
        f"truncated params={len(profiles.truncated_params)}",
        flush=True,
    )
    static = pd.DataFrame(
        {
            "dataset": profiles.datasets,
            "log10_trace_F": [np.log10(max(profiles.trace[d], TINY)) for d in profiles.datasets],
        }
    ).sort_values("log10_trace_F")
    if out_dir is not None:
        profiles.similarity_matrix().to_csv(out_dir / "fisher_profile_cosine.csv")
        static.to_csv(out_dir / "static_ranking.csv", index=False)

    requested = list(methods)
    unknown = set(requested) - {"mcts", "beam", "exact"}
    if unknown:
        raise ValueError(f"Unknown search methods: {sorted(unknown)}")
    if not requested:
        raise ValueError("methods must contain at least one of {'mcts', 'beam', 'exact'}.")

    exact_ok = (
        "exact" in requested
        and decay == 1.0
        and len(profiles.datasets) <= exact_max_datasets
    )

    config = {
        "fisher_cache": str(cache_path),
        "lams": list(lams),
        "interaction": interaction,
        "decay": decay,
        "aggregate": aggregate,
        "methods": requested,
        "mcts_simulations": mcts_simulations,
        "mcts_c": mcts_c,
        "mcts_backup": mcts_backup,
        "mcts_rollout": mcts_rollout,
        "mcts_rollout_eps": mcts_rollout_eps,
        "mcts_restarts": mcts_restarts,
        "beam_width": beam_width,
        "top_k": top_k,
        "exact_max_datasets": exact_max_datasets,
        "tabulate_max_datasets": tabulate_max_datasets,
        "seed": seed,
    }

    all_orderings, all_nodes, all_roots, all_convergence, summary = [], [], [], [], []
    for lam in lams:
        cost_model = CostModel(
            profiles,
            lam=float(lam),
            interaction=interaction,
            decay=decay,
            tabulate_max=tabulate_max_datasets,
        )
        mode = "tabulated" if cost_model.table is not None else "lazy"
        print(f"\n=== lam={lam} | cost model {mode} ({cost_model.n_evaluations} evaluations)", flush=True)

        entry: dict = {
            "lam": float(lam),
            "interaction": interaction,
            "decay": decay,
            "aggregate": aggregate,
            "cost_model": mode,
            "static_ranking": static["dataset"].tolist(),
        }
        recommended: Ordering | None = None

        if "mcts" in requested:
            best, roots, convergence = mcts_search(
                cost_model,
                aggregate=aggregate,
                n_simulations=mcts_simulations,
                c_uct=mcts_c,
                backup=mcts_backup,
                rollout=mcts_rollout,
                rollout_eps=mcts_rollout_eps,
                seed=seed,
                restarts=mcts_restarts,
            )
            roots.insert(0, "lam", float(lam))
            convergence.insert(0, "lam", float(lam))
            all_roots.append(roots)
            all_convergence.append(convergence)
            all_orderings.append(orderings_frame([best], float(lam), "mcts"))
            entry["mcts_order"] = list(best.path)
            entry["mcts_objective"] = best.key(aggregate)
            entry["mcts_total_cost"] = best.total
            entry["mcts_simulations"] = mcts_simulations * mcts_restarts
            recommended = best
            print(f"  mcts  : {' > '.join(best.path)}  ({aggregate}={best.key(aggregate):.6f})", flush=True)

        if "beam" in requested:
            beam, tree = beam_search(
                cost_model, aggregate=aggregate, beam_width=beam_width
            )
            tree.insert(0, "lam", float(lam))
            all_nodes.append(tree)
            all_orderings.append(orderings_frame(beam[:top_k], float(lam), "beam"))
            entry["beam_order"] = list(beam[0].path)
            entry["beam_objective"] = beam[0].key(aggregate)
            if recommended is None:
                recommended = beam[0]
            print(f"  beam  : {' > '.join(beam[0].path)}  ({aggregate}={beam[0].key(aggregate):.6f})", flush=True)

        if exact_ok:
            optimum = exact_search(cost_model, aggregate=aggregate)
            all_orderings.append(orderings_frame([optimum], float(lam), "exact_dp"))
            entry["exact_order"] = list(optimum.path)
            entry["exact_objective"] = optimum.key(aggregate)
            print(f"  exact : {' > '.join(optimum.path)}  ({aggregate}={optimum.key(aggregate):.6f})", flush=True)
            for name in ("mcts", "beam"):
                if f"{name}_objective" in entry:
                    gap = entry[f"{name}_objective"] - entry["exact_objective"]
                    entry[f"{name}_gap"] = gap
                    entry[f"{name}_found_optimum"] = bool(gap <= 1e-9)
                    verdict = "matched optimum" if gap <= 1e-9 else f"MISSED by {gap:.4g}"
                    print(f"  {name} vs exact: {verdict}", flush=True)
            recommended = optimum

        assert recommended is not None, "methods selected no usable strategy"
        entry["recommended_order"] = list(recommended.path)
        entry["recommended_source"] = (
            "exact_dp" if exact_ok else ("mcts" if "mcts" in requested else "beam")
        )
        entry["recommended_step_costs"] = [float(c) for c in recommended.costs]
        summary.append(entry)

    result = {"config": config, "results": summary}
    if out_dir is not None:
        pd.concat(all_orderings, ignore_index=True).to_csv(out_dir / "orderings.csv", index=False)
        if all_roots:
            pd.concat(all_roots, ignore_index=True).to_csv(out_dir / "mcts_root_actions.csv", index=False)
        if all_convergence:
            pd.concat(all_convergence, ignore_index=True).to_csv(
                out_dir / "mcts_convergence.csv", index=False
            )
        if all_nodes:
            pd.concat(all_nodes, ignore_index=True).to_csv(out_dir / "beam_tree_nodes.csv", index=False)
        with open(out_dir / "search_summary.json", "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2, default=str)
    return result


def run_dataset_order_search(
    output_dir: str | Path,
    *,
    stage: str = "all",
    query_h5ad: str | Path | None = None,
    reference_model: str | Path | None = None,
    reference_h5ad: str | Path | None = None,
    fisher_cache: str | Path | None = None,
    study_key: str = DEFAULT_STUDY_KEY,
    disease_key: str = DEFAULT_DISEASE_KEY,
    control_values: Sequence[str] = DEFAULT_CONTROL_VALUES,
    required_obs: Sequence[str] = DEFAULT_REQUIRED_OBS,
    exclude_studies: Sequence[str] = (),
    ctrl_cells: int = 2000,
    ctrl_frac: float = 0.2,
    min_ctrl_cells: int = 64,
    replay_cells: int = 512,
    fisher_repeats: int = 3,
    use_gpu: str | int | bool | None = "auto",
    overwrite_fisher: bool = False,
    lams: Sequence[float] = (0.0, 0.5, 1.0, 2.0),
    interaction: str = "redundancy",
    decay: float = 1.0,
    aggregate: str = "sum",
    methods: Sequence[str] = ("mcts", "beam", "exact"),
    mcts_simulations: int = 20000,
    mcts_c: float = 1.0,
    mcts_backup: str = "mixed",
    mcts_rollout: str = "eps-greedy",
    mcts_rollout_eps: float = 0.25,
    mcts_restarts: int = 1,
    beam_width: int = 8,
    top_k: int = 20,
    exact_max_datasets: int = 14,
    tabulate_max_datasets: int = 14,
    seed: int = 0,
) -> dict:
    """Run Fisher extraction and/or MCTS order search.

    ``stage`` is ``"fisher"``, ``"search"``, or ``"all"``. Search-only runs need
    ``fisher_cache`` or an existing ``output_dir/fisher_cache.npz``.

    Examples
    --------
    >>> from cscanvi.order_search import run_dataset_order_search
    >>> result = run_dataset_order_search(
    ...     "results/dataset_order_search/my_run",
    ...     stage="all",
    ...     query_h5ad="adata_ready/ten_crc_query.h5ad",
    ...     reference_model="models/preterm_ord_step_ewc1em2_ep50_R2",
    ...     reference_h5ad="07072026_fetal_oliver2024_panGI_EWC2.0_replay0.2_100epoch_emb_counts.h5ad",
    ...     ctrl_cells=2000,
    ...     lams=(0.0, 0.5, 1.0, 2.0),
    ...     methods=("mcts", "beam", "exact"),
    ...     mcts_simulations=20000,
    ... )
    >>> result["results"][0]["recommended_order"]

    Search-only from an existing cache::

    >>> result = run_dataset_order_search(
    ...     "results/dataset_order_search/my_run",
    ...     stage="search",
    ...     lams=(1.0,),
    ... )
    """
    if stage not in ("fisher", "search", "all"):
        raise ValueError(f"stage must be 'fisher', 'search', or 'all'; got {stage!r}")

    out_dir = _as_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _as_path(fisher_cache) if fisher_cache is not None else out_dir / "fisher_cache.npz"
    result: dict = {}

    if stage in ("fisher", "all"):
        if query_h5ad is None or reference_model is None or reference_h5ad is None:
            raise ValueError(
                "query_h5ad, reference_model, and reference_h5ad are required for Fisher extraction."
            )
        cache_path = extract_control_fisher(
            query_h5ad,
            reference_model,
            reference_h5ad,
            out_dir,
            study_key=study_key,
            disease_key=disease_key,
            control_values=control_values,
            required_obs=required_obs,
            exclude_studies=exclude_studies,
            ctrl_cells=ctrl_cells,
            ctrl_frac=ctrl_frac,
            min_ctrl_cells=min_ctrl_cells,
            replay_cells=replay_cells,
            seed=seed,
            fisher_repeats=fisher_repeats,
            use_gpu=use_gpu,
            overwrite=overwrite_fisher,
        )

    if stage in ("search", "all"):
        if not cache_path.exists():
            raise FileNotFoundError(
                f"No Fisher cache at {cache_path}. Run stage='fisher' first."
            )
        result = search_dataset_orders(
            cache_path,
            out_dir,
            lams=lams,
            interaction=interaction,
            decay=decay,
            aggregate=aggregate,
            methods=methods,
            mcts_simulations=mcts_simulations,
            mcts_c=mcts_c,
            mcts_backup=mcts_backup,
            mcts_rollout=mcts_rollout,
            mcts_rollout_eps=mcts_rollout_eps,
            mcts_restarts=mcts_restarts,
            beam_width=beam_width,
            top_k=top_k,
            exact_max_datasets=exact_max_datasets,
            tabulate_max_datasets=tabulate_max_datasets,
            seed=seed,
        )

    print(f"\nDone. Artifacts in {out_dir}", flush=True)
    return result
