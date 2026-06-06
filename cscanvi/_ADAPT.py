import logging
import warnings
from copy import deepcopy
from typing import List, Optional, Sequence, Union, Iterable, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import scipy.sparse as sp
from ._scanvi import SCANVI
import torch
from anndata import AnnData

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
# from scvi.data import AnnDataManager
from cscanvi.data._manager import AnnDataManager
from scvi.data._constants import _SETUP_ARGS_KEY
from scvi.data._utils import get_anndata_attribute
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LabelsWithUnlabeledObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ObsmField,
)



from scvi.dataloaders._ann_dataloader import AnnDataLoader
from torch.utils.data.dataloader import default_collate


from scvi.utils import setup_anndata_dsp
from scvi.module.base import auto_move_data


from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
from scvi.data import _constants
from ._adapt import Adapt



# from scvi.model._scvi import SCVI
# from scvi.model.base import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin

logger = logging.getLogger(__name__)


class UnfreezeM0Callback(pl.Callback):
    """Two-stage training schedule for the :class:`Adapt` module.

    The pretrained ``m0`` submodule is expected to be frozen before training
    starts (so the optimizer only contains ``projection_layer`` and decoder
    params). At ``unfreeze_at_epoch`` this callback re-enables grads on ``m0``
    and registers its parameters as a new optimizer param group, so from then
    on both stages are trained jointly.
    """

    def __init__(
        self,
        unfreeze_at_epoch: int,
        m0_lr: Optional[float] = None,
        stage2_ewc_importance: Optional[float] = None,
    ):
        super().__init__()
        self.unfreeze_at_epoch = unfreeze_at_epoch
        self.m0_lr = m0_lr
        self.stage2_ewc_importance = stage2_ewc_importance
        self._unfrozen = False

    def on_train_epoch_start(self, trainer, pl_module):
        if self._unfrozen or self.unfreeze_at_epoch <= 0:
            return
        if trainer.current_epoch < self.unfreeze_at_epoch:
            return

        m0 = pl_module.module.m0
        new_params = []
        for p in m0.parameters():
            if not p.requires_grad:
                p.requires_grad_(True)
                new_params.append(p)

        if new_params:
            optimizer = trainer.optimizers[0]
            base_group = optimizer.param_groups[0]
            group = {k: v for k, v in base_group.items() if k != "params"}
            group["params"] = new_params
            if self.m0_lr is not None:
                group["lr"] = self.m0_lr
            optimizer.add_param_group(group)
            logger.info(
                "Unfroze m0 at epoch %d and added %d parameter tensors to the "
                "optimizer (lr=%s).",
                trainer.current_epoch,
                len(new_params),
                group["lr"],
            )

        if self.stage2_ewc_importance is not None and hasattr(pl_module, "loss_kwargs"):
            pl_module.loss_kwargs["ewc_importance"] = self.stage2_ewc_importance
            logger.info(
                "Activated stage-2 EWC at epoch %d with ewc_importance=%s.",
                trainer.current_epoch,
                self.stage2_ewc_importance,
            )

        self._unfrozen = True


class TTA_SCANVI(SCANVI):
    """Adapt a pretrained :class:`SCANVI` model (``m0``) to new data.

    ``TTA_SCANVI`` wraps a pretrained reference model and learns to map a
    precomputed scGPT embedding (``embedding_key``) into the reference latent
    space, while reconstructing a marker-gene count panel (``x_adapt_key``).

    Two decoders are involved:

    * ``Adapt.decoder`` reconstructs ``x_adapt_key`` from
      ``projection_layer(embedding)``.
    * ``m0.decoder`` reconstructs full ``adata.X`` (used in stage 2 replay loss).

    The workflow is explicitly split into two function calls:

    1. :meth:`train_stage1_adaptation` — frozen ``m0`` encodes gene counts to
       ``z``; ``projection_layer`` maps embeddings to ``z_proj``; loss =
       NB reconstruction of ``x_adapt_key`` + energy score between ``z`` and
       ``z_proj``. Only the adaptation head is trained.
    2. :meth:`load_query_data_with_replay_stage2` — attaches stage-1 weights,
       sets up SCANVI replay + EWC on ``m0``, and trains jointly: replay/EWC
       on ``adata.X`` plus the same adaptation loss on **every** cell (query and
       replay), using the same minibatch rows.

    Parameters
    ----------
    adata
        AnnData used to initialize this model directly. Prefer
        :meth:`from_trained_scanvi` to avoid manual setup-key plumbing.
    m0_model
        A trained reference :class:`SCANVI` model to adapt from.
    adapt_kwargs
        Keyword args forwarded to the :class:`Adapt` module, e.g. ``n_input``
        (scGPT embedding dimension), ``n_output`` (number of genes in
        ``x_adapt_key``), ``n_latent`` (should match ``m0``'s latent dim),
        ``n_hidden``, ``n_layers``, ``dropout_rate``.
    **kwargs
        Keyword args forwarded to :class:`SCANVI`.

    Examples
    --------
    >>> # ref_model: trained SCANVI; ad_old: reference cells with obsm keys
    >>> n_in = ad_old.obsm["X_scgpt"].shape[1]
    >>> n_out = ad_old.obsm["X_target"].shape[1]
    >>> model = TTA_SCANVI.from_trained_scanvi(
    ...     ref_model,
    ...     adapt_kwargs=dict(
    ...         n_input=n_in,
    ...         n_output=n_out,
    ...         n_latent=ref_model.module.n_latent,
    ...     ),
    ... )
    >>> # Stage 1: frozen m0 encodes adata.X; train projection + decoder only
    >>> model.train_stage1_adaptation(
    ...     adata=ad_old,
    ...     embedding_key="X_scgpt",
    ...     x_adapt_key="X_target",
    ...     max_epochs=50,
    ...     batch_size=128,
    ... )
    >>> # Stage 2: all cells need obsm["X_scgpt"] and obsm["X_target"]
    >>> adata_query = adata_query.concatenate(adata_replay)
    >>> adata_query.uns["ctrl_query"] = ctrl_adata
    >>> adata_query.uns["replay_adata"] = adata_replay
    >>> model_stage2 = TTA_SCANVI.load_query_data_with_replay_stage2(
    ...     adata=adata_query,
    ...     reference_model=ref_model,
    ...     control_uns_key="ctrl_query",
    ...     replay_uns_key="replay_adata",
    ...     adapt_reference_model=model,
    ...     embedding_key="X_scgpt",
    ...     x_adapt_key="X_target",
    ... )
    >>> model_stage2.train(
    ...     max_epochs=200,
    ...     batch_size=128,
    ...     plan_kwargs=dict(ewc_importance=1.2),
    ... )
    >>> adata_query.obsm["X_TTA_SCANVI"] = model_stage2.get_latent_representation()
    """

    def __init__(
        self,
        adata: AnnData,
        m0_model: SCANVI,
        adapt_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Initialize from explicit adata + scanvi model.

        Note
        ----
        For test-time adaptation of an already trained continual model,
        prefer :meth:`from_trained_scanvi`, which reuses the trained model's
        registry/keys directly.
        """
        super().__init__(adata, **kwargs)
        self.m0_model = m0_model

        adapt_kwargs = adapt_kwargs or {}
        self.module = Adapt(m0_module=self.m0_model.module, **adapt_kwargs)

        self.was_pretrained = False

    @classmethod
    def from_trained_scanvi(
        cls,
        reference_model: SCANVI,
        adapt_kwargs: Optional[dict] = None,
    ):
        """Create TTA model directly from a trained SCANVI model.

        This avoids passing/setup of ``labels_key``, ``unlabeled_category``,
        ``batch_key`` and other setup arguments: all tensor registration metadata
        is inherited from the trained reference model.
        """
        reference_model._check_if_trained(warn=False)
        model = deepcopy(reference_model)
        model.__class__ = cls
        model.m0_model = reference_model
        adapt_kwargs = adapt_kwargs or {}
        model.module = Adapt(m0_module=deepcopy(reference_model.module), **adapt_kwargs)
        model.was_pretrained = True
        return model

    def train_stage1_adaptation(
        self,
        adata: AnnData,
        embedding_key: str,
        x_adapt_key: str,
        max_epochs: Optional[int] = None,
        batch_size: int = 128,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        use_gpu: Optional[Union[str, int, bool]] = None,
        check_val_every_n_epoch: Optional[int] = None,
        plan_kwargs: Optional[dict] = None,
        **kwargs,
    ):
        """Stage 1: train adaptation only (projection + decoder).

        This stage runs a direct minibatch torch loop (without Lightning
        TrainRunner). For each cell:

        * gene counts ``X`` are encoded by **frozen** ``m0.z_encoder`` → ``z``
        * scgpt embeddings are mapped by ``projection_layer`` → ``z_emb``
        * loss = NB reconstruction of ``x_adapt_key`` from ``z_emb`` + energy
          score between ``z`` and ``z_emb``

        ``Adapt.z_encoder`` is not used. Only ``projection_layer``,
        ``decoder``, and ``px_r_m0`` are trained; ``m0`` stays frozen.

        Parameters
        ----------
        adata
            AnnData with SCANVI-registered ``adata.X`` (gene counts) and
            ``adata.obsm[embedding_key]``, ``adata.obsm[x_adapt_key]``.
        embedding_key
            ``obsm`` key for scGPT embeddings (encoder input to
            ``projection_layer``).
        x_adapt_key
            ``obsm`` key for marker-gene counts reconstructed by
            ``Adapt.decoder``.
        max_epochs
            Number of training epochs (default 50).
        batch_size
            Minibatch size for the manual training loop (default 128).
        train_size
            Fraction of cells used for training (default 0.9).
        plan_kwargs
            Optional dict with ``lr`` and ``weight_decay`` for Adam.

        Examples
        --------
        >>> model.train_stage1_adaptation(
        ...     adata=ad_old,
        ...     embedding_key="X_scgpt",
        ...     x_adapt_key="X_target",
        ...     max_epochs=50,
        ...     batch_size=128,
        ... )
        """
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)

        # Device selection consistent with `use_gpu` intent.
        if use_gpu is False:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Prepare adaptation tensors.
        embedding_data = adata.obsm[embedding_key]
        if sp.issparse(embedding_data):
            embedding_data = embedding_data.toarray()
        x_m1_data = adata.obsm[x_adapt_key]
        if sp.issparse(x_m1_data):
            x_m1_data = x_m1_data.toarray()
        x_m1_data = np.asarray(x_m1_data, dtype=np.float32)
        if not np.isfinite(x_m1_data).all():
            raise ValueError("`x_adapt_key` contains non-finite values (NaN/Inf).")
        if (x_m1_data < 0).any():
            raise ValueError(
                "`x_adapt_key` contains negative values; NB reconstruction expects non-negative counts/intensities."
            )

        # Gene counts for the frozen m0 encoder (SCANVI-registered X).
        x_m0_data = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)

        # Optional covariates/labels from SCANVI registry.
        batch_data = adata_manager.get_from_registry(REGISTRY_KEYS.BATCH_KEY)
        labels_data = (
            adata_manager.get_from_registry(REGISTRY_KEYS.LABELS_KEY)
            if REGISTRY_KEYS.LABELS_KEY in adata_manager.data_registry
            else None
        )
        cont_covs_data = (
            adata_manager.get_from_registry(REGISTRY_KEYS.CONT_COVS_KEY)
            if REGISTRY_KEYS.CONT_COVS_KEY in adata_manager.data_registry
            else None
        )
        cat_covs_data = (
            adata_manager.get_from_registry(REGISTRY_KEYS.CAT_COVS_KEY)
            if REGISTRY_KEYS.CAT_COVS_KEY in adata_manager.data_registry
            else None
        )

        def _slice_rows(data, idx):
            if data is None:
                return None
            if isinstance(data, pd.DataFrame):
                return data.iloc[idx, :].to_numpy()
            if isinstance(data, pd.Series):
                return data.iloc[idx].to_numpy()
            return data[idx]

        # Stage-1 settings.
        for p in self.module.m0.parameters():
            p.requires_grad_(False)
        # Adapt inherits a VAE z_encoder from SCANVAE but it is not part of the
        # adaptation design; keep it frozen so only projection/decoder train.
        for name, p in self.module.named_parameters():
            if name.startswith("z_encoder.") or name.startswith("l_encoder."):
                p.requires_grad_(False)
        self.module.use_m0_loss = False
        self.module.use_embedding_for_inference = True
        self.module.to(device)
        self.module.train()

        # Keep optimizer configurable via plan_kwargs to match existing API style.
        plan_kwargs = dict(plan_kwargs or {})
        lr = float(plan_kwargs.get("lr", 1e-3))
        weight_decay = float(plan_kwargs.get("weight_decay", 1e-6))
        params = [p for p in self.module.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay, eps=0.01)

        n_obs = adata.n_obs
        if max_epochs is None:
            max_epochs = 50
        train_n = int(np.floor(train_size * n_obs))
        all_idx = np.arange(n_obs)
        np.random.shuffle(all_idx)
        train_idx = all_idx[:train_n]
        if len(train_idx) == 0:
            raise ValueError("Stage-1 train split is empty; increase `train_size`.")

        epoch_losses = []
        for _ in range(max_epochs):
            perm = np.random.permutation(train_idx)
            running = 0.0
            n_batches = 0
            for i in range(0, len(perm), batch_size):
                idx = perm[i : i + batch_size]
                x_slice = _slice_rows(x_m0_data, idx)
                if sp.issparse(x_slice):
                    x_slice = x_slice.toarray()
                x = torch.as_tensor(x_slice, dtype=torch.float32, device=device)
                emb = torch.as_tensor(embedding_data[idx], dtype=torch.float32, device=device)
                x_m1 = torch.as_tensor(x_m1_data[idx], dtype=torch.float32, device=device)
                batch = torch.as_tensor(
                    _slice_rows(batch_data, idx), dtype=torch.int64, device=device
                )

                tensors = {
                    REGISTRY_KEYS.X_KEY: x,
                    REGISTRY_KEYS.BATCH_KEY: batch,
                    "embedding": emb,
                    "x_m1": x_m1,
                }
                if labels_data is not None:
                    tensors[REGISTRY_KEYS.LABELS_KEY] = torch.as_tensor(
                        _slice_rows(labels_data, idx), dtype=torch.int64, device=device
                    )
                if cont_covs_data is not None:
                    tensors[REGISTRY_KEYS.CONT_COVS_KEY] = torch.as_tensor(
                        _slice_rows(cont_covs_data, idx),
                        dtype=torch.float32,
                        device=device,
                    )
                if cat_covs_data is not None:
                    tensors[REGISTRY_KEYS.CAT_COVS_KEY] = torch.as_tensor(
                        _slice_rows(cat_covs_data, idx),
                        dtype=torch.int64,
                        device=device,
                    )

                optimizer.zero_grad()
                with torch.no_grad():
                    m0_inference_inputs = self.module.m0._get_inference_input(tensors)
                    m0_inference_outputs = self.module.m0.inference(
                        **m0_inference_inputs
                    )
                losses = self.module._adaptation_head_loss(tensors, m0_inference_outputs)
                loss = losses.loss if losses.loss.ndim == 0 else losses.loss.mean()
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite stage1 loss encountered. "
                        f"batch_start={i}, "
                        f"emb_finite={torch.isfinite(emb).all().item()}, "
                        f"x_m1_finite={torch.isfinite(x_m1).all().item()}, "
                        f"emb_absmax={float(emb.abs().max().detach().cpu())}, "
                        f"x_m1_absmax={float(x_m1.abs().max().detach().cpu())}"
                    )
                loss.backward()
                optimizer.step()
                running += float(loss.detach().cpu())
                n_batches += 1

            epoch_losses.append(running / max(n_batches, 1))

        self.module.eval()
        self.is_trained_ = True
        self.history_ = {"stage1_train_loss": epoch_losses}
        return self.history_

    @classmethod
    def load_query_data_with_replay_stage2(
        cls,
        adata: AnnData,
        reference_model: Union[str, SCANVI],
        control_uns_key: str = None,
        replay_uns_key: str = None,
        adapt_reference_model: Optional["TTA_SCANVI"] = None,
        embedding_key: str = "X_scgpt",
        x_adapt_key: str = "X_target",
        **kwargs,
    ):
        """Stage 2: SCANVI replay loading + adaptation module reattachment.

        This follows :meth:`SCANVI.load_query_data_with_replay` for registry and
        continual-learning setup, then attaches the adaptation module/weights.

        The stage-2 objective on **every cell** (query and replay) is:

        * NB reconstruction of ``x_adapt_key`` from the scgpt embedding via the
          adaptation head, plus an energy score aligning ``projection_layer(z)``
          to ``m0``'s gene-count latent ``z``, and
        * the standard SCANVI replay loss plus EWC on gene counts via ``m0``.

        Both terms are evaluated on the same minibatch rows, so the model learns
        to align the embedding and gene-count representations jointly.

        Parameters
        ----------
        adata
            Query (+ replay concatenated) AnnData with SCANVI-registered
            ``adata.X`` and ``obsm[embedding_key]`` / ``obsm[x_adapt_key]`` on
            **every** row.
        reference_model
            Trained reference :class:`SCANVI` (path or model object).
        control_uns_key, replay_uns_key
            ``adata.uns`` keys for control and replay subsets (see
            :meth:`SCANVI.load_query_data_with_replay`).
        adapt_reference_model
            Stage-1 :class:`TTA_SCANVI` whose adaptation weights are copied
            and attached. Pass the object returned from stage 1.
        embedding_key, x_adapt_key
            ``obsm`` keys for scGPT embeddings and marker-gene targets.

        Examples
        --------
        >>> adata_query.obsm["X_scgpt"] = emb_pilot.X
        >>> adata_query.obsm["X_target"] = target_counts
        >>> adata_replay.obsm["X_scgpt"] = emb_atlas[adata_replay.obs_names].X
        >>> adata_replay.obsm["X_target"] = target_counts_replay
        >>> adata_train = adata_query.concatenate(adata_replay)
        >>> adata_train.uns["ctrl_query"] = ctrl_adata
        >>> adata_train.uns["replay_adata"] = adata_replay
        >>> model_stage2 = TTA_SCANVI.load_query_data_with_replay_stage2(
        ...     adata=adata_train,
        ...     reference_model=ref_model,
        ...     control_uns_key="ctrl_query",
        ...     replay_uns_key="replay_adata",
        ...     adapt_reference_model=model,
        ...     embedding_key="X_scgpt",
        ...     x_adapt_key="X_target",
        ... )
        >>> model_stage2.train(
        ...     max_epochs=200,
        ...     batch_size=128,
        ...     plan_kwargs=dict(ewc_importance=1.2),
        ... )
        """
        model = SCANVI.load_query_data_with_replay(
            adata=adata,
            reference_model=reference_model,
            control_uns_key=control_uns_key,
            replay_uns_key=replay_uns_key,
            **kwargs,
        )
        if adapt_reference_model is None and isinstance(reference_model, TTA_SCANVI):
            adapt_reference_model = reference_model
        if adapt_reference_model is not None:
            # Start from a deep copy so stage-2 model does not share module
            # references with stage-1 model.
            adapted_module = deepcopy(adapt_reference_model.module)
            stage2_base_module = model.module

            # Refresh SCANVI/SCANVAE backbone params from replay-loaded stage-2
            # model while keeping adaptation-specific heads/states from stage 1.
            adapted_state = adapted_module.state_dict()
            for key, value in stage2_base_module.state_dict().items():
                if key in adapted_state and adapted_state[key].shape == value.shape:
                    adapted_state[key] = value
            adapted_module.load_state_dict(adapted_state, strict=False)

            # Rewire reference module pointers to stage-2 continual model.
            adapted_module.m0 = stage2_base_module
            # Keep stage-1 ``px_r_m0`` (size = n_output / X_target genes). Do not
            # replace with ``m0.px_r`` (full adata.X gene dimension).
            adapted_module.use_m0_loss = False
            adapted_module.use_embedding_for_inference = False

            registered_adata = model.adata
            if embedding_key not in registered_adata.obsm:
                raise KeyError(
                    f"Stage-2 requires obsm['{embedding_key}'] on all training cells."
                )
            if x_adapt_key not in registered_adata.obsm:
                raise KeyError(
                    f"Stage-2 requires obsm['{x_adapt_key}'] on all training cells."
                )

            # Register adaptation obsm fields so every gene-count minibatch also
            # carries the scgpt embedding and reconstruction target tensors.
            adata_manager = model.get_anndata_manager(registered_adata, required=True)
            for field in (
                ObsmField("embedding", embedding_key, is_count_data=False),
                ObsmField("x_m1", x_adapt_key, is_count_data=True),
            ):
                if field.registry_key not in adata_manager.data_registry:
                    adata_manager._add_field(field, registered_adata)

            emb_data = registered_adata.obsm[embedding_key]
            if sp.issparse(emb_data):
                emb_data = emb_data.toarray()
            x_m1_data = registered_adata.obsm[x_adapt_key]
            if sp.issparse(x_m1_data):
                x_m1_data = x_m1_data.toarray()
            adapted_module.set_stage2_adaptation_tensors(
                embedding=np.asarray(emb_data, dtype=np.float32),
                x_m1=np.asarray(x_m1_data, dtype=np.float32),
            )

            model.module = adapted_module
        return model

    def register_ewc_anchor(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: int = 256,
        uniform: bool = False,
    ):
        """Set the EWC anchor for the adaptation module.

        Snapshots the module's current trainable parameters as the EWC anchor
        and computes (Fisher) importances so that subsequent training with
        ``plan_kwargs={"ewc_importance": > 0}`` penalizes drift away from this
        state. This regularizes the *adaptation* module against itself; it does
        not reference or modify the reference ``m0`` weights.

        Call this after constructing the model (and, if desired, after stage 1)
        and before the EWC-regularized training run.

        Parameters
        ----------
        adata
            AnnData to estimate importances on. Defaults to the model's adata.
        indices
            Optional subset of observations to use.
        batch_size
            Minibatch size for the importance estimation loader.
        uniform
            If ``True``, skip Fisher estimation and use uniform (ones)
            importances, i.e. a plain quadratic anchor.
        """
        if uniform:
            self.module.register_ewc_anchor()
            return

        adata = self._validate_anndata(adata)
        dataloader = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        importances = self._compute_importances(model=self, dataloader=dataloader)
        self.module.register_ewc_anchor(importances=importances)
