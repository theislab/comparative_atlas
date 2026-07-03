from ._scanvae import SCANVAE

from anndata import AnnData
from typing import Literal

from scvi import REGISTRY_KEYS
from scvi.module.base import LossRecorder, auto_move_data
from scvi.nn import DecoderSCVI
from scvi.distributions import NegativeBinomial
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
import torch
import numpy as np
from typing import Optional, Tuple, Union
from torch.linalg import vector_norm


class LLaVAStyleProjector(torch.nn.Module):
    """Two-layer MLP projector: Linear → GELU → Linear (no batch/layer norm)."""

    def __init__(self, n_in: int, n_hidden: int, n_out: int):
        super().__init__()
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(n_in, n_hidden),
            torch.nn.GELU(),
            torch.nn.Linear(n_hidden, n_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Adapt(SCANVAE):
    """Adaptation module: map scGPT embeddings into the SCANVI latent space.

    The module wraps a pretrained SCANVI backbone (``m0``) and adds:

    * ``projection_layer``: scGPT embedding → latent (LLaVA-style 2-layer GELU MLP)
    * ``decoder``: latent → ``X_target`` (marker-gene counts)
    * ``px_r_m0``: NB dispersion for the adaptation decoder
    * ``alignment_loss_weight``: scalar multiplier on the energy alignment term
      (which is typically smaller in magnitude than NB reconstruction)

    ``Adapt.z_encoder`` (inherited from :class:`SCANVAE`) is **not** used in
    the training objective. Latent alignment always compares
    ``projection_layer(embedding)`` to ``m0.z_encoder(gene counts)``.

    Stage 1 trains only ``projection_layer``, ``decoder``, and ``px_r_m0``
    while ``m0`` is frozen. Stage 2 (via :class:`TTA_SCANVI`) additionally
    trains ``m0`` with SCANVI replay + EWC on full ``adata.X``, while the
    adaptation head reconstructs ``X_target`` and aligns to ``m0`` latents on
    the same cells.
    """

    def __init__(
        self, 
        m0_module: SCANVAE,
        n_input: int = 100,
        n_output: int = 100,
        n_hidden: int = 128,
        n_layers: int = 2,
        dropout_rate: float = 0.1,
        alignment_loss_weight: float = 10.0,
        latent_l2_weight: float = 10.0,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        **model_kwargs,
    ):
        # Inherit mandatory SCANVAE shape/config defaults from trained m0 when
        # not explicitly provided. This prevents invalid defaults like
        # n_labels=0 when constructing from a trained reference model.
        model_kwargs = dict(model_kwargs)
        model_kwargs.setdefault("n_batch", getattr(m0_module, "n_batch", 0))
        model_kwargs.setdefault("n_labels", max(1, getattr(m0_module, "n_labels", 1)))
        model_kwargs.setdefault("n_latent", getattr(m0_module, "n_latent", 10))
        model_kwargs.setdefault("dispersion", getattr(m0_module, "dispersion", "gene"))
        model_kwargs.setdefault(
            "gene_likelihood", getattr(m0_module, "gene_likelihood", "zinb")
        )

        super().__init__(
            n_input=n_input,
            n_hidden=n_hidden,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **model_kwargs,
        )
        self.projection_layer = LLaVAStyleProjector(
            n_in=n_input,
            n_hidden=n_hidden,
            n_out=m0_module.n_latent,
        )
        self.decoder = DecoderSCVI(
            n_input=m0_module.n_latent,
            n_output=n_output,
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # self.px_r_m0 = m0_module.px_r
        self.px_r_m0 = torch.nn.Parameter(torch.randn(n_output))
        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False
        self.m0 = m0_module
        self.use_m0_loss = True
        self.use_embedding_for_inference = True
        self.alignment_loss_weight = float(alignment_loss_weight)
        self.latent_l2_weight = float(latent_l2_weight)
        self.stage1_x_m1 = None
        self.stage1_embedding = None
        # Full-dataset adaptation tensors for stage 2 (all query + replay cells).
        self.stage2_x_m1 = None
        self.stage2_embedding = None

        # EWC state. Empty until `register_ewc_anchor` is called; while empty the
        # penalty computed in `SCANVAE.loss_with_replay` is 0 (the zip is empty).
        self.old_params = []
        self.importances = []
        self.ctrl_importances = []

    def _get_init_params(self, locals):
        return {k: v for k, v in locals.items() if k != "self"}

    def _get_inference_input(self, tensors):
        """Build inference inputs for adaptation.

        In stage-1 adaptation mode (``use_embedding_for_inference=True``) we run
        the Adapt encoder on the embedding tensor instead of the
        SCANVI-registered count matrix ``X``. This decouples the adaptation path
        from SCANVI registry dimensions (e.g. 512-d embedding vs 4917 genes).

        In stage-2 continual mode (``use_embedding_for_inference=False``) the
        batches are gene counts and the Adapt encoder (sized for the embedding
        modality) cannot consume them, so we route the standard ``X`` path
        through the reference/continual backbone ``m0``.
        """
        if not self.use_embedding_for_inference:
            return self.m0._get_inference_input(tensors)
        inputs = super()._get_inference_input(tensors)
        if "embedding" in tensors:
            inputs["x"] = tensors["embedding"]
        return inputs

    def inference(self, *args, **kwargs):
        """Encode inputs.

        In stage-2 continual mode the gene-count ``X`` path is delegated to the
        ``m0`` backbone so the embedding-sized Adapt encoder is bypassed.
        """
        if not self.use_embedding_for_inference:
            return self.m0.inference(*args, **kwargs)
        return super().inference(*args, **kwargs)

    def _get_generative_input(self, tensors, inference_outputs, **kwargs):
        if not self.use_embedding_for_inference:
            return self.m0._get_generative_input(tensors, inference_outputs, **kwargs)
        return super()._get_generative_input(tensors, inference_outputs, **kwargs)

    def generative(self, *args, **kwargs):
        if not self.use_embedding_for_inference:
            return self.m0.generative(*args, **kwargs)
        return super().generative(*args, **kwargs)

    def set_stage2_adaptation_tensors(self, embedding, x_m1):
        """Cache per-cell adaptation tensors aligned with stage-2 ``adata`` rows."""
        self.stage2_embedding = torch.as_tensor(
            np.asarray(embedding), dtype=torch.float32
        )
        self.stage2_x_m1 = torch.as_tensor(np.asarray(x_m1), dtype=torch.float32)

    def _inject_stage2_adapt_tensors(self, tensors):
        """Add ``embedding`` / ``x_m1`` to a gene-count batch via row indices."""
        if not isinstance(tensors, dict):
            tensors = dict(tensors)
        if (
            "embedding" not in tensors
            and self.stage2_embedding is not None
            and REGISTRY_KEYS.INDICES_KEY in tensors
        ):
            batch_indices = tensors[REGISTRY_KEYS.INDICES_KEY].long()
            device = next(self.parameters()).device
            tensors["embedding"] = self.stage2_embedding[batch_indices].to(device)
        if (
            "x_m1" not in tensors
            and self.stage2_x_m1 is not None
            and REGISTRY_KEYS.INDICES_KEY in tensors
        ):
            batch_indices = tensors[REGISTRY_KEYS.INDICES_KEY].long()
            device = next(self.parameters()).device
            tensors["x_m1"] = self.stage2_x_m1[batch_indices].to(device)
        return tensors

    def _adaptation_head_loss(
        self,
        tensors,
        m0_inference_outputs,
        alignment_only=False,
        reconstruction_only=False,
    ):
        """Shared adaptation objective (stages 1–3).

        .. math::

            L = \\mathrm{NB}(X_{target})
            + w_{align} \\cdot \\mathrm{energy}(z_{m0}, z_{proj})
            + w_{l2} \\cdot \\| z_{m0} - z_{proj} \\|^2

        where ``z_proj = projection_layer(embedding)``, ``z_m0`` comes from
        ``m0``'s gene-count encoder (not ``Adapt.z_encoder``).

        * NB reconstruction of ``x_m1`` from scgpt ``embedding`` via
          ``projection_layer`` + ``decoder``
        * Energy score and L2 penalty aligning ``z_proj`` to ``z_m0``

        Stage 1 may set ``reconstruction_only`` or ``alignment_only`` for
        curriculum phases. When ``m0`` is trainable (stage 3), gradients update
        ``m0`` through the alignment terms and the auxiliary head through all
        active terms.
        """
        if "embedding" not in tensors or "x_m1" not in tensors:
            raise KeyError(
                "Adaptation loss expected `embedding` and `x_m1` in tensors. "
                f"Available keys: {list(tensors.keys())}"
            )
        emb = tensors["embedding"]
        x_m1 = tensors["x_m1"]
        z_proj = self.projection_layer(emb)
        library_emb = torch.log(x_m1.sum(dim=1, keepdim=True).clamp_min(1e-8))
        _, _, px_rate_emb, _ = self.decoder("gene", z_proj, library_emb)
        theta = torch.exp(self.px_r_m0.clamp(min=-12, max=12))
        if theta.shape[-1] != px_rate_emb.shape[-1]:
            raise ValueError(
                "Adapt decoder output and px_r_m0 must match X_target gene "
                f"dimension (got px_rate {px_rate_emb.shape[-1]} vs "
                f"px_r_m0 {theta.shape[-1]}). Ensure stage-2 setup does not "
                "overwrite px_r_m0 with m0.px_r."
            )
        reconst_loss_emb = -NegativeBinomial(mu=px_rate_emb, theta=theta).log_prob(
            x_m1
        ).sum(dim=-1)
        m0_z = m0_inference_outputs["z"]
        if not any(p.requires_grad for p in self.m0.parameters()):
            m0_z = m0_z.detach()
        energy_score_loss = self.energy_loss(m0_z, z_proj, beta=2, verbose=False)
        latent_l2_loss = ((z_proj - m0_z) ** 2).mean(dim=-1)
        recon_mean = reconst_loss_emb.mean()
        energy_mean = energy_score_loss.mean()
        l2_mean = latent_l2_loss.mean()
        alignment_mean = (
            self.alignment_loss_weight * energy_mean
            + self.latent_l2_weight * l2_mean
        )
        if reconstruction_only:
            loss = recon_mean
        elif alignment_only:
            loss = alignment_mean
        else:
            loss = recon_mean + alignment_mean
        return LossRecorder(
            loss=loss,
            reconstruction_loss=recon_mean,
            energy_score_loss=energy_mean,
            latent_l2_loss=l2_mean,
            weighted_energy_score_loss=(self.alignment_loss_weight * energy_mean),
        )

    @torch.no_grad()
    def _compute_stage3_log_metrics(
        self, tensors, m0_inference_outputs, loss_kwargs=None
    ):
        """Log-only m0 reconstruction and EWC metrics for stage 3.

        These values are **not** added to the stage-3 optimization objective.
        """
        loss_kwargs = dict(loss_kwargs or {})
        loss_kwargs.setdefault("ewc_importance", 0.0)
        loss_kwargs.setdefault("feed_labels", False)

        m0_generative_inputs = self.m0._get_generative_input(
            tensors, m0_inference_outputs
        )
        m0_generative_outputs = self.m0.generative(**m0_generative_inputs)
        m0_losses = self.m0.loss_with_replay(
            tensors,
            m0_inference_outputs,
            m0_generative_outputs,
            loss_kwargs=loss_kwargs,
        )

        m0_reconstruction_loss = m0_losses.reconstruction_loss
        if m0_reconstruction_loss.ndim > 0:
            m0_reconstruction_loss = m0_reconstruction_loss.mean()

        ewc_loss = getattr(
            m0_losses,
            "ewc_loss",
            torch.zeros((), device=m0_reconstruction_loss.device),
        )
        if isinstance(ewc_loss, torch.Tensor) and ewc_loss.ndim > 0:
            ewc_loss = ewc_loss.mean()

        return {
            "m0_reconstruction_loss": float(m0_reconstruction_loss.detach().cpu()),
            "ewc_loss": float(ewc_loss.detach().cpu()),
        }

    @auto_move_data
    def _replay_forward(
        self,
        tensors,
        get_inference_input_kwargs: Optional[dict] = None,
        get_generative_input_kwargs: Optional[dict] = None,
        inference_kwargs: Optional[dict] = None,
        generative_kwargs: Optional[dict] = None,
        loss_kwargs: Optional[dict] = None,
        compute_loss=True,
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, LossRecorder],
    ]:
        """Forward pass used by training plans.

        In stage-1 adaptation mode we bypass SCANVAE replay-specific routing and
        run a direct inference/generative/loss pass, so the objective depends on
        adaptation tensors (`embedding`, `x_m1`) instead of SCANVI replay `X`.
        """
        if not self.use_embedding_for_inference:
            # Stage-2: same cells, two objectives on one batch.
            #   (1) gene-count SCANVI replay + EWC via ``m0``
            #   (2) scgpt reconstruction + energy alignment (projection z vs m0 z)
            tensors = self._inject_stage2_adapt_tensors(tensors)
            m0_out = self.m0._replay_forward(
                tensors,
                get_inference_input_kwargs=get_inference_input_kwargs,
                get_generative_input_kwargs=get_generative_input_kwargs,
                inference_kwargs=inference_kwargs,
                generative_kwargs=generative_kwargs,
                loss_kwargs=loss_kwargs,
                compute_loss=compute_loss,
            )
            if not compute_loss:
                return m0_out

            m0_inf_out, m0_gen_out, m0_losses = m0_out
            adapt_losses = self._adaptation_head_loss(tensors, m0_inf_out)
            total_loss = m0_losses.loss + adapt_losses.loss
            combined = LossRecorder(
                total_loss,
                m0_losses.reconstruction_loss + adapt_losses.reconstruction_loss,
                m0_losses.kl_local,
                ewc_loss=getattr(m0_losses, "ewc_loss", torch.zeros_like(total_loss)),
                adapt_reconstruction_loss=adapt_losses.reconstruction_loss.detach(),
                energy_score_loss=adapt_losses.energy_score_loss.detach(),
            )
            return m0_inf_out, m0_gen_out, combined

        get_inference_input_kwargs = (
            {} if get_inference_input_kwargs is None else get_inference_input_kwargs
        )
        get_generative_input_kwargs = (
            {} if get_generative_input_kwargs is None else get_generative_input_kwargs
        )
        inference_kwargs = {} if inference_kwargs is None else inference_kwargs
        generative_kwargs = {} if generative_kwargs is None else generative_kwargs
        loss_kwargs = {} if loss_kwargs is None else loss_kwargs

        # In stage-1, batches come from SCANVI manager and may not include
        # adaptation keys. Inject them from cached full tensors via indices.
        if (
            "embedding" not in tensors
            and self.stage1_embedding is not None
            and REGISTRY_KEYS.INDICES_KEY in tensors
        ):
            batch_indices = tensors[REGISTRY_KEYS.INDICES_KEY].long()
            tensors = dict(tensors)
            tensors["embedding"] = self.stage1_embedding[batch_indices].to(
                next(self.parameters()).device
            )
        if (
            "x_m1" not in tensors
            and self.stage1_x_m1 is not None
            and REGISTRY_KEYS.INDICES_KEY in tensors
        ):
            batch_indices = tensors[REGISTRY_KEYS.INDICES_KEY].long()
            if not isinstance(tensors, dict):
                tensors = dict(tensors)
            tensors["x_m1"] = self.stage1_x_m1[batch_indices].to(
                next(self.parameters()).device
            )
        if "embedding" not in tensors and self.use_embedding_for_inference:
            raise KeyError(
                "Stage-1 adaptation expected `embedding` in tensors or "
                f"`{REGISTRY_KEYS.INDICES_KEY}` for cached lookup. "
                f"Available keys: {list(tensors.keys())}"
            )

        inference_inputs = self._get_inference_input(
            tensors, **get_inference_input_kwargs
        )
        inference_outputs = self.inference(**inference_inputs, **inference_kwargs)
        generative_inputs = self._get_generative_input(
            tensors, inference_outputs, **get_generative_input_kwargs
        )
        generative_outputs = self.generative(**generative_inputs, **generative_kwargs)

        if compute_loss:
            losses = self.loss(
                tensors, inference_outputs, generative_outputs, **loss_kwargs
            )
            return inference_outputs, generative_outputs, losses
        return inference_outputs, generative_outputs

    def register_ewc_anchor(self, importances=None, ctrl_importances=None):
        """Snapshot the current trainable params as the EWC anchor.

        After calling this, `SCANVAE.loss_with_replay` regularizes the module's
        trainable parameters toward this snapshot, weighted by the (Fisher)
        importances. This anchors the *adaptation* module to its current state;
        it does not touch or reference the reference ``m0`` weights.

        Parameters
        ----------
        importances
            List of ``(name, tensor)`` importances aligned with the module's
            trainable ``named_parameters()`` (e.g. produced by
            ``ADAPT._compute_importances``). If ``None``, uniform importances
            (ones) are used, i.e. a plain quadratic anchor.
        ctrl_importances
            List of ``(name, tensor)`` control importances. If ``None``, ones
            are used so the ``"product"`` penalty reduces to
            ``importance * (param - anchor) ** 2``.
        """
        self.old_params = [
            (n, p.clone().detach())
            for n, p in self.named_parameters()
            if p.requires_grad
        ]
        if importances is None:
            importances = [(n, torch.ones_like(p)) for n, p in self.old_params]
        if ctrl_importances is None:
            ctrl_importances = [(n, torch.ones_like(p)) for n, p in self.old_params]
        self.importances = importances
        self.ctrl_importances = ctrl_importances
    
    def vectorize(self,x, multichannel=False):
        """Vectorize data in any shape.

        Args:
            x (torch.Tensor): input data
            multichannel (bool, optional): whether to keep the multiple channels (in the second dimension). Defaults to False.

        Returns:
            torch.Tensor: data of shape (sample_size, dimension) or (sample_size, num_channel, dimension) if multichannel is True.
        """
        if len(x.shape) == 1:
            return x.unsqueeze(1)
        if len(x.shape) == 2:
            return x
        else:
            if not multichannel: # one channel
                return x.reshape(x.shape[0], -1)
            else: # multi-channel
                return x.reshape(x.shape[0], x.shape[1], -1)

    def energy_loss(self, x_true, x_est, beta=1, verbose=True):
        """
        Energy score loss, returned per data example (not averaged).

        Args:
            x_true (torch.Tensor): shape [N, D]
            x_est (list of Tensors or a single tensor): 
                - List of M tensors of shape [N, D], or 
                - Tensor of shape [N*M, D] to be split into M samples.
            beta (float): power parameter.
            verbose (bool): if True, also return s1 and s2 terms per example.

        Returns:
            Tensor of shape [N] (if verbose=False), or (loss, s1, s2) if verbose=True.
        """
        if isinstance(beta, torch.Tensor):
            beta_val = beta.item()
        else:
            beta_val = float(beta)
        EPS = 0 if beta_val.is_integer() else 1e-5
        x_true = self.vectorize(x_true).unsqueeze(1)  # shape: [N, 1, D]

        if not isinstance(x_est, list):
            N = x_true.shape[0]
            M = x_est.shape[0] // N
            x_est = list(torch.split(x_est, N, dim=0))
        M = len(x_est)
        x_est = [self.vectorize(xi).unsqueeze(1) for xi in x_est]  # each: [N, 1, D]
        x_est = torch.cat(x_est, dim=1)  # shape: [N, M, D]

        # --- s1: distance from x_true to each sample ---
        s1 = (vector_norm(x_est - x_true, 2, dim=2) + EPS).pow(beta).mean(dim=1)  # shape: [N]

        # --- s2: average pairwise distance among samples per example ---
        # For M <= 1, the pairwise term is undefined (division by zero in
        # unbiased scaling), so we set it to 0.
        if M <= 1:
            s2 = torch.zeros_like(s1)
        else:
            dists = torch.cdist(x_est, x_est, p=2) + EPS  # shape: [N, M, M]
            s2 = dists.pow(beta).mean(dim=(1, 2)) * M / (M - 1)  # shape: [N]

        # --- final loss per example ---
        loss = s1 - s2 / 2  # shape: [N]

        
        if verbose:
            return loss, s1, s2
        else:
            return loss



    def energy_loss_two_sample(self, x0, x, xp, x0p=None, beta=1, verbose=True, weights=None, mask=None):
        """
        Per-example loss function based on the energy score (estimated from two samples).

        Args:
            x0 (torch.Tensor): Sample from the true distribution. Shape: [N, D]
            x (torch.Tensor): Sample from the estimated distribution. Shape: [N, D]
            xp (torch.Tensor): Another sample from the estimated distribution. Shape: [N, D]
            x0p (torch.Tensor, optional): Another sample from the true distribution. Shape: [N, D]
            beta (float): Power parameter in the energy score.
            verbose (bool): Whether to return s1, s2 (and s3 if x0p is given) per example.
            weights (float or torch.Tensor, optional): Scalar or tensor of shape [N] for per-example weights.

        Returns:
            If verbose:
                Tuple of three or four tensors of shape [N]: (loss, s1, s2[, s3])
            Else:
                Tensor of shape [N]: per-example loss
        """
        if isinstance(beta, torch.Tensor):
            beta_val = beta.item()
        else:
            beta_val = float(beta)
        EPS = 0 if beta_val.is_integer() else 1e-5

        x0 = self.vectorize(x0)
        x = self.vectorize(x)
        xp = self.vectorize(xp)

        if weights is None:
            weights = 1.0
        
        weights = torch.tensor(weights, device=x.device, dtype=x.dtype)
        if weights.ndim == 0:
            weights = weights.expand(x.shape[0])
        elif weights.ndim != 1 or weights.shape[0] != x.shape[0]:
            raise ValueError(f"Weights must be a scalar or a tensor of shape [{x.shape[0]}]")

        if x0p is None:
            # s1 terms
            s1_term1 = (vector_norm(x - x0, 2, dim=1) + EPS).pow(beta) / 2
            s1_term2 = (vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta) / 2
            s1 = s1_term1 + s1_term2

            # s2 term
            s2 = (vector_norm(x - xp, 2, dim=1) + EPS).pow(beta) / 2

            loss = (s1 - s2) * weights

            if mask is not None:
                if not torch.is_floating_point(mask) and mask.dtype != torch.bool:
                    mask = mask.bool()
                loss = loss[mask]
                s1 = s1[mask]
                s2 = s2[mask]
                weights = weights[mask]
            if verbose:
                return loss, s1 * weights, s2 * weights
            else:
                return loss

        else:
            x0p = self.vectorize(x0p)

            # s1 terms
            s1_term1 = (vector_norm(x - x0, 2, dim=1) + EPS).pow(beta) / 4
            s1_term2 = (vector_norm(xp - x0, 2, dim=1) + EPS).pow(beta) / 4
            s1_term3 = (vector_norm(x - x0p, 2, dim=1) + EPS).pow(beta) / 4
            s1_term4 = (vector_norm(xp - x0p, 2, dim=1) + EPS).pow(beta) / 4
            s1 = s1_term1 + s1_term2 + s1_term3 + s1_term4

            # s2 and s3 terms
            s2 = (vector_norm(x - xp, 2, dim=1) + EPS).pow(beta) / 2
            s3 = (vector_norm(x0 - x0p, 2, dim=1) + EPS).pow(beta) / 2

            loss = (s1 - s2 - s3) * weights

            if mask is not None:
                if not torch.is_floating_point(mask) and mask.dtype != torch.bool:
                    mask = mask.bool()
                loss = loss[mask]
                s1 = s1[mask]
                s2 = s2[mask]
                weights = weights[mask]
            if verbose:
                return loss, s1 * weights, s2 * weights, s3 * weights
            else:
                return loss


    
    def loss(self, tensors, inference_outputs, generative_outputs, **loss_kwargs):
        # ``loss_kwargs`` (feed_labels, labelled_tensors, classification_ratio,
        # kl_weight, ewc_importance, ...) are accepted for compatibility with
        # the ``loss_with_replay`` call path but are not used by this objective.
        #
        # Latent alignment uses ``m0``'s gene-count encoder (``m0.z``), not
        # ``Adapt.z_encoder``. ``inference_outputs`` from Adapt is ignored.

        m0_tensors = dict(tensors)
        m0_inference_inputs = self.m0._get_inference_input(m0_tensors)
        with torch.set_grad_enabled(any(p.requires_grad for p in self.m0.parameters())):
            m0_inference_outputs = self.m0.inference(**m0_inference_inputs)

        adapt_losses = self._adaptation_head_loss(tensors, m0_inference_outputs)

        if not self.use_m0_loss:
            return adapt_losses

        m0_generative_inputs = self.m0._get_generative_input(
            m0_tensors, m0_inference_outputs
        )
        m0_generative_outputs = self.m0.generative(**m0_generative_inputs)
        m0_loss_kwargs = {}
        for key in (
            "feed_labels",
            "kl_weight",
            "labelled_tensors",
            "classification_ratio",
            "replay",
        ):
            if key in loss_kwargs:
                m0_loss_kwargs[key] = loss_kwargs[key]
        loss_m0 = self.m0.loss(
            m0_tensors,
            m0_inference_outputs,
            m0_generative_outputs,
            **m0_loss_kwargs,
        )
        total = adapt_losses.loss + loss_m0.loss
        return LossRecorder(
            loss=total,
            reconstruction_loss=adapt_losses.reconstruction_loss + loss_m0.reconstruction_loss,
            energy_score_loss=adapt_losses.energy_score_loss,
        )