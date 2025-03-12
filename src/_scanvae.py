from typing import Iterable, Optional, Sequence, Union, Tuple

import numpy as np
import torch
from torch.distributions import Categorical, Normal
from torch.distributions import kl_divergence as kl
from torch.nn import functional as F

from copy import deepcopy

from scvi import REGISTRY_KEYS
from scvi._compat import Literal
from scvi.module.base import LossRecorder, auto_move_data
from scvi.nn import Decoder, Encoder

from scvi.module._classifier import Classifier
from scvi.module._utils import broadcast_labels
# from scvi.module._vae import VAE
from ._vae import VAE_GR

from typing import NamedTuple

class EXTRA_KEYS(NamedTuple):
    REPLAY_Z_KEY: str =  'replay_z_key'
    REPLAY_X_KEY: str =  'replay_x_key'
    REPLAY_BATCH_KEY: str = 'replay_batch_key'
    REPLAY_LABELS_KEY: str = 'replay_labels_key'


EXTRA_KEYS = EXTRA_KEYS()


class SCANVAE(VAE_GR):
    """
    Single-cell annotation using variational inference.
    This is an implementation of the scANVI model described in [Xu21]_,
    inspired from M1 + M2 model, as described in (https://arxiv.org/pdf/1406.5298.pdf).
    Parameters
    ----------
    n_input
        Number of input genes
    n_batch
        Number of batches
    n_labels
        Number of labels
    n_hidden
        Number of nodes per hidden layer
    n_latent
        Dimensionality of the latent space
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    dispersion
        One of the following
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
    y_prior
        If None, initialized to uniform probability over cell types
    labels_groups
        Label group designations
    use_labels_groups
        Whether to use the label groups
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    **vae_kwargs
        Keyword args for :class:`~scvi.module.VAE`
    """

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        dispersion: str = "gene",
        log_variational: bool = True,
        gene_likelihood: str = "zinb",
        y_prior=None,
        labels_groups: Sequence[int] = None,
        use_labels_groups: bool = False,
        classifier_parameters: dict = dict(),
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        # n_control: int = None,
        **vae_kwargs
    ):
        super().__init__(
            n_input,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            n_continuous_cov=n_continuous_cov,
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            n_batch=n_batch,
            dispersion=dispersion,
            log_variational=log_variational,
            gene_likelihood=gene_likelihood,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **vae_kwargs
        )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # hard-code for now
        self.combine_type = "product"
        
        self.n_labels = n_labels
        # Classifier takes n_latent as input
        cls_parameters = {
            "n_layers": n_layers,
            "n_hidden": n_hidden,
            "dropout_rate": dropout_rate,
        }
        cls_parameters.update(classifier_parameters)
        self.classifier = Classifier(
            n_latent,
            n_labels=n_labels,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            **cls_parameters
        )
         
        # self.control_size = n_control
        self.encoder_z2_z1 = Encoder(
            n_latent,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
        )
        self.decoder_z1_z2 = Decoder(
            n_latent,
            n_latent,
            n_cat_list=[self.n_labels],
            n_layers=n_layers,
            n_hidden=n_hidden,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
        )

        self.y_prior = torch.nn.Parameter(
            y_prior
            if y_prior is not None
            else (1 / n_labels) * torch.ones(1, n_labels),
            requires_grad=False,
        )
        self.use_labels_groups = use_labels_groups
        self.labels_groups = (
            np.array(labels_groups) if labels_groups is not None else None
        )
        if self.use_labels_groups:
            if labels_groups is None:
                raise ValueError("Specify label groups")
            unique_groups = np.unique(self.labels_groups)
            self.n_groups = len(unique_groups)
            if not (unique_groups == np.arange(self.n_groups)).all():
                raise ValueError()
            self.classifier_groups = Classifier(
                n_latent, n_hidden, self.n_groups, n_layers, dropout_rate
            )
            self.groups_index = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.tensor(
                            (self.labels_groups == i).astype(np.uint8),
                            dtype=torch.uint8,
                        ),
                        requires_grad=False,
                    )
                    for i in range(self.n_groups)
                ]
            )

    @auto_move_data
    def classify(self, x, batch_index=None, cont_covs=None, cat_covs=None):
        if self.log_variational:
            x = torch.log(1 + x)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x, cont_covs), dim=-1)
        else:
            encoder_input = x
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = tuple()

        qz_m, _, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        # We classify using the inferred mean parameter of z_1 in the latent space
        z = qz_m
        if self.use_labels_groups:
            w_g = self.classifier_groups(z)
            unw_y = self.classifier(z)
            w_y = torch.zeros_like(unw_y)
            for i, group_index in enumerate(self.groups_index):
                unw_y_g = unw_y[:, group_index]
                w_y[:, group_index] = unw_y_g / (
                    unw_y_g.sum(dim=-1, keepdim=True) + 1e-8
                )
                w_y[:, group_index] *= w_g[:, [i]]
        else:
            w_y = self.classifier(z)
        return w_y

    @auto_move_data
    def classification_loss(self, labelled_dataset):
        x = labelled_dataset[REGISTRY_KEYS.X_KEY]
        y = labelled_dataset[REGISTRY_KEYS.LABELS_KEY]
        batch_idx = labelled_dataset[REGISTRY_KEYS.BATCH_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = (
            labelled_dataset[cont_key] if cont_key in labelled_dataset.keys() else None
        )

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = (
            labelled_dataset[cat_key] if cat_key in labelled_dataset.keys() else None
        )
        classification_loss = F.cross_entropy(
            self.classify(
                x, batch_index=batch_idx, cat_covs=cat_covs, cont_covs=cont_covs
            ),
            y.view(-1).long(),
        )
        return classification_loss

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        feed_labels=False,
        replay_importance=None, # placeholder, not used
        ewc_importance=None, # placeholder, not used
        l2_ewc=None, # placeholder, not used
        kl_weight=1,
        labelled_tensors=None,
        classification_ratio=None,
        replay=False,
    ):
        
        if not replay:
            px_r = generative_outputs["px_r"]
            px_rate = generative_outputs["px_rate"]
            px_dropout = generative_outputs["px_dropout"]
            qz1_m = inference_outputs["qz_m"]
            qz1_v = inference_outputs["qz_v"]
            z1 = inference_outputs["z"]
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
            if feed_labels:
                y = tensors[REGISTRY_KEYS.LABELS_KEY]
            else:
                y = None
            is_labelled = False if y is None else True
        else:
            px_r = generative_outputs["px_r_replay"]
            px_rate = generative_outputs["px_rate_replay"]
            px_dropout = generative_outputs["px_dropout_replay"]
            qz1_m = inference_outputs["qz_m_replay"]
            qz1_v = inference_outputs["qz_v_replay"]
            z1 = inference_outputs["z_replay"]
            x = tensors[EXTRA_KEYS.REPLAY_X_KEY]
            batch_index = tensors[EXTRA_KEYS.REPLAY_BATCH_KEY]
            if feed_labels:
                y = tensors[EXTRA_KEYS.REPLAY_LABELS_KEY]
            else:
                y = None
            is_labelled = False if y is None else True
            

        # Enumerate choices of label
        ys, z1s = broadcast_labels(y, z1, n_broadcast=self.n_labels)
        qz2_m, qz2_v, z2 = self.encoder_z2_z1(z1s, ys)
        pz1_m, pz1_v = self.decoder_z1_z2(z2, ys)
        
        
        reconst_loss = self.get_reconstruction_loss(x, px_rate, px_r, px_dropout)


        # # replay stats
        # if EXTRA_KEYS.REPLAY_X_KEY in tensors.keys():
        #     x_replay = tensors[EXTRA_KEYS.REPLAY_X_KEY]
        #     px_rate_replay = generative_outputs["px_rate_replay"]
        #     px_r_replay = generative_outputs["px_r_replay"]
        #     px_dropout_replay = generative_outputs["px_dropout_replay"]
        #     reconst_loss_replay = self.get_reconstruction_loss(x_replay, px_rate_replay, px_r_replay, px_dropout_replay)

        # else:
        #     x_replay = px_rate_replay = px_r_replay = px_dropout_replay = None
        #     reconst_loss_replay = torch.tensor(0.0)
        



        # KL Divergence
        mean = torch.zeros_like(qz2_m)
        scale = torch.ones_like(qz2_v)

        kl_divergence_z2 = kl(
            Normal(qz2_m, torch.sqrt(qz2_v)), Normal(mean, scale)
        ).sum(dim=1)
        

        
        loss_z1_unweight = -Normal(pz1_m, torch.sqrt(pz1_v)).log_prob(z1s).sum(dim=-1)
        loss_z1_weight = Normal(qz1_m, torch.sqrt(qz1_v)).log_prob(z1).sum(dim=-1)
        if not self.use_observed_lib_size:
            ql_m = inference_outputs["ql_m"]
            ql_v = inference_outputs["ql_v"]
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)

            kl_divergence_l = kl(
                Normal(ql_m, torch.sqrt(ql_v)),
                Normal(local_library_log_means, torch.sqrt(local_library_log_vars)),
            ).sum(dim=1)
        else:
            kl_divergence_l = 0.0

        if is_labelled:
            loss = reconst_loss + loss_z1_weight + loss_z1_unweight
            kl_locals = {
                "kl_divergence_z2": kl_divergence_z2,
                "kl_divergence_l": kl_divergence_l,
            }
            if labelled_tensors is not None:
                classifier_loss = self.classification_loss(labelled_tensors)
                loss += classifier_loss * classification_ratio
                return LossRecorder(
                    loss,
                    reconst_loss,
                    kl_locals,
                    classification_loss=classifier_loss,
                    n_labelled_tensors=labelled_tensors[REGISTRY_KEYS.X_KEY].shape[0],
                    #reconst_loss_replay=reconst_loss_replay,
                )
            return LossRecorder(
                loss,
                reconst_loss,
                kl_locals,
                kl_global=torch.tensor(0.0),
                #reconst_loss_replay=reconst_loss_replay,
            )

        probs = self.classifier(z1)
        reconst_loss += loss_z1_weight + (
            (loss_z1_unweight).view(self.n_labels, -1).t() * probs
        ).sum(dim=1)

        kl_divergence = (kl_divergence_z2.view(self.n_labels, -1).t() * probs).sum(
            dim=1
        )
        kl_divergence += kl(
            Categorical(probs=probs),
            Categorical(probs=self.y_prior.repeat(probs.size(0), 1)),
        )
        kl_divergence += kl_divergence_l

        loss = torch.mean(reconst_loss + kl_divergence * kl_weight)

        if labelled_tensors is not None:
            classifier_loss = self.classification_loss(labelled_tensors)
            loss += classifier_loss * classification_ratio
            return LossRecorder(
                loss,
                reconst_loss,
                kl_divergence,
                classification_loss=classifier_loss,
                #reconst_loss_replay=reconst_loss_replay,
            )
        return LossRecorder(loss, reconst_loss, kl_divergence, #reconst_loss_replay=reconst_loss_replay
                           )
    
    
    
    @auto_move_data
    def loss_with_replay(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        loss_kwargs,
       
    ):
        loss_kwargs = _get_dict_if_none(loss_kwargs)
        replay_importance = loss_kwargs['replay_importance']
        ewc_importance = loss_kwargs['ewc_importance']
        l2_ewc = loss_kwargs['l2_ewc']
        
        losses = self.loss(
            tensors, 
            inference_outputs, 
            generative_outputs, 
            **loss_kwargs
        )

        if EXTRA_KEYS.REPLAY_X_KEY in tensors.keys():
            losses_replay = self.loss(
                tensors, 
                inference_outputs, 
                generative_outputs,
                replay=True,
                **loss_kwargs
            )
        else:
            losses_replay = None

       
        
        keep_params = [n for n,p in self.old_params]
        
        cur_params = [ (n, p) for n,p in self.named_parameters() if n in keep_params]
        
        
        # should be computed here so that they appear correctly in logs even if inactive
        penalty = torch.tensor(0.0)
        penalty = penalty.to(self.device)
       
        for (_, ctrl_imp), (_, cur_param), (n, saved_param), (_, imp) in zip(
                    self.ctrl_importances,
                    cur_params,
                    self.old_params,
                    self.importances,
                ):
            if cur_param.size() == saved_param.size():
                if self.combine_type == "product":
                    penalty += ((imp * ctrl_imp) * (cur_param - saved_param).pow(2)).sum()
                if self.combine_type == "additive":
                    penalty += ((imp + ctrl_imp) * (cur_param - saved_param).pow(2)).sum()
                    
            else:
                penalty += 0.0
            
        
        # # ELBO loss - assumes replay_importance is in loss_kwargs
        if losses_replay is not None:
            loss_total = losses.loss + ewc_importance*penalty + replay_importance*losses_replay.loss
        else:
            loss_total = losses.loss + ewc_importance*penalty 

        return LossRecorder(
            loss_total, losses.reconstruction_loss, losses.kl_local, 
            replay_reconst_loss=torch.mean(losses_replay.reconstruction_loss) if losses_replay is not None else torch.tensor(0.0),
            replay_kl_loss=torch.mean(losses_replay.kl_local) if losses_replay is not None else torch.tensor(0.0),
            ewc_loss = penalty,
            # ctrl_ewc_loss = penalty_ctrl
        )  # note the component of this LossRecorder different from original
        
        
    
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
        """
        Forward pass through the network.

        Parameters
        ----------
        tensors
            tensors to pass through
        get_inference_input_kwargs
            Keyword args for ``_get_inference_input()``
        get_generative_input_kwargs
            Keyword args for ``_get_generative_input()``
        inference_kwargs
            Keyword args for ``inference()``
        generative_kwargs
            Keyword args for ``generative()``
        loss_kwargs
            Keyword args for ``loss()``
        compute_loss
            Whether to compute loss on forward pass. This adds
            another return value.
        """
        return _replay_generic_forward(
            self,
            tensors,
            inference_kwargs,
            generative_kwargs,
            loss_kwargs,
            get_inference_input_kwargs,
            get_generative_input_kwargs,
            compute_loss,
        )
    


    
def _get_dict_if_none(param):
    param = {} if not isinstance(param, dict) else param

    return param    

def _replay_generic_forward(
    module,
    tensors,
    inference_kwargs,
    generative_kwargs,
    loss_kwargs,
    get_inference_input_kwargs,
    get_generative_input_kwargs,
    compute_loss,
):
    """Core of the forward call shared by PyTorch- and Jax-based modules."""
    inference_kwargs = _get_dict_if_none(inference_kwargs)
    generative_kwargs = _get_dict_if_none(generative_kwargs)
#     loss_kwargs = _get_dict_if_none(loss_kwargs)
    get_inference_input_kwargs = _get_dict_if_none(get_inference_input_kwargs)
    get_generative_input_kwargs = _get_dict_if_none(get_generative_input_kwargs)

    inference_inputs = module._get_inference_input(
        tensors,  **get_inference_input_kwargs
    )

    inference_outputs = module.inference(**inference_inputs, **inference_kwargs)

    generative_inputs = module._get_generative_input(
        tensors, inference_outputs, **get_generative_input_kwargs
    )

    generative_outputs = module.generative(**generative_inputs, **generative_kwargs)



    


    if compute_loss:
        losses = module.loss_with_replay(
            tensors, 
            inference_outputs, 
            generative_outputs,
            loss_kwargs # **loss_kwargs  -- replay_importance is passed here 
        )

        return inference_outputs, generative_outputs, losses
    else:
        return inference_outputs, generative_outputs
