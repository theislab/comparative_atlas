import logging
import warnings
from copy import deepcopy
from typing import List, Optional, Sequence, Union, Iterable, Tuple

import numpy as np
import pandas as pd
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
from scvi.dataloaders import SemiSupervisedDataSplitter
from scvi.model._utils import _init_library_size, parse_use_gpu_arg
# from scvi.module import SCANVAE
from ._scanvae import SCANVAE
from ._utils import mask_augment, compute_uncertainty_scores, BI_LSE


from scvi.dataloaders._ann_dataloader import AnnDataLoader
from torch.utils.data.dataloader import default_collate



# from scvi.train import SemiSupervisedTrainingPlan, TrainRunner
from ._trainingplans import CLSemiSupervisedTrainingPlan
from scvi.train import TrainRunner
from scvi.train._callbacks import SubSampleLabels
from scvi.utils import setup_anndata_dsp

# ArchesMixin-specific params
from scvi.model.base._utils import _initialize_model, _load_saved_files, _validate_var_names
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
from scvi.data import _constants
from scvi.nn import FCLayers


from scvi.model._scvi import SCVI
from scvi.model.base import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin

logger = logging.getLogger(__name__)

# adds-on
from typing import NamedTuple


class SCANVI(RNASeqMixin, VAEMixin, ArchesMixin, BaseModelClass):
    """
    Single-cell annotation using variational inference [Xu21]_.
    Inspired from M1 + M2 model, as described in (https://arxiv.org/pdf/1406.5298.pdf).
    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCANVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:
        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    **model_kwargs
        Keyword args for :class:`~scvi.module.SCANVAE`
    Examples
    --------
    >>> adata = anndata.read_h5ad(path_to_anndata)
    >>> scvi.model.SCANVI.setup_anndata(adata, batch_key="batch", labels_key="labels")
    >>> vae = scvi.model.SCANVI(adata, "Unknown")
    >>> vae.train()
    >>> adata.obsm["X_scVI"] = vae.get_latent_representation()
    >>> adata.obs["pred_label"] = vae.predict()
    Notes
    -----
    See further usage examples in the following tutorials:
    1. :doc:`/tutorials/notebooks/harmonization`
    2. :doc:`/tutorials/notebooks/scarches_scvi_tools`
    3. :doc:`/tutorials/notebooks/seed_labeling`
    """

    def __init__(
        self,
        adata: AnnData,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene-cell",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        **model_kwargs,
    ):
        super(SCANVI, self).__init__(adata)
        scanvae_model_kwargs = dict(model_kwargs)

        self._set_indices_and_labels()

        # ignores unlabeled catgegory
        n_labels = self.summary_stats.n_labels - 1
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )

        n_batch = self.summary_stats.n_batch
        use_size_factor_key = (
            REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        )
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(
                self.adata_manager, n_batch
            )
            
        

        self.module = SCANVAE(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            # n_control = self.control_size,
            **scanvae_model_kwargs,
        )

        self.unsupervised_history_ = None
        self.semisupervised_history_ = None
        
   

        self._model_summary_string = (
        "ScanVI Model with the following params: \nunlabeled_category: {}, n_hidden: {}, n_latent: {}"
        ", n_layers: {}, dropout_rate: {}, dispersion: {}, gene_likelihood: {}"
        ).format(
            self.unlabeled_category_,
            n_hidden,
            n_latent,
            n_layers,
            dropout_rate,
            dispersion,
            gene_likelihood,

        )
            

        self.init_params_ = self._get_init_params(locals())
        self.was_pretrained = False
        
        

    @classmethod
    def from_scvi_model(
        cls,
        scvi_model: SCVI,
        unlabeled_category: str,
        labels_key: Optional[str] = None,
        adata: Optional[AnnData] = None,
        **scanvi_kwargs,
    ):
        """
        Initialize scanVI model with weights from pretrained :class:`~scvi.model.SCVI` model.
        Parameters
        ----------
        scvi_model
            Pretrained scvi model
        labels_key
            key in `adata.obs` for label information. Label categories can not be different if
            labels_key was used to setup the SCVI model. If None, uses the `labels_key` used to
            setup the SCVI model. If that was None, and error is raised.
        unlabeled_category
            Value used for unlabeled cells in `labels_key` used to setup AnnData with scvi.
        adata
            AnnData object that has been registered via :meth:`~scvi.model.SCANVI.setup_anndata`.
        scanvi_kwargs
            kwargs for scANVI model
        """
        scvi_model._check_if_trained(
            message="Passed in scvi model hasn't been trained yet."
        )

        scanvi_kwargs = dict(scanvi_kwargs)
        init_params = scvi_model.init_params_
        non_kwargs = init_params["non_kwargs"]
        kwargs = init_params["kwargs"]
        kwargs = {k: v for (i, j) in kwargs.items() for (k, v) in j.items()}
        for k, v in {**non_kwargs, **kwargs}.items():
            if k in scanvi_kwargs.keys():
                warnings.warn(
                    "Ignoring param '{}' as it was already passed in to ".format(k)
                    + "pretrained scvi model with value {}.".format(v)
                )
                del scanvi_kwargs[k]

        if adata is None:
            adata = scvi_model.adata
        else:
            # validate new anndata against old model
            scvi_model._validate_anndata(adata)

        scvi_setup_args = deepcopy(scvi_model.adata_manager.registry[_SETUP_ARGS_KEY])
        scvi_labels_key = scvi_setup_args["labels_key"]
        if labels_key is None and scvi_labels_key is None:
            raise ValueError(
                "A `labels_key` is necessary as the SCVI model was initialized without one."
            )
        if scvi_labels_key is None:
            scvi_setup_args.update(dict(labels_key=labels_key))
        cls.setup_anndata(
            adata,
            unlabeled_category=unlabeled_category,
            **scvi_setup_args,
        )
        scanvi_model = cls(adata, **non_kwargs, **kwargs, **scanvi_kwargs)
        scvi_state_dict = scvi_model.module.state_dict()
        scanvi_model.module.load_state_dict(scvi_state_dict, strict=False)
        scanvi_model.was_pretrained = True

        return scanvi_model

    def _set_indices_and_labels(self):
        """
        Set indices for labeled and unlabeled cells.
        """
        labels_state_registry = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.LABELS_KEY
        )
        self.original_label_key = labels_state_registry.original_key
        self.unlabeled_category_ = labels_state_registry.unlabeled_category

        labels = get_anndata_attribute(
            self.adata,
            self.adata_manager.data_registry.labels.attr_name,
            self.original_label_key,
        ).ravel()
        self._label_mapping = labels_state_registry.categorical_mapping

        # set unlabeled and labeled indices
        self._unlabeled_indices = np.argwhere(
            labels == self.unlabeled_category_
        ).ravel()
        self._labeled_indices = np.argwhere(labels != self.unlabeled_category_).ravel()
        self._code_to_label = {i: l for i, l in enumerate(self._label_mapping)}

    def predict(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        soft: bool = False,
        batch_size: Optional[int] = None,
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Return cell label predictions.
        Parameters
        ----------
        adata
            AnnData object that has been registered via :meth:`~scvi.model.SCANVI.setup_anndata`.
        indices
            Return probabilities for each class label.
        soft
            If True, returns per class probabilities
        batch_size
            Minibatch size for data loading into model. Defaults to `scvi.settings.batch_size`.
        """
        adata = self._validate_anndata(adata)

        if indices is None:
            indices = np.arange(adata.n_obs)

        scdl = self._make_data_loader(
            adata=adata,
            indices=indices,
            batch_size=batch_size,
        )
        y_pred = []
        for _, tensors in enumerate(scdl):
            x = tensors[REGISTRY_KEYS.X_KEY]
            batch = tensors[REGISTRY_KEYS.BATCH_KEY]

            cont_key = REGISTRY_KEYS.CONT_COVS_KEY
            cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

            cat_key = REGISTRY_KEYS.CAT_COVS_KEY
            cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

            pred = self.module.classify(
                x, batch_index=batch, cat_covs=cat_covs, cont_covs=cont_covs
            )
            if not soft:
                pred = pred.argmax(dim=1)
            y_pred.append(pred.detach().cpu())

        y_pred = torch.cat(y_pred).numpy()
        if not soft:
            predictions = []
            for p in y_pred:
                predictions.append(self._code_to_label[p])

            return np.array(predictions)
        else:
            n_labels = len(pred[0])
            pred = pd.DataFrame(
                y_pred,
                columns=self._label_mapping[:n_labels],
                index=adata.obs_names[indices],
            )
            return pred

    def train(
        self,
        max_epochs: Optional[int] = None,
        n_samples_per_label: Optional[float] = None,
        check_val_every_n_epoch: Optional[int] = None,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        batch_size: int = 128,
        use_gpu: Optional[Union[str, int, bool]] = None,
        plan_kwargs: Optional[dict] = None,
        **trainer_kwargs,
    ):
        """
        Train the model.
        Parameters
        ----------
        max_epochs
            Number of passes through the dataset for semisupervised training.
        n_samples_per_label
            Number of subsamples for each label class to sample per epoch. By default, there
            is no label subsampling.
        check_val_every_n_epoch
            Frequency with which metrics are computed on the data for validation set for both
            the unsupervised and semisupervised trainers. If you'd like a different frequency for
            the semisupervised trainer, set check_val_every_n_epoch in semisupervised_train_kwargs.
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size
            Minibatch size to use during training.
        use_gpu
            Use default GPU if available (if None or True), or index of GPU to use (if int),
            or name of GPU (if str, e.g., `'cuda:0'`), or use CPU (if False).
        plan_kwargs
            Keyword args for :class:`~scvi.train.SemiSupervisedTrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **trainer_kwargs
            Other keyword args for :class:`~scvi.train.Trainer`.
        """

        if max_epochs is None:
            n_cells = self.adata.n_obs
            max_epochs = np.min([round((20000 / n_cells) * 400), 400])

            if self.was_pretrained:
                max_epochs = int(np.min([10, np.max([2, round(max_epochs / 3.0)])]))

        logger.info("Training for {} epochs.".format(max_epochs))

        plan_kwargs = {} if plan_kwargs is None else plan_kwargs

        # if we have labeled cells, we want to subsample labels each epoch
        sampler_callback = (
            [SubSampleLabels()] if len(self._labeled_indices) != 0 else []
        )

        data_splitter = SemiSupervisedDataSplitter(
            adata_manager=self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            n_samples_per_label=n_samples_per_label,
            batch_size=batch_size,
            use_gpu=use_gpu,
        )
        training_plan = CLSemiSupervisedTrainingPlan(self.module, **plan_kwargs)
        if "callbacks" in trainer_kwargs.keys():
            trainer_kwargs["callbacks"].concatenate(sampler_callback)
        else:
            trainer_kwargs["callbacks"] = sampler_callback

        runner = TrainRunner(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            use_gpu=use_gpu,
            check_val_every_n_epoch=check_val_every_n_epoch,
            **trainer_kwargs,
        )
        return runner()

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        labels_key: str,
        unlabeled_category: str ,
        layer: Optional[str] = None,
        batch_key: Optional[str] = None,
        size_factor_key: Optional[str] = None,
        categorical_covariate_keys: Optional[List[str]] = None,
        continuous_covariate_keys: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        %(summary)s.
        Parameters
        ----------
        %(param_layer)s
        %(param_batch_key)s
        %(param_labels_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            LabelsWithUnlabeledObsField(
                REGISTRY_KEYS.LABELS_KEY, labels_key, unlabeled_category
            ),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        
        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)
        
        
    @classmethod
    def get_uncertainty(
        cls,
        adata: AnnData,
        reference_model: Union[str, BaseModelClass],
        num_points: int = 10,
        order = 'top-k',
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        tta_rep: int=10,
    ):
        
        reference_model._check_if_trained(warn=False)

        adata = reference_model._validate_anndata(adata)
        scdl = reference_model._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        unc_scores = []
        device = reference_model.device
        for tensors in scdl:
            inference_inputs = reference_model.module._get_inference_input(tensors)
            unc_batch = compute_uncertainty_scores(inference_inputs, reference_model, device, tta_rep=10)
            unc_scores.extend(unc_batch)
        if order == 'top-k':
            score_idx = torch.sort(torch.tensor(unc_scores), descending=True)[1][:num_points]
        elif order == 'bottom-k':
            score_idx = torch.sort(torch.tensor(unc_scores), descending=False)[1][:num_points]
        elif order == 'step':
            skip = len(unc_scores) // num_points
            steps = np.arange(0, len(unc_scores), skip)
            score_idx = torch.sort(torch.tensor(unc_scores), descending=True)[1][steps]
        else:
            raise ValueError(f"Invalid value for 'order': {order}. Expected 'top-k', 'bottom-k' or 'step' for the 'order' parameter.")
        return unc_scores, score_idx
           

    @torch.no_grad()
    def get_px_rate(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
    ) -> np.ndarray:

        
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            inference_outputs = self.module.inference(**inference_inputs)
            generative_inputs = self.module._get_generative_input(tensors, inference_outputs)
            generative_outputs = self.module.generative(**generative_inputs)
            px_rate = generative_outputs["px_rate"]
            latent += [px_rate.cpu()]
        return torch.cat(latent).numpy() 
    
    
        
    @classmethod
    def load_query_data_with_replay(
        cls,
        adata: AnnData,
        reference_model: Union[str, BaseModelClass],
        control_uns_key: str = None,
        replay_uns_key: str = None,
        inplace_subset_query_vars: bool = False,
        use_gpu: Optional[Union[str, int, bool]] = None,
        unfrozen: bool = True, # fully unfrozen model. Overrides everything else
        freeze_dropout: bool = False,
        freeze_expression: bool = True,
        freeze_decoder_first_layer: bool = True,
        freeze_batchnorm_encoder: bool = True,
        freeze_batchnorm_decoder: bool = False,
        freeze_classifier: bool = True,
    ):
        """
        Online update of a reference model with scArches algorithm [Lotfollahi21]_.
        Parameters
        ----------
        adata
            AnnData organized in the same way as data used to train model.
            It is not necessary to run setup_anndata,
            as AnnData is validated against the ``registry``.
        reference_model
            Either an already instantiated model of the same class, or a path to
            saved outputs for reference model.
        inplace_subset_query_vars
            Whether to subset and rearrange query vars inplace based on vars used to
            train reference model.
        use_gpu
            Load model on default GPU if available (if None or True),
            or index of GPU to use (if int), or name of GPU (if str), or use CPU (if False).
        unfrozen
            Override all other freeze options for a fully unfrozen model
        freeze_dropout
            Whether to freeze dropout during training
        freeze_expression
            Freeze neurons corersponding to expression in first layer
        freeze_decoder_first_layer
            Freeze neurons corersponding to first layer in decoder
        freeze_batchnorm_encoder
            Whether to freeze batchnorm weight and bias during training for encoder
        freeze_batchnorm_decoder
            Whether to freeze batchnorm weight and bias during training for decoder
        freeze_classifier
            Whether to freeze classifier completely. Only applies to `SCANVI`.
        """
        use_gpu, device = parse_use_gpu_arg(use_gpu)

        attr_dict, var_names, load_state_dict = _get_loaded_data(
            reference_model, device=device
        )

        if inplace_subset_query_vars:
            logger.debug("Subsetting query vars to reference vars.")
            adata._inplace_subset_var(var_names)
        _validate_var_names(adata, var_names)

        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError(
                "It appears you are loading a model from a different class."
            )

        if _SETUP_ARGS_KEY not in registry:
            raise ValueError(
                "Saved model does not contain original setup inputs. "
                "Cannot load the original setup."
            )
        

        
    
        cls.setup_anndata(
            adata,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY],
        )

        model = _initialize_model(cls, adata, attr_dict)
        adata_manager = model.get_anndata_manager(adata, required=True)
        model.old_adata_manager = adata_manager

        
        

        if REGISTRY_KEYS.CAT_COVS_KEY in adata_manager.data_registry:
            raise NotImplementedError(
                "scArches currently does not support models with extra categorical covariates."
            )

        version_split = adata_manager.registry[_constants._SCVI_VERSION_KEY].split(".")
        if int(version_split[1]) < 8 and int(version_split[0]) == 0:
            warnings.warn(
                "Query integration should be performed using models trained with version >= 0.8"
            )

        model.to_device(device)

        # make a copy of the model so that it is protected from change when computing importances
        old_model = deepcopy(model)

        # model tweaking
        new_state_dict = model.module.state_dict()
        for key, load_ten in load_state_dict.items():
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                continue
            # new categoricals changed size
            else:
                dim_diff = new_ten.size()[-1] - load_ten.size()[-1]
                fixed_ten = torch.cat([load_ten, new_ten[..., -dim_diff:]], dim=-1)
                load_state_dict[key] = fixed_ten

        model.module.load_state_dict(load_state_dict)
        # was this potentially a wrong place to make a copy?
        old_model_batch_extend = deepcopy(model)
        model.module.eval()
        
        
        

        _set_params_online_update(
            model.module,
            unfrozen=unfrozen,
            freeze_decoder_first_layer=freeze_decoder_first_layer,
            freeze_batchnorm_encoder=freeze_batchnorm_encoder,
            freeze_batchnorm_decoder=freeze_batchnorm_decoder,
            freeze_dropout=freeze_dropout,
            freeze_expression=freeze_expression,
            freeze_classifier=freeze_classifier,
        )
        model.is_trained_ = False
        # model.replay_size = np.unique(adata.obsm[replay_x_key],axis=0).shape[0]
        # model.control_size = adata.uns[control_uns_key].shape[0] if control_uns_key is not None else None
         
            
        # update model summary
        model._model_summary_string = ( "{}, n_replay:{}, n_control:{}"
        ).format(
            model._model_summary_string,
            None,
            None,
            # model.replay_size,
            # model.control_size,
        )


        

        ##### control and replay dataloaders ----------------

        # replay_adata = AnnData(X=adata.uns[replay_uns_key])
        replay_adata = adata.uns[replay_uns_key]
        replay_adata = model._validate_anndata(replay_adata)
        rehearsal_rdl = model._make_data_loader(replay_adata, batch_size=256)

        ctrl_adata = adata.uns[control_uns_key] # this is a anndata
        ctrl_adata = old_model_batch_extend._validate_anndata(ctrl_adata)
        ctrl_rdl = old_model_batch_extend._make_data_loader(ctrl_adata, batch_size=256)
        
        # if model.control_size is not None:
        #     ctrl_adata = adata.uns[control_uns_key] # this is a anndata
        #     ctrl_adata = old_model_batch_extend._validate_anndata(ctrl_adata)
        #     ctrl_rdl = old_model_batch_extend._make_data_loader(ctrl_adata, batch_size=256) # model._make_data_loader
        # else:
        #     None
        
        
        ##### parameter regularisation ----------------

        # can we make importances available as buffer? only allowed for tensors
        model.module.importances = model._compute_importances(model = old_model, dataloader = rehearsal_rdl)
        # importances = model._compute_importances(model = old_model, dataloader = rehearsal_rdl)
        # model.module.register_buffer("importances", importances)
        
        
        if control_uns_key is not None:
            model.module.ctrl_importances = model._compute_importances(model = old_model_batch_extend, dataloader = ctrl_rdl)
            # ctrl_importances = model._compute_importances(model = old_model_batch_extend, dataloader = ctrl_rdl)
            # model.module.register_buffer("ctrl_importances", ctrl_importances)
        
        #check this may need to be old_model_batch_extend instead
        model.module.old_params =  [
                (k, p.clone().detach())
                for k, p in old_model.module.named_parameters() if p.requires_grad
                # for k, p in old_model.module.named_parameters() if requires_penalty(k)
            ]
        # old_params = [
        #         (k, p.clone().detach())
        #         for k, p in old_model.module.named_parameters() if requires_penalty(k)
        #     ]
        # model.module.register_buffer("old_params", old_params)
        # model.module.register_buffer ("ref_px_r", model.module.px_r)

        
        
        return model 
    
 
    def _compute_importances(
        self,
        model, 
        dataloader, 
    ):

        # initialize importances
        importances = zerolike_params_dict(model.module)
        
        # setup optimizer
        params = filter(lambda p: p.requires_grad, model.module.parameters())
        optimizer = torch.optim.Adam(
            params, lr=1e-3, eps=0.01, weight_decay=1e-6
        )
        
        # setup device
        device = model.device
        
        model.module.eval()

        for i, batch in enumerate(dataloader):
            # get only input, target and task_id from the batch
            tensors = batch

            ## move tensors to device 
            for key, val in tensors.items():
                tensors[key] = val.to(device)

            optimizer.zero_grad()
            inference_inputs = model.module._get_inference_input(tensors)
            inference_outputs = model.module.inference(**inference_inputs)

            generative_inputs = model.module._get_generative_input(tensors, inference_outputs)
            generative_outputs = model.module.generative(**generative_inputs)
            scvi_loss = model.module.loss(tensors, inference_outputs, generative_outputs)
            loss = scvi_loss.loss
            loss.backward()

            # param_dict = [ (n,p) for n,p in model.module.named_parameters() if requires_penalty(n)]
            param_dict = [ (n,p) for n,p in model.module.named_parameters() if p.requires_grad]


            for (k1, p), (k2, imp) in zip(
                param_dict, importances
            ):
                assert k1 == k2
                if p.grad is not None:
                    imp += p.grad.data.clone().pow(2)

        # average over mini batch length
        for _, imp in importances:
            imp /= float(len(dataloader))

        return importances   
    

    
def requires_penalty(key):
    one = 'z_encoder' in key.split(".")[0] 
    ten = 'l_encoder' in key.split(".")[0] 
#             one = 'z_encoder.encoder.fc_layers.Layer 0.0' in key
#             two = 'decoder.px_decoder.fc_layers.Layer 0.0' in key
#             one = 'encoder' in key.split(".") 
    two = 'decoder' in key.split(".") 
    three = 'classifier' in key.split(".") 
    four = 'encoder_z2_z1' in key.split(".")
    five = 'decoder_z1_z2' in key.split(".")
    # five = one and 'z_encoder.encoder.fc_layers.Layer 0.0' not in key
#             five = one and 'z_encoder.encoder.fc_layers.Layer 1.0' not in key 
#             five = one and 'z_encoder.encoder.fc_layers.Layer 1.0' not in key and 'z_encoder.encoder.fc_layers.Layer 0.0' not in key

    six = two and 'decoder.px_decoder.fc_layers.Layer 0.0.' not in key
    seven = 'px_decoder' not in key.split(".") 
    nine = 'px_scale_decoder' not in key.split(".")
    eight = two and seven 
#             eight = two and seven and nine
#             two = 'px_decoder' in key.split(".")
#             three = 'px_scale_decoder' in key.split(".")
#             four = 'px_r_decoder' in key.split(".")
#             five = 'px_dropout_decoder' in key.split(".")

#             if one or (two and 'decoder.px_decoder.fc_layers.Layer 0.0.' not in key): # 
#             if (two and 'decoder.px_decoder.fc_layers.Layer 0.0.' not in key) or three or four: # 
#             if one or (two and 'decoder.px_decoder.fc_layers.Layer 0.0.' not in key) or three or four: # 
#             if five or eight or three or four: # 
#             if five or two or three or four:
#             if five or six or three or four or ten: # this works
    if one or ten or two or three or four or five: 
        return True
    else:
        return False



def zerolike_params_dict(model):
    """
    Create a list of (name, parameter), where parameter is initalized to zero.
    The list has as many parameters as model, with the same size.
    :param model: a pytorch model
    """

    return [
        (k, torch.zeros_like(p).to(p.device))
        for k, p in model.named_parameters() if p.requires_grad
        # for k, p in model.named_parameters() if requires_penalty(k)
    ]

     
    
    
def _set_params_online_update(
    module,
    unfrozen,
    freeze_decoder_first_layer,
    freeze_batchnorm_encoder,
    freeze_batchnorm_decoder,
    freeze_dropout,
    freeze_expression,
    freeze_classifier,
):
    """Freeze parts of network for scArches."""
    # do nothing if unfrozen
    if unfrozen:
        return

    mod_inference_mode = {"encoder_z2_z1", "decoder_z1_z2"}
    mod_no_hooks_yes_grad = {"l_encoder"}
    if not freeze_classifier:
        mod_no_hooks_yes_grad.add("classifier")
    parameters_yes_grad = {"background_pro_alpha", "background_pro_log_beta"}

    def no_hook_cond(key):
        one = (not freeze_expression) and "encoder" in key
        two = (not freeze_decoder_first_layer) and "px_decoder" in key
        return one or two

    def requires_grad(key):
        mod_name = key.split(".")[0]
        # linear weights and bias that need grad
        one = "fc_layers" in key and ".0." in key and mod_name not in mod_inference_mode
        # modules that need grad
        two = mod_name in mod_no_hooks_yes_grad
        three = sum([p in key for p in parameters_yes_grad]) > 0
        # batch norm option
        four = (
            "fc_layers" in key
            and ".1." in key
            and "encoder" in key
            and (not freeze_batchnorm_encoder)
        )
        five = (
            "fc_layers" in key
            and ".1." in key
            and "decoder" in key
            and (not freeze_batchnorm_decoder)
        )
        if one or two or three or four or five:
            return True
        else:
            return False

    for key, mod in module.named_modules():
        # skip over protected modules
        if key.split(".")[0] in mod_no_hooks_yes_grad:
            continue
        if isinstance(mod, FCLayers):
            hook_first_layer = False if no_hook_cond(key) else True
            mod.set_online_update_hooks(hook_first_layer)
        if isinstance(mod, torch.nn.Dropout):
            if freeze_dropout:
                mod.p = 0
        # momentum freezes the running stats of batchnorm
        freeze_batchnorm = ("decoder" in key and freeze_batchnorm_decoder) or (
            "encoder" in key and freeze_batchnorm_encoder
        )
        if isinstance(mod, torch.nn.BatchNorm1d) and freeze_batchnorm:
            mod.momentum = 0

    for key, par in module.named_parameters():
        if requires_grad(key):
            par.requires_grad = True
        else:
            par.requires_grad = False
    
    


def _get_loaded_data(reference_model, device=None):
    if isinstance(reference_model, str):
        attr_dict, var_names, load_state_dict, _ = _load_saved_files(
            reference_model, load_adata=False, map_location=device
        )
    else:
        attr_dict = reference_model._get_user_attributes()
        attr_dict = {a[0]: a[1] for a in attr_dict if a[0][-1] == "_"}
        var_names = reference_model.adata.var_names
        load_state_dict = deepcopy(reference_model.module.state_dict())

    return attr_dict, var_names, load_state_dict