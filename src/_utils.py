from typing import Iterable, Optional, Sequence, Union, Tuple
from scvi.module.base import LossRecorder, auto_move_data

import torch
from torch.utils.data import DataLoader
from scvi.data import AnnDataManager

from scvi.dataloaders._anntorchdataset import AnnTorchDataset






def _update_adata_and_manager_post_minification(
        self,
        minified_adata: AnnData,
        minified_data_type: MinifiedDataType,
    ):
        """Update the :class:`~anndata.AnnData` and :class:`~scvi.data.AnnDataManager` in-place.

        Parameters
        ----------
        minified_adata
            Minified version of :attr:`~scvi.model.base.BaseModelClass.adata`.
        minified_data_type
            Method used for minifying the data.
        keep_count_data
            If ``True``, the full count matrix is kept in the minified
            :attr:`~scvi.model.base.BaseModelClass.adata`.
        """
        self._validate_anndata(minified_adata)
        new_adata_manager = self.get_anndata_manager(minified_adata, required=True)
        new_adata_manager.register_new_fields(
            self._get_fields_for_adata_minification(minified_data_type)
        )
        self.adata = minified_adata

def register_new_fields(self, fields: list[AnnDataField]):
        """Register new fields to a manager instance.

        This is useful to augment the functionality of an existing manager.

        Parameters
        ----------
        fields
            List of AnnDataFields to register
        """
        if self.adata is None:
            raise AssertionError(
                "No AnnData object has been registered with this Manager instance."
            )
        self.validate()
        for field in fields:
            self._add_field(
                field=field,
                adata=self.adata,
            )

        # Source registry is not None if this manager was created from transfer_fields
        # In this case self._registry is originally equivalent to self._source_registry
        # However, with newly registered fields the equality breaks so we reset it
        if self._source_registry is not None:
            self._source_registry = deepcopy(self._registry)

        self.fields += fields
    
def _add_field(
        self,
        field,
        adata: AnnData,
        source_registry: dict | None = None,
        **transfer_kwargs,
    ):
        """Internal function for adding a field with optional transferring."""
        field_registries = self._registry[_constants._FIELD_REGISTRIES_KEY]
        field_registries[field.registry_key] = {
            _constants._DATA_REGISTRY_KEY: field.get_data_registry(),
            _constants._STATE_REGISTRY_KEY: {},
        }
        field_registry = field_registries[field.registry_key]

        # A field can be empty if the model has optional fields (e.g. extra covariates).
        # If empty, we skip registering the field.
        if not field.is_empty:
            # Transfer case: Source registry is used for validation and/or setup.
            if source_registry is not None:
                field_registry[_constants._STATE_REGISTRY_KEY] = field.transfer_field(
                    source_registry[_constants._FIELD_REGISTRIES_KEY][field.registry_key][
                        _constants._STATE_REGISTRY_KEY
                    ],
                    adata,
                    **transfer_kwargs,
                )
            else:
                field_registry[_constants._STATE_REGISTRY_KEY] = field.register_field(adata)
        # Compute and set summary stats for the given field.
        state_registry = field_registry[_constants._STATE_REGISTRY_KEY]
        field_registry[_constants._SUMMARY_STATS_KEY] = field.get_summary_stats(state_registry)

 

class AnnDataLoader(DataLoader):
    """
    DataLoader for loading tensors from AnnData objects.
    Parameters
    ----------
    adata_manager
        :class:`~scvi.data.AnnDataManager` object with a registered AnnData object.
    shuffle
        Whether the data should be shuffled
    indices
        The indices of the observations in the adata to load
    batch_size
        minibatch size to load each iteration
    data_and_attributes
        Dictionary with keys representing keys in data registry (``adata_manager.data_registry``)
        and value equal to desired numpy loading type (later made into torch tensor).
        If ``None``, defaults to all registered data.
    data_loader_kwargs
        Keyword arguments for :class:`~torch.utils.data.DataLoader`
    iter_ndarray
        Whether to iterate over numpy arrays instead of torch tensors
    """

    def __init__(
        self,
        adata_manager: AnnDataManager,
        replay_adata_manager: AnnDataManager,
        shuffle=False,
        indices=None,
        batch_size=128,
        data_and_attributes: Optional[dict] = None,
        drop_last: Union[bool, int] = False,
        iter_ndarray: bool = False,
        **data_loader_kwargs,
    ):
        if adata_manager.adata is None:
            raise ValueError(
                "Please run register_fields() on your AnnDataManager object first."
            )

        if data_and_attributes is not None:
            data_registry = adata_manager.data_registry
            for key in data_and_attributes.keys():
                if key not in data_registry.keys():
                    raise ValueError(
                        f"{key} required for model but not registered with AnnDataManager."
                    )

        self.dataset = AnnTorchDataset(
            adata_manager, getitem_tensors=data_and_attributes
        )

        sampler_kwargs = {
            "batch_size": batch_size,
            "shuffle": shuffle,
            "drop_last": drop_last,
        }

        if indices is None:
            indices = np.arange(len(self.dataset))
            sampler_kwargs["indices"] = indices
        else:
            if hasattr(indices, "dtype") and indices.dtype is np.dtype("bool"):
                indices = np.where(indices)[0].ravel()
            indices = np.asarray(indices)
            sampler_kwargs["indices"] = indices

        self.indices = indices
        self.sampler_kwargs = sampler_kwargs
        sampler = BatchSampler(**self.sampler_kwargs)
        self.data_loader_kwargs = copy.copy(data_loader_kwargs)
        # do not touch batch size here, sampler gives batched indices
        self.data_loader_kwargs.update({"sampler": sampler, "batch_size": None})

        if iter_ndarray:
            self.data_loader_kwargs.update({"collate_fn": _dummy_collate})

        super().__init__(self.dataset, **self.data_loader_kwargs)
