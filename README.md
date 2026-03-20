
<img width="1694" height="646" alt="Supp_fig_modelOverview" src="https://github.com/user-attachments/assets/56a5846d-05d2-4987-8b44-7afc59d59979" />

# Incremental comparative Atlas Construction with Bregman-Regularized Replay


This repository implements an incremental comparative atlas construction framework using:

1. **Bregman Information (BI)** to build a replay buffer  
2. **Fisher Information–based importance weighting**  
3. **Regularized incremental model updates**  

The workflow enables robust case–control integration without catastrophic forgetting.

---

# Overview of the Method

The pipeline consists of four stages:

### Input
- **Integrated reference atlas** (multi-study healthy reference)
- **Case–control query data**

### (1) Compute Bregman Information → Create Replay Buffer
Select informative reference cells to preserve during continual training.

### (2) Compute Importance Weights
Estimate Fisher Information for encoder and decoder weights.

### Comparative Atlas Construction
Perform incremental update with regularization from (1) and (2).

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/theislab/comparative_atlas
cd comparative_atlas
```

## 2. Setup a conda environment
Use the `.yml` file provided in the repo

``` bash
conda env create -f environment.yml
conda activate cscanvi
```

# Quick start

Here we provide an example on training a `scANVI` model incrementally. You can download a simulated case–control PBMC scRNA-seq dataset—featuring increased IFN signaling in monocytes from case samples—along with the corresponding reference model from [this link](https://doi.org/10.6084/m9.figshare.31825075). 

Import the modified `SCANVI` class from source
```python
from cscanvi._scanvi import SCANVI
```

Construct a Replay Buffer by computing the Bregman Information metric for each cell.
Here we select 20% of cells from the reference model , `ref_model`. The gene expression counts from the atlas are stored in `adata`. We compute BI by generating 200 augmentations to score each, then choose cells following the `step` approach. 

```python
import scvi
ref_model = scvi.model.SCANVI.load(ref_model_path, adata)

prop_cells_to_replay = 0.2
num_points_bi = int(adata.n_obs * prop_cells_to_replay)

N=200

unc_scores, score_idx  = SCANVI.get_uncertainty(adata, 
                                                ref_model, 
                                                order='step', 
                                                num_points = num_points_bi,
                                                tta_rep = N)

adata_healthyRef_replay = adata[score_idx.detach().cpu().numpy()]
```
Next we compute Fisher Information to estimate parameter importance. To compute Fisher Information, we first need to select a subset of control cells from the query:

```python
# select a small proportion of query control cells for computing Fisher Information 
healthy_controls = (query_adata.obs.condition.isin(['control']))
adata_queryCtrl = sc.pp.subsample(query_adata[healthy_controls].copy(), 0.5, copy = True)

# concatenate reply buffer with query data
query_adata = query_adata.concatenate(adata_healthyRef_replay)

# add the query-control subset and replay buffer to uns. 
query_adata.uns['ctrl_query'] = adata_queryCtrl
query_adata.uns['replay_adata'] = adata_healthyRef_replay

# compute importance weights
query_model = SCANVI.load_query_data_with_replay(query_adata, 
                                                 reference_model = ref_model_path,
                                                 unfrozen=True,
                                                 control_uns_key = 'ctrl_query',
                                                 replay_uns_key = 'replay_adata'
                                                )
```

Set the desired value for `ewc_importance` (regularization strenght) and train:

```python
contl_epochs = 150
train_kwargs_surgery = {
    "early_stopping": True,
    "early_stopping_monitor": "elbo_train",
    "early_stopping_patience": 10,
    "early_stopping_min_delta": 0.001,
    "plan_kwargs": {"ewc_importance": 0.1 ,"weight_decay": 0.0},
}


query_model.train(
    max_epochs=contl_epochs,
    **train_kwargs_surgery, 
)
```


# Reproducibility

The scANVI models of the comparative CRC all-lineage, Epithelial lineage and NK-T cell lineage integrations, and the notebooks to reproduce the figures from the manuscript will be released progressively.

# Citation