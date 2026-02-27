
<img width="1694" height="646" alt="Supp_fig_modelOverview" src="https://github.com/user-attachments/assets/56a5846d-05d2-4987-8b44-7afc59d59979" />

# Comparative Atlas Construction with Bregman-Regularized Replay


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
cd your-repo
```

## 2. Setup a conda environment
Use the `.yml` file provided in the repo

``` bash
conda env create -f environment.yml
conda activate cscanvi
```
