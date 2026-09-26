# MixRoute: Mixed Parameterization with Adaptive Information Routing for Time Series Forecasting with Exogenous Variables

This repository contains the runnable code for the submitted paper *MixRoute*. It
is provided for double-blind review; author identity and the paper reference will
be revealed upon acceptance.

MixRoute is a unified Transformer framework that makes representation and
information flow type-aware for endogenous patches, exogenous variables, and
learnable global tokens. Under **Mixed Parameterization**, endogenous patches
share one set of parameters, whereas each exogenous and global token has a private
set. **Adaptive Information Routing (AIR)** uses separate queries for an intrinsic
stream over the endogenous history and a cross-type stream over the exogenous and
global tokens, and then combines the stream outputs with a gate computed from both
streams.

## Quickstart

### 1. Requirements

```shell
pip install -r requirements.txt
```

The code was developed and tested with Python 3.8+ and PyTorch 2.4 on a single
NVIDIA RTX 4090 GPU.

### 2. Data preparation

The 12 benchmark datasets (BE, DE, FR, NP, PJM, Energy, Colbun, Rapel,
Sdwpfm1, Sdwpfm2, Sdwpfh1, Sdwpfh2) are the covariate-forecasting data used by
recent TSF-X papers such as DAG and GCGNet, and can be obtained from the public
DAG release:

- Datasets (Google Drive link quoted from the DAG repository):
  <https://drive.google.com/file/d/1K2AvogpOpSz1PiQ53dPchzGv_PqlCWAK/view?usp=sharing>

Place the 12 CSV files under `./dataset/forecasting/`. We do not redistribute the
raw data in this anonymous package. The package ships a
`dataset/forecasting/FORECAST_META.csv` index that pre-registers these 12 file
names, so no additional setup is needed once the CSVs are in place.

### 3. Train and evaluate model

You can quickly train the model with the following script:

```shell
python ./scripts/run_benchmark.py --config-path "rolling_forecast_config.json" --data-name-list "DE.csv" --strategy-args '{"horizon": 24, "target_channel": [-1]}' --model-name "mixroute.MixRoute" --model-hyper-params '{"air_orth_weight": 0.0, "batch_size": 128, "d_ff": 384, "d_model": 128, "drop_path_rate": 0.2, "dropout": 0.0, "e_layers": 2, "horizon": 24, "loss": "MAE", "lr": 5e-05, "lradj": "constant", "n_heads": 8, "num_epochs": 100, "patch_len": 24, "patience": 30, "seq_len": 168, "weight_decay": 0.001}' --gpus 0 --num-workers 1 --timeout 60000 --save-path "DE/MixRoute"
```