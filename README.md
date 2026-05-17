# CAST-Norm

This is an official implementation of paper: [CAST-Norm: Coupled Adaptive Spatio-Temporal Normalization for Multivariate Time Series Forecasting](KDD 2026). CAST-Norm is a normalization framework for time series forecasting that addresses non-stationarity through spatial-temporal coupling perception and community-aware spatial purification.

## Modules
- **Temporal Normalization**: Normalizes time series along the temporal dimension
- **Spatial-Temporal Coupling Perception (STCP)**: Captures dynamic spatial-temporal relationships via patch-wise graph learning
- **Community-Aware Spatial Purification (CASD)**: Separates invariant and variant patterns using community detection
- **Coupling-Aware Recalibration**: Restores forecasts to original scale using learned coupling relationships

## Installation

```bash
pip install torch numpy pandas scikit-learn
```

For S_Mamba model:

```bash
pip install mamba-ssm
```

## Quick Start

### Training

```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --model_id ETTh1_informer_castnorm_96_96 \
    --model Informer \
    --data ETTh1 \
    --root_path ./data/ETT/ \
    --data_path ETTh1.csv \
    --features M \
    --freq h \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --d_ff 2048 \
    --dropout 0.1 \
    --norm_type CAST-Norm \
    --cast_norm_denorm recalib \
    --enable_stcp \
    --enable_casd \
    --lambda1 1.0 \
    --lambda2 1.0 \
    --train_epochs 10 \
    --batch_size 32 \
    --learning_rate 0.0001
```

### Testing

```bash
python run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --model_id ETTh1_informer_castnorm_96_96 \
    --model Informer \
    --data ETTh1 \
    --root_path ./data/ETT/ \
    --data_path ETTh1.csv \
    --features M \
    --freq h \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --enc_in 7 \
    --dec_in 7 \
    --c_out 7 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --d_layers 1 \
    --d_ff 2048 \
    --dropout 0.1 \
    --norm_type CAST-Norm
```

## Supported Models

- **Informer**: Transformer-based model with ProbSparse attention
- **S_Mamba**: Mamba-based model with state space mechanisms
- **DLinear**: Linear model with series decomposition
- **TCN**: Temporal Convolutional Network

## Dataset

Place your dataset CSV files in the `./data/` directory. The framework supports:

- ETT datasets (ETTh1, ETTh2, ETTm1, ETTm2)
- Custom datasets (use `--data custom`)

## Key Parameters

- `--norm_type`: Normalization type (`none` or `CAST-Norm`)
- `--cast_norm_denorm`: Denormalization method (`recalib`, `simple`, or `none`)
- `--enable_stcp`: Enable Spatial-Temporal Coupling Perception module
- `--enable_casd`: Enable Community-Aware Spatial Purification module
- `--lambda1`: Weight for mincut and orthogonality losses
- `--lambda2`: Weight for consistency loss


## Acknowledgement

We appreciate the following github repos a lot for their valuable code base or datasets:

https://github.com/thuml/Time-Series-Library

https://github.com/zhouhaoyi/Informer2020

https://github.com/wzhwzhwzh0921/S-D-Mamba

https://github.com/cure-lab/LTSF-Linear

https://github.com/luodhhh/ModernTCN
