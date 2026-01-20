# TDA Time-Series Mini-Project (UCR: ECG5000 + GunPoint, Python 3.14, Local)

A reproducible mini-project comparing **hand-crafted baseline features** vs **topology-aware (TDA) features** for time-series classification on two UCR datasets (**ECG5000**, **GunPoint**) under **Repeated Stratified 5-fold CV (5×3 = 15 folds)**. Runs on **Python 3.14 (Windows)** using **local UCR files** (no `sktime`, no `giotto-tda`).

---

## What’s inside (3 pipelines)

1. **Baseline(LogReg)**  
   Basic time-series features (mean/std/quantiles/energy/autocorr/FFT-peak) → StandardScaler → LogisticRegression

2. **TDA_single(LogReg)**  
   Takens/sliding-window embedding → persistence (H0/H1 via `ripser`) → diagram vectorization (fixed-length stats) → StandardScaler → LogisticRegression

3. **TDA_multiscale(SVM)**  
   Multi-scale TDA features by concatenating multiple embedding configs → StandardScaler → RBF-SVM

---

## Repo structure

├─ run_tda_ucr_miniproject_local.py
├─ tda_timeseries.py
├─ data/
│ ├─ ECG5000/
│ │ ├─ ECG5000_TRAIN.tsv
│ │ ├─ ECG5000_TEST.tsv
│ │ └─ README.md
│ └─ GunPoint/
│ ├─ GunPoint_TRAIN.tsv
│ ├─ GunPoint_TEST.tsv
│ └─ README.md
└─ outputs/
├─ ECG5000/
│ ├─ summary.csv
│ ├─ fold_scores.csv
│ └─ config.json
└─ GunPoint/
├─ summary.csv
├─ fold_scores.csv
└─ config.json

Data (UCR Archive)

Download ECG5000 and GunPoint from the UCR Time Series Archive (2018), extract, and place:

ECG5000_TRAIN.tsv, ECG5000_TEST.tsv into data/ECG5000/

GunPoint_TRAIN.tsv, GunPoint_TEST.tsv into data/GunPoint/

Results
ECG5000

TDA_multiscale(SVM): acc 0.937333 (std 0.005538), macro-F1 0.553186 (std 0.042184)

Baseline(LogReg): acc 0.928000 (std 0.005542), macro-F1 0.493192 (std 0.034160)

TDA_single(LogReg): acc 0.919400 (std 0.007500), macro-F1 0.435702 (std 0.034074)

GunPoint

TDA_multiscale(SVM): acc 0.961667 (std 0.029681), macro-F1 0.961622 (std 0.029731)

TDA_single(LogReg): acc 0.936667 (std 0.036433), macro-F1 0.936548 (std 0.036536)

Baseline(LogReg): acc 0.913333 (std 0.039940), macro-F1 0.913132 (std 0.040019)
