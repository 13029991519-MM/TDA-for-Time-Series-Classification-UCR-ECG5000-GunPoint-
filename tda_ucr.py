# run_tda_ucr_miniproject_local.py
# One-click mini-project: Baseline vs (single/multi-scale) TDA features on LOCAL UCR datasets
# Python 3.14-friendly: uses ripser for persistent homology; no aeon/sktime required
# Outputs:
#   outputs/<dataset>/summary.csv
#   outputs/<dataset>/fold_scores.csv
#   outputs/<dataset>/config.json

from __future__ import annotations

import os
import json
import math
import time
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, f1_score


# ---------- Local UCR loader ----------
def load_ucr_local(dataset_dir: str, dataset_name: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads UCR files from local directory.
    Expected files:
      <dataset_name>_TRAIN.tsv and <dataset_name>_TEST.tsv
    or .txt / whitespace-separated variants.

    Returns:
      X: (n_samples, n_timestamps) float
      y: (n_samples,) labels (string)
    """
    train_candidates = [
        os.path.join(dataset_dir, f"{dataset_name}_TRAIN.tsv"),
        os.path.join(dataset_dir, f"{dataset_name}_TRAIN.txt"),
        os.path.join(dataset_dir, f"{dataset_name}_TRAIN"),
    ]
    test_candidates = [
        os.path.join(dataset_dir, f"{dataset_name}_TEST.tsv"),
        os.path.join(dataset_dir, f"{dataset_name}_TEST.txt"),
        os.path.join(dataset_dir, f"{dataset_name}_TEST"),
    ]

    train_path = next((p for p in train_candidates if os.path.exists(p)), None)
    test_path = next((p for p in test_candidates if os.path.exists(p)), None)

    if train_path is None or test_path is None:
        listing = os.listdir(dataset_dir) if os.path.isdir(dataset_dir) else []
        raise FileNotFoundError(
            f"[{dataset_name}] Cannot find TRAIN/TEST under: {dataset_dir}\n"
            f"Tried: {train_candidates + test_candidates}\n"
            f"Dir listing: {listing}"
        )

    def _read(path: str) -> pd.DataFrame:
        # Try TSV, fallback to whitespace
        try:
            df = pd.read_csv(path, sep="\t", header=None)
            if df.shape[1] <= 2:
                raise ValueError("TSV parse produced too few columns")
        except Exception:
            df = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        return df

    train_df = _read(train_path)
    test_df = _read(test_path)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)

    y = df.iloc[:, 0].astype(str).to_numpy()
    X = df.iloc[:, 1:].astype(float).to_numpy()
    return X, y


# ---------- Baseline features ----------
class BasicTSFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    @staticmethod
    def _skew(x: np.ndarray) -> float:
        m = x.mean()
        s = x.std() + 1e-12
        return float(np.mean(((x - m) / s) ** 3))

    @staticmethod
    def _kurt(x: np.ndarray) -> float:
        m = x.mean()
        s = x.std() + 1e-12
        return float(np.mean(((x - m) / s) ** 4))

    @staticmethod
    def _autocorr1(x: np.ndarray) -> float:
        if x.size < 2:
            return 0.0
        x0 = x[:-1] - x[:-1].mean()
        x1 = x[1:] - x[1:].mean()
        denom = (np.sqrt((x0**2).sum()) * np.sqrt((x1**2).sum())) + 1e-12
        return float((x0 * x1).sum() / denom)

    @staticmethod
    def _fft_peak(x: np.ndarray) -> float:
        f = np.fft.rfft(x)
        mag = np.abs(f)
        if mag.size <= 1:
            return 0.0
        return float(np.max(mag[1:]))

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        feats = np.empty((X.shape[0], 14), dtype=np.float64)
        for i in range(X.shape[0]):
            x = X[i]
            q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
            feats[i, :] = [
                x.mean(), x.std(), self._skew(x), self._kurt(x),
                x.min(), x.max(),
                q[0], q[1], q[2], q[3], q[4],
                np.mean(x**2),
                self._autocorr1(x),
                self._fft_peak(x),
            ]
        return feats


# ---------- TDA via ripser ----------
def takens_embedding_1d(x: np.ndarray, dim: int, tau: int, stride: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    T = x.size
    max_start = T - (dim - 1) * tau
    if max_start <= 1:
        return np.empty((0, dim), dtype=np.float64)
    idx0 = np.arange(0, max_start, stride, dtype=int)
    emb = np.empty((idx0.size, dim), dtype=np.float64)
    for j in range(dim):
        emb[:, j] = x[idx0 + j * tau]
    return emb


def persistence_summary(diagrams: List[np.ndarray], maxdim: int = 1) -> np.ndarray:
    feats = []
    for d in range(maxdim + 1):
        D = diagrams[d] if d < len(diagrams) else np.empty((0, 2), dtype=np.float64)
        if D.size == 0:
            feats.extend([0.0] * 6)
            continue
        birth = D[:, 0]
        death = D[:, 1]
        finite = np.isfinite(death)
        birth = birth[finite]
        death = death[finite]
        if birth.size == 0:
            feats.extend([0.0] * 6)
            continue
        life = np.maximum(death - birth, 0.0)
        cnt = float(life.size)
        s = float(life.sum())
        mu = float(life.mean())
        sd = float(life.std())
        mx = float(life.max())
        p = life / (life.sum() + 1e-12)
        ent = float(-(p * np.log(p + 1e-12)).sum())
        feats.extend([cnt, s, mu, sd, mx, ent])
    return np.asarray(feats, dtype=np.float64)


class TDAFeaturesRipser(BaseEstimator, TransformerMixin):
    def __init__(self, dim: int = 5, tau: int = 2, stride: int = 3, maxdim: int = 1):
        self.dim = dim
        self.tau = tau
        self.stride = stride
        self.maxdim = maxdim

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        from ripser import ripser
        X = np.asarray(X, dtype=np.float64)
        out = []
        for i in range(X.shape[0]):
            emb = takens_embedding_1d(X[i], dim=self.dim, tau=self.tau, stride=self.stride)
            if emb.shape[0] < self.dim + 2:
                out.append(np.zeros((self.maxdim + 1) * 6, dtype=np.float64))
                continue
            dgms = ripser(emb, maxdim=self.maxdim)["dgms"]
            out.append(persistence_summary(dgms, maxdim=self.maxdim))
        return np.vstack(out)


class MultiScaleTDAFeaturesRipser(BaseEstimator, TransformerMixin):
    def __init__(self, configs: List[Dict[str, int]], maxdim: int = 1):
        self.configs = configs
        self.maxdim = maxdim
        self.models: List[TDAFeaturesRipser] = []

    def fit(self, X, y=None):
        self.models = [
            TDAFeaturesRipser(dim=c["dim"], tau=c["tau"], stride=c["stride"], maxdim=self.maxdim)
            for c in self.configs
        ]
        return self

    def transform(self, X):
        return np.hstack([m.transform(X) for m in self.models])


# ---------- Evaluation ----------
@dataclass
class EvalConfig:
    dataset: str
    dataset_dir: str
    out_dir: str = "outputs"
    seed: int = 42
    n_splits: int = 5
    n_repeats: int = 3
    tda_single: Dict[str, int] = None
    tda_multiscale: List[Dict[str, int]] = None
    svm_C: float = 5.0

    def __post_init__(self):
        if self.tda_single is None:
            self.tda_single = {"dim": 5, "tau": 2, "stride": 3}
        if self.tda_multiscale is None:
            self.tda_multiscale = [
                {"dim": 4, "tau": 1, "stride": 3},
                {"dim": 5, "tau": 2, "stride": 3},
                {"dim": 6, "tau": 3, "stride": 3},
            ]


def standard_error(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    if x.size <= 1:
        return 0.0
    return float(x.std(ddof=1) / math.sqrt(x.size))


def run_one_dataset(cfg: EvalConfig):
    t0 = time.time()
    print(f"\n=== Loading LOCAL dataset: {cfg.dataset} ===")
    print(f"Data dir listing: {os.listdir(cfg.dataset_dir)}")

    X, y = load_ucr_local(cfg.dataset_dir, cfg.dataset)
    print(f"X shape: {X.shape}  y shape: {y.shape}  classes: {np.unique(y)}")

    scoring = {
        "acc": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score, average="macro"),
    }
    cv = RepeatedStratifiedKFold(
        n_splits=cfg.n_splits, n_repeats=cfg.n_repeats, random_state=cfg.seed
    )

    baseline = Pipeline([
        ("feat", BasicTSFeatures()),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=800)),
    ])

    tda_single = Pipeline([
        ("feat", TDAFeaturesRipser(**cfg.tda_single, maxdim=1)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=800)),
    ])

    tda_multi = Pipeline([
        ("feat", MultiScaleTDAFeaturesRipser(configs=cfg.tda_multiscale, maxdim=1)),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=cfg.svm_C, gamma="scale")),
    ])

    models = {
        "Baseline(LogReg)": baseline,
        "TDA_single(LogReg)": tda_single,
        "TDA_multiscale(SVM)": tda_multi,
    }

    fold_rows = []
    summary_rows = []

    for name, model in models.items():
        print(f"\n--- CV running: {name} ---")
        out = cross_validate(
            model, X, y, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False
        )
        acc = out["test_acc"]
        f1 = out["test_f1"]

        for k in range(acc.size):
            fold_rows.append({
                "dataset": cfg.dataset,
                "model": name,
                "fold_id": k,
                "acc": float(acc[k]),
                "f1_macro": float(f1[k]),
            })

        summary_rows.append({
            "dataset": cfg.dataset,
            "model": name,
            "acc_mean": float(np.mean(acc)),
            "acc_std": float(np.std(acc, ddof=1)),
            "acc_se": standard_error(acc),
            "f1_mean": float(np.mean(f1)),
            "f1_std": float(np.std(f1, ddof=1)),
            "f1_se": standard_error(f1),
            "n_folds_total": int(acc.size),
        })

    summary = pd.DataFrame(summary_rows).sort_values(by="acc_mean", ascending=False)
    folds = pd.DataFrame(fold_rows)

    out_dir = os.path.join(cfg.out_dir, cfg.dataset)
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "summary.csv")
    folds_path = os.path.join(out_dir, "fold_scores.csv")
    cfg_path = os.path.join(out_dir, "config.json")

    summary.to_csv(summary_path, index=False)
    folds.to_csv(folds_path, index=False)
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved:\n  {summary_path}\n  {folds_path}\n  {cfg_path}")
    print(f"Runtime: {time.time() - t0:.1f}s")


def main():
    # >>> Set your local paths here <<<
    ecg_dir = r"C:\Users\MM\Desktop\1\data\ECG5000"
    gun_dir = r"C:\Users\MM\Desktop\1\data\GunPoint"

    configs = [
        EvalConfig(dataset="ECG5000", dataset_dir=ecg_dir),
        EvalConfig(dataset="GunPoint", dataset_dir=gun_dir),
    ]

    for cfg in configs:
        run_one_dataset(cfg)


if __name__ == "__main__":
    main()
