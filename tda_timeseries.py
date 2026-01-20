# tda_timeseries_miniproject_py314.py
# Python 3.14 compatible: Baseline vs TDA features using ripser (no giotto-tda)

import os
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score, accuracy_score

from ripser import ripser

# ---------------- UCR local loader (NO sktime needed) ----------------
def load_ucr_local(dataset_dir, dataset_name="ECG200"):
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
    test_path  = next((p for p in test_candidates if os.path.exists(p)), None)

    if train_path is None or test_path is None:
        raise FileNotFoundError(
            f"Cannot find TRAIN/TEST files under: {dataset_dir}\n"
            f"Tried: {train_candidates + test_candidates}"
        )

    def _read(path):
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

# ---------------- Baseline feature extractor ----------------
class BasicTSFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    @staticmethod
    def _skew(x):
        m = x.mean()
        s = x.std() + 1e-12
        return np.mean(((x - m) / s) ** 3)

    @staticmethod
    def _kurt(x):
        m = x.mean()
        s = x.std() + 1e-12
        return np.mean(((x - m) / s) ** 4)

    @staticmethod
    def _autocorr1(x):
        if len(x) < 2:
            return 0.0
        x0 = x[:-1] - x[:-1].mean()
        x1 = x[1:] - x[1:].mean()
        denom = (np.sqrt((x0**2).sum()) * np.sqrt((x1**2).sum())) + 1e-12
        return float((x0 * x1).sum() / denom)

    @staticmethod
    def _fft_peak(x):
        f = np.fft.rfft(x)
        mag = np.abs(f)
        if len(mag) <= 1:
            return 0.0
        return float(np.max(mag[1:]))

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        feats = []
        for i in range(X.shape[0]):
            x = X[i]
            q = np.quantile(x, [0.05, 0.25, 0.5, 0.75, 0.95])
            feats.append([
                x.mean(), x.std(), self._skew(x), self._kurt(x),
                x.min(), x.max(),
                q[0], q[1], q[2], q[3], q[4],
                np.mean(x**2),
                self._autocorr1(x),
                self._fft_peak(x),
            ])
        return np.asarray(feats, dtype=float)

# ---------------- Utility: Takens embedding ----------------
def takens_embedding_1d(x, time_delay=2, dimension=5, stride=1):
    """
    x: (T,)
    returns point cloud: (n_points, dimension)
    """
    x = np.asarray(x, dtype=float)
    T = x.shape[0]
    max_start = T - (dimension - 1) * time_delay
    if max_start <= 1:
        # too short -> fallback: single point cloud of zeros
        return np.zeros((2, dimension), dtype=float)

    idx0 = np.arange(0, max_start, stride)
    pts = np.stack([x[idx0 + j * time_delay] for j in range(dimension)], axis=1)
    return pts

# ---------------- Utility: persistence featureization ----------------
def persistence_entropy(lifetimes, eps=1e-12):
    lifetimes = np.asarray(lifetimes, dtype=float)
    lifetimes = lifetimes[lifetimes > eps]
    if lifetimes.size == 0:
        return 0.0
    p = lifetimes / (lifetimes.sum() + eps)
    return float(-(p * np.log(p + eps)).sum())

def diag_to_features(diag, topk=10):
    """
    diag: array of (birth, death)
    returns a vector of fixed length:
      - topk lifetimes (sorted desc, padded)
      - sum, mean, std, max lifetime
      - persistence entropy
      - count of finite points
    """
    if diag is None or len(diag) == 0:
        lifetimes = np.array([], dtype=float)
    else:
        b = diag[:, 0]
        d = diag[:, 1]
        finite = np.isfinite(d)
        lifetimes = (d[finite] - b[finite]).astype(float)
        lifetimes = lifetimes[lifetimes > 0]

    lifetimes_sorted = np.sort(lifetimes)[::-1]
    top = np.zeros(topk, dtype=float)
    m = min(topk, lifetimes_sorted.size)
    if m > 0:
        top[:m] = lifetimes_sorted[:m]

    s = float(lifetimes.sum()) if lifetimes.size else 0.0
    mu = float(lifetimes.mean()) if lifetimes.size else 0.0
    sd = float(lifetimes.std()) if lifetimes.size else 0.0
    mx = float(lifetimes.max()) if lifetimes.size else 0.0
    pe = persistence_entropy(lifetimes)
    cnt = float(lifetimes.size)

    return np.concatenate([top, [s, mu, sd, mx, pe, cnt]])

# ---------------- TDA feature extractor (ripser) ----------------
class TDAFeaturesRipser(BaseEstimator, TransformerMixin):
    """
    Takens embedding -> ripser persistence -> vectorization (top-k lifetimes + stats + entropy)
    """
    def __init__(self, time_delay=2, dimension=5, stride=1, maxdim=1, topk=10):
        self.time_delay = time_delay
        self.dimension = dimension
        self.stride = stride
        self.maxdim = maxdim
        self.topk = topk

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        feats = []

        for i in range(X.shape[0]):
            x = X[i]
            pc = takens_embedding_1d(
                x,
                time_delay=self.time_delay,
                dimension=self.dimension,
                stride=self.stride
            )
            out = ripser(pc, maxdim=self.maxdim)
            dgms = out["dgms"]  # list: H0, H1, ...

            # concat features across dimensions 0..maxdim
            f_all = []
            for d in range(self.maxdim + 1):
                f_all.append(diag_to_features(dgms[d], topk=self.topk))
            feats.append(np.concatenate(f_all))

        return np.asarray(feats, dtype=float)

# ---------------- Multi-scale TDA feature extractor ----------------
class MultiScaleTDAFeaturesRipser(BaseEstimator, TransformerMixin):
    def __init__(self, configs):
        self.configs = configs

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        blocks = []
        for cfg in self.configs:
            blocks.append(TDAFeaturesRipser(**cfg).transform(X))
        return np.hstack(blocks)

def evaluate(X, y):
    scoring = {
        "acc": make_scorer(accuracy_score),
        "f1": make_scorer(f1_score, average="macro"),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)

    baseline_clf = Pipeline([
        ("feat", BasicTSFeatures()),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=800))
    ])

    tda_single = Pipeline([
        ("feat", TDAFeaturesRipser(time_delay=2, dimension=5, stride=1, maxdim=1, topk=10)),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=800))
    ])

    ms_configs = [
        dict(time_delay=1, dimension=4, stride=1, maxdim=1, topk=10),
        dict(time_delay=2, dimension=5, stride=1, maxdim=1, topk=10),
        dict(time_delay=3, dimension=6, stride=1, maxdim=1, topk=10),
    ]
    tda_multiscale = Pipeline([
        ("feat", MultiScaleTDAFeaturesRipser(configs=ms_configs)),
        ("scaler", StandardScaler()),
        ("clf", SVC(kernel="rbf", C=5.0, gamma="scale"))
    ])

    models = {
        "Baseline(LogReg)": baseline_clf,
        "TDA_single(LogReg)": tda_single,
        "TDA_multiscale(SVM)": tda_multiscale
    }

    rows = []
    for name, model in models.items():
        out = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1, return_train_score=False)
        rows.append({
            "model": name,
            "acc_mean": float(out["test_acc"].mean()),
            "acc_std": float(out["test_acc"].std()),
            "f1_mean": float(out["test_f1"].mean()),
            "f1_std": float(out["test_f1"].std()),
        })

    return pd.DataFrame(rows).sort_values(by="acc_mean", ascending=False)

if __name__ == "__main__":
    # Put your UCR files here, e.g.
    # C:\Users\MM\Desktop\1\data\ECG200\ECG200_TRAIN.tsv
    # C:\Users\MM\Desktop\1\data\ECG200\ECG200_TEST.tsv
    data_dir = r"C:\Users\MM\Desktop\1\data\ECG200"
    dataset = "ECG200"  # change to "ECG5000" once you downloaded that dataset

    print("Data dir listing:", os.listdir(data_dir))
    X, y_raw = load_ucr_local(data_dir, dataset)

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    print("X shape:", X.shape, "y shape:", y.shape, "classes:", le.classes_)

    df = evaluate(X, y)
    print("\n=== Results (Repeated 5-fold CV) ===")
    print(df.to_string(index=False))
    df.to_csv("results_tda_vs_baseline.csv", index=False)
    print("\nSaved: results_tda_vs_baseline.csv")
