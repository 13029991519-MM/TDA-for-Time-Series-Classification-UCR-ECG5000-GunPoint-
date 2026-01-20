TDA for Time-Series Classification (UCR ECG5000 + GunPoint)

This mini-project compares hand-crafted time-series features vs topology-aware (TDA) features for classification on two UCR datasets: ECG5000 and GunPoint. The pipeline is designed to be reproducible, audit-friendly, and runnable on Python 3.14 (Windows) with local UCR files.

1. What this project does
Goal

Evaluate whether geometric/topological structure extracted from time-delay embeddings improves time-series classification beyond simple statistical/FFT features.

Three model pipelines (feature → scaler → classifier)

Baseline(LogReg): hand-crafted features + standardization + logistic regression

TDA_single(LogReg): Takens embedding → persistent homology → TDA vectorization + standardization + logistic regression

TDA_multiscale(SVM): multi-scale Takens embedding configs → PH features concatenation + standardization + RBF-SVM

These three pipelines match the executed configuration in the experiment report. 
2. Method overview
2.1 Local UCR loader (no sktime required)

Reads <dataset>_TRAIN.tsv and <dataset>_TEST.tsv (also supports whitespace .txt variants).

Concatenates TRAIN+TEST for a unified dataset.

Parses: y = first column, X = remaining columns. 
2.2 Baseline features

A compact set of reproducible statistics + quantiles + FFT peak magnitude (hand-crafted features).

2.3 TDA features

Takens / sliding-window embedding to convert each 1D series into a point cloud

Vietoris–Rips persistence diagrams in dimensions H0 and H1 (computed via ripser)

Diagram vectorization via fixed-length summary statistics (counts, lifetime sums, mean/std/max, entropy), concatenated across homology dimensions

Multi-scale: concatenate TDA features computed from multiple (dimension d, delay τ, stride s) configurations.
Executed settings:

Single-scale: (d, τ, s) = (5, 2, 3)

Multi-scale set: {(4,1,3), (5,2,3), (6,3,3)}

PH max dimension: maxdim=1
3. Evaluation protocol (robust + uncertainty-aware)

Cross-validation: RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42) → 15 test folds

Metrics:

Accuracy

Macro-F1 (uniform average across classes; sensitive to minority-class performance)

Uncertainty reporting: mean, std, and standard error (SE = std / sqrt(n), n=15)

All fold-level scores are exported for audit / statistical tests.

4. Data: where to get ECG5000 / GunPoint

Both datasets are from the UCR Time Series Archive (2018 version). Download the dataset ZIP from the archive page, then place the extracted *_TRAIN.tsv and *_TEST.tsv files under data/<DatasetName>/.

5. Outputs (artifacts)

For each dataset, the script saves:

outputs/<dataset>/summary.csv — table-level metrics (mean/std/SE)

outputs/<dataset>/fold_scores.csv — fold-level scores (audit + statistical tests)

outputs/<dataset>/config.json — exact experiment config (CV seed/splits/repeats; embedding configs)

6. Results (exact executed outputs)
6.1 ECG5000 (5 classes; N=5000, T=140)
Model	Acc (mean ± std)	Macro-F1 (mean ± std)
TDA_multiscale(SVM)	0.937333 ± 0.005538	0.553186 ± 0.042184
Baseline(LogReg)	0.928000 ± 0.005542	0.493192 ± 0.034160
TDA_single(LogReg)	0.919400 ± 0.007500	0.435702 ± 0.034074

Interpretation: multi-scale TDA improves mean accuracy vs baseline, suggesting delay-embedded geometric/topological structure is informative beyond hand-crafted moments; macro-F1 is notably lower than accuracy, consistent with class imbalance / minority-class difficulty. 

6.2 GunPoint (2 classes; N=200, T=150)
Model	Acc (mean ± std)	Macro-F1 (mean ± std)
TDA_multiscale(SVM)	0.961667 ± 0.029681	0.961622 ± 0.029731
TDA_single(LogReg)	0.936667 ± 0.036433	0.936548 ± 0.036536
Baseline(LogReg)	0.913333 ± 0.039940	0.913132 ± 0.040019

Interpretation: topology-aware features yield clear gains under the same CV protocol; because the task is binary, accuracy and macro-F1 align closely.

7. References

Ripser (Vietoris–Rips persistence): Ulrich Bauer, 2021. 
UCR Time Series Archive, 2018. 
Sliding windows + persistence for signals: Perea & Harer, 2015. 
Takens embedding: Takens, 1981. 
