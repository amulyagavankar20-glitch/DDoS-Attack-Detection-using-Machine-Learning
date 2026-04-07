# DDoS Attack Detection Pipeline

This project implements an end-to-end ML pipeline for multiclass DDoS traffic classification using tabular network-flow features.
It covers data preparation, preprocessing, feature reduction, class balancing with SMOTE, model training, evaluation, and artifact generation.

## 1. End-To-End Pipeline Overview

```
Class-wise raw parquet files
    -> merged_training.parquet + merged_testing.parquet (dataset preparation stage)
    -> merged_training.csv + merged_testing.csv (pipeline input files)
    -> load_and_merge()
    -> split_data() [label encoding + train/test split + feature filtering]
    -> apply_smote() on training split
    -> model training (RF, XGB, LGBM, NB, KNN)
    -> save models (.pkl)
    -> evaluate() metrics + confusion matrices
    -> write outputs/results.csv + feature importance plots
```

## 2. Dataset Lineage And Preparation

The `data` folder contains many attack-wise parquet files such as LDAP, MSSQL, NetBIOS, UDP, UDPLag, Syn, NTP, SNMP, TFTP, DNS, etc.

Dataset flow used in this workspace:

1. Multiple class-wise parquet files were consolidated into:
   - `data/merged_training.parquet`
   - `data/merged_testing.parquet`
2. These merged parquet files were exported to CSV:
   - `data/merged_training.csv`
   - `data/merged_testing.csv`
3. The training pipeline currently reads the merged CSV files through `load_and_merge()` in `src/data_loader.py`.

During loading:

- Train and test merged CSVs are concatenated.
- Label column is normalized to `label`.
- Selected label names are mapped (for example `DrDoS_UDP -> UDP`, `UDP-lag -> UDPLag`).

## 3. Dataset Profile (Current Workspace)

Measured from the current code/data state:

- Total rows after merge: `431,371`
- Number of classes: `13`
- Class labels:
  - Benign
  - DrDoS_DNS
  - DrDoS_NTP
  - DrDoS_SNMP
  - LDAP
  - MSSQL
  - NetBIOS
  - Portmap
  - Syn
  - TFTP
  - UDP
  - UDPLag
  - WebDDoS

## 4. Feature Engineering And Feature Count Reduction

Feature handling happens in `split_data()` and `src/feature_engineering.py`.

### Steps

1. Start from raw feature matrix (drop `label`).
2. Remove low-variance features with threshold `1e-5`.
3. Remove highly correlated features with threshold `0.9`.
4. Apply the same selected columns to test split.

### Actual counts from this project run

- Raw input features: `77`
- After low-variance filter: `65`
- Dropped by low-variance: `12`
- After high-correlation filter: `38`
- Dropped by high-correlation: `27`
- Total dropped: `39`
- Final retained features used by all models: `38`

The retained feature names are written to `outputs/x_features.csv`, and target name is saved to `outputs/y_feature.txt`.

## 5. Imbalanced Data To Balanced Data (SMOTE)

Class imbalance is handled by `apply_smote()` in `src/smote.py` using `SMOTE(random_state=42)`.

SMOTE is applied only on training data (never on test data).

### Current imbalance statistics

- Classes: `13`
- Training class count before SMOTE (min/max): `41 / 97,094`
- Training class count after SMOTE (min/max): `97,094 / 97,094`
- Training shape before SMOTE: `(345,096, 38)`
- Training shape after SMOTE: `(1,262,222, 38)`

This gives each class equal representation in training.

## 6. Algorithms Used And Why

### 6.1 Random Forest (RF)

File: `models/random_forest.py`

- Ensemble of decision trees with bagging.
- Good baseline for tabular features and nonlinear boundaries.
- Current setup uses optimized fixed parameters (no hyperparameter search in main pipeline):
  - `n_estimators`: 300 (laptop-safe) / 500 (full)
  - `max_depth`: 15 (laptop-safe) / 20 (full)
  - `class_weight='balanced'`, `max_features='sqrt'`

### 6.2 K-Nearest Neighbors (KNN)

File: `models/knn.py`

- Distance-based classifier.
- Often strong accuracy but can be expensive at inference time.
- Uses `n_neighbors=5` in laptop-safe mode (`10` otherwise).

### 6.3 XGBoost (XGB)

File: `models/xgboost_model.py`

- Gradient-boosted trees, typically very strong on tabular data.
- Uses `tree_method='hist'` and CUDA device when available.
- Includes early stopping on validation split:
  - `early_stopping_rounds`: 15 (laptop-safe) / 40 (full)

### 6.4 LightGBM (LGBM)

File: `models/lightgbm_model.py`

- Leaf-wise gradient boosting model optimized for speed.
- Uses validation split and LightGBM early stopping callback.
- Current trainer does not force `device_type='gpu'` directly in this file.

### 6.5 Naive Bayes (GaussianNB)

File: `models/naive_bayes.py`

- Probabilistic baseline with Gaussian assumption per feature.
- Now hyper-tuned using `GridSearchCV` on `var_smoothing`:
  - laptop-safe grid: `10^-12` to `10^-6`
  - full grid: `10^-12` to `10^-3`
- Scoring metric for search: weighted F1.

### 6.6 SMOTE

File: `src/smote.py`

- Synthetic Minority Oversampling Technique.
- Generates synthetic minority class points to reduce class imbalance.
- Helps classifiers avoid bias toward majority classes.

## 7. Key Functions Used In The Pipeline

### Data/Preprocessing layer

- `load_and_merge()` in `src/data_loader.py`
  - Loads merged CSVs, concatenates, normalizes labels.
- `split_data()` in `src/preprocessing.py`
  - Encodes labels, stratified split, feature filtering, and saves processed train/test as parquet+csv.
- `remove_low_variance()` and `remove_high_corr()` in `src/feature_engineering.py`
  - Feature reduction utilities.
- `apply_smote()` in `src/smote.py`
  - Balances class distribution on training set.

### Model layer

- `train_rf()`
- `train_xgb()`
- `train_lgb()`
- `train_naive_bayes()`
- `train_knn()`

### Evaluation/Output layer

- `evaluate()` in `evaluation/compare_models.py`
  - Predicts on test data, computes metrics, saves confusion matrices, writes `outputs/results.csv`.
- `plot_feature_importance()` in `evaluation/feature_importance.py`
  - Creates top feature-importance plot when model supports it.
- `save_model()` in `src/utils.py`
  - Persists trained models to `outputs/models/*.pkl`.

## 8. How The Main Pipeline Works (`main.py`)

1. Read `LAPTOP_SAFE_MODE` environment flag (default enabled).
2. Load merged dataset via `load_and_merge()`.
3. Preprocess via `split_data()`.
4. Save feature metadata (`x_features.csv`, `y_feature.txt`).
5. Balance training set with SMOTE.
6. Train all configured models sequentially with timing logs.
7. Save all models to `outputs/models`.
8. Evaluate all models and overwrite `outputs/results.csv`.
9. Save feature-importance plots for supported models.

## 9. Current Results (From `outputs/results.csv`)

| Model | Accuracy | Precision | Recall | F1 Score | Time (s) |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.974686 | 0.976256 | 0.974686 | 0.975171 | 1.349263 |
| XGBoost GPU | 0.975080 | 0.976914 | 0.975080 | 0.975743 | 0.291308 |
| LightGBM GPU | 0.975335 | 0.977008 | 0.975335 | 0.975898 | 1.631033 |
| Naive Bayes | 0.275677 | 0.631200 | 0.275677 | 0.337657 | 0.619264 |
| KNN | 0.937201 | 0.949564 | 0.937201 | 0.942376 | 79.512510 |

Observations:

- Best weighted F1 in current run: LightGBM (very close to XGBoost and RF).
- KNN remains accurate but has very high inference time.
- Naive Bayes improved after hyper-tuning but is still far behind tree ensembles.

## 10. Running Scripts

### Full training/evaluation for all models

`python main.py`

### Retrain only Naive Bayes and update only its row in results

`python train_naive_bayes_only.py`

### GPU availability checks

`python gpu_access_test.py`

## 11. Important Output Artifacts

- `outputs/results.csv`: final metrics table.
- `outputs/models/*.pkl`: serialized models.
- `outputs/plots/confusion_matrices/*_confusion_matrix.png`: per-model confusion matrices.
- `outputs/plots/*_feature_importance.png`: model feature importance plots.
- `data/processed/train.csv`, `test.csv`, `train.parquet`, `test.parquet`: processed splits.

## 12. Environment Setup

1. `python -m venv venv`
2. PowerShell: `venv\Scripts\Activate.ps1`
3. `pip install -r requirements.txt`

Main dependencies include: pandas, numpy, scikit-learn, xgboost, lightgbm, imbalanced-learn, matplotlib, seaborn, joblib, pyarrow, torch.