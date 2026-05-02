# =============================================================================
# ensemble_voting_heart.py
# =============================================================================
# Soft-Voting Ensemble (XGBoost + Random Forest) for Heart Disease Prediction
# Dataset  : heart_statlog_cleveland_hungary_final.csv
# Optuna   : separate studies for RF and XGBoost  (80 trials each)
# Metric   : ROC-AUC (maximised) via 5-fold Stratified CV
# Output   : Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
#            + best hyperparameters for both models
#
# ── GOOGLE COLAB INSTRUCTIONS ────────────────────────────────────────────────
#  1. Run Cell 1  → installs required packages
#  2. Run Cell 2  → uploads your CSV when prompted  (or mount Drive — see note)
#  3. Run the remaining cells top-to-bottom
# ─────────────────────────────────────────────────────────────────────────────


# =============================================================================
# CELL 1 — Install dependencies  (run once per Colab session)
# =============================================================================
# !pip install -q optuna xgboost scikit-learn pandas matplotlib seaborn


# =============================================================================
# CELL 2 — Upload / locate the dataset
# =============================================================================
# ── Option A : Upload from your computer (default) ───────────────────────────
# from google.colab import files
# uploaded = files.upload()          # choose heart_statlog_cleveland_hungary_final.csv
# DATA_PATH = list(uploaded.keys())[0]

# ── Option B : Google Drive ───────────────────────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')
# DATA_PATH = '/content/drive/MyDrive/<YOUR_FOLDER>/heart_statlog_cleveland_hungary_final.csv'

# ── Option C : File already in /content (e.g. cloned repo) ───────────────────
DATA_PATH = "heart_statlog_cleveland_hungary_final.csv"   # ← adjust if needed


# =============================================================================
# CELL 3 — Imports
# =============================================================================
import os
import warnings
import numpy  as np
import pandas as pd

import optuna
import xgboost as xgb

from sklearn.model_selection import (
    train_test_split, StratifiedKFold, cross_val_score,
)
from sklearn.preprocessing   import StandardScaler, OneHotEncoder
from sklearn.compose         import ColumnTransformer
from sklearn.ensemble        import RandomForestClassifier, VotingClassifier
from sklearn.metrics         import (
    accuracy_score, recall_score, precision_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, ConfusionMatrixDisplay,
)

import matplotlib.pyplot as plt   # inline in Colab by default

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("[✓] All libraries imported successfully.")


# =============================================================================
# CELL 4 — Configuration
# =============================================================================
RANDOM_STATE  = 42
TARGET        = "target"    # 1 = Disease, 0 = Healthy
TEST_SIZE     = 0.20
CV_FOLDS      = 5
N_OPTUNA_RF   = 80          # trials for Random Forest study
N_OPTUNA_XGB  = 80          # trials for XGBoost study  (mirrors Analysis notebook)

np.random.seed(RANDOM_STATE)

# Feature schema (mirrors Analysis/analysis_between_rf&XG.ipynb)
CATEGORICAL_FEATURES = [
    "sex",                  # 0 = Female, 1 = Male
    "chest pain type",      # 1-4 nominal
    "fasting blood sugar",  # binary
    "resting ecg",          # 0-2 ordinal
    "exercise angina",      # binary
    "ST slope",             # 0-2 ordinal
]

NUMERICAL_FEATURES = [
    "age",
    "resting bp s",
    "cholesterol",
    "max heart rate",
    "oldpeak",
]

print("[✓] Configuration set.")


# =============================================================================
# CELL 5 — Load & split data
# =============================================================================
print("=" * 70)
print("  HEART DISEASE — Soft Voting Ensemble  (XGBoost + Random Forest)")
print("  Hyperparameter optimisation via Optuna  |  Metric: ROC-AUC")
print("=" * 70)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"\n[✗] Dataset not found: '{DATA_PATH}'\n"
        "    → Run Cell 2 to upload / point to the correct path."
    )

df = pd.read_csv(DATA_PATH)

print(f"\n[1/5] Dataset loaded")
print(f"      Shape         : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"      Missing values: {df.isnull().sum().sum()}")
print(f"      Target balance:\n"
      f"{df[TARGET].value_counts().rename({0:'No Disease', 1:'Disease'})}\n")

X_raw = df.drop(columns=[TARGET])
y     = df[TARGET]

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y,
    test_size    = TEST_SIZE,
    random_state = RANDOM_STATE,
    stratify     = y,
)

print(f"[2/5] Train/Test split (80 / 20 stratified)")
print(f"      Train : {X_train_raw.shape[0]}  |  Test : {X_test_raw.shape[0]}")
print(f"      Train balance → {dict(y_train.value_counts())}")
print(f"      Test  balance → {dict(y_test.value_counts())}\n")


# =============================================================================
# CELL 6 — Preprocessing  (StandardScaler + OneHotEncoder)
# =============================================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(),                                  NUMERICAL_FEATURES),
        ("cat", OneHotEncoder(drop="first", sparse_output=False),  CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

X_train_proc = preprocessor.fit_transform(X_train_raw)
X_test_proc  = preprocessor.transform(X_test_raw)

cat_names     = (preprocessor
                 .named_transformers_["cat"]
                 .get_feature_names_out(CATEGORICAL_FEATURES)
                 .tolist())
FEATURE_NAMES = NUMERICAL_FEATURES + cat_names

print(f"[✓] Preprocessing complete  |  {len(FEATURE_NAMES)} features")
print(f"    Features: {FEATURE_NAMES}\n")


# =============================================================================
# CELL 7 — Optuna: Random Forest  (80 trials, maximise ROC-AUC)
# =============================================================================
def rf_objective(trial):
    """
    Optimise RandomForestClassifier hyperparameters.
    Tuned params (per spec): n_estimators, max_depth,
                              min_samples_split, min_samples_leaf
    Extra: max_features, class_weight='balanced' (stabilises small dataset).
    """
    params = {
        "n_estimators"     : trial.suggest_int("n_estimators",       50, 500),
        "max_depth"        : trial.suggest_int("max_depth",           3,  20),
        "min_samples_split": trial.suggest_int("min_samples_split",   2,  20),
        "min_samples_leaf" : trial.suggest_int("min_samples_leaf",    1,  10),
        "max_features"     : trial.suggest_categorical("max_features",
                                                        ["sqrt", "log2"]),
        "class_weight"     : "balanced",
        "bootstrap"        : True,
        "random_state"     : RANDOM_STATE,
        "n_jobs"           : -1,
    }
    model  = RandomForestClassifier(**params)
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train_proc, y_train,
                             cv=skf, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


print("[3/5] Optuna – Random Forest hyperparameter search")
print(f"      Trials: {N_OPTUNA_RF}  |  CV folds: {CV_FOLDS}  |  Metric: ROC-AUC")

study_rf = optuna.create_study(
    direction  = "maximize",
    study_name = "random_forest_heart_hpo",
    sampler    = optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study_rf.optimize(rf_objective, n_trials=N_OPTUNA_RF, show_progress_bar=True)

# Merge search-space params with fixed params
best_rf_params = study_rf.best_params
best_rf_params.update({
    "class_weight": "balanced",
    "bootstrap"   : True,
    "random_state": RANDOM_STATE,
    "n_jobs"      : -1,
})

print(f"\n      [✓] Best CV ROC-AUC (RF) : {study_rf.best_value:.4f}")
print(f"      Best RF params          : {best_rf_params}\n")


# =============================================================================
# CELL 8 — Optuna: XGBoost  (80 trials, maximise ROC-AUC)
# =============================================================================
def xgb_objective(trial):
    """
    Optimise XGBClassifier hyperparameters.
    Tuned params (per spec): n_estimators, max_depth, learning_rate,
                              subsample, colsample_bytree
    Extra params from Analysis notebook: min_child_weight, gamma,
                                          reg_alpha, reg_lambda
    """
    params = {
        "objective"        : "binary:logistic",
        "eval_metric"      : "logloss",
        "random_state"     : RANDOM_STATE,
        "n_estimators"     : trial.suggest_int  ("n_estimators",      50,  500),
        "max_depth"        : trial.suggest_int  ("max_depth",          3,   10),
        "learning_rate"    : trial.suggest_float("learning_rate",   0.01,  0.3,
                                                  log=True),
        "subsample"        : trial.suggest_float("subsample",        0.5,  1.0),
        "colsample_bytree" : trial.suggest_float("colsample_bytree", 0.5,  1.0),
        "min_child_weight" : trial.suggest_int  ("min_child_weight",   1,   10),
        "gamma"            : trial.suggest_float("gamma",             0,    5),
        "reg_alpha"        : trial.suggest_float("reg_alpha",         0,    1),
        "reg_lambda"       : trial.suggest_float("reg_lambda",       0.5,   5),
    }
    model  = xgb.XGBClassifier(**params, verbosity=0)
    skf    = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True,
                              random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train_proc, y_train,
                             cv=skf, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


print("[4/5] Optuna – XGBoost hyperparameter search")
print(f"      Trials: {N_OPTUNA_XGB}  |  CV folds: {CV_FOLDS}  |  Metric: ROC-AUC")

study_xgb = optuna.create_study(
    direction  = "maximize",
    study_name = "xgboost_heart_hpo",
    sampler    = optuna.samplers.TPESampler(seed=RANDOM_STATE),
)
study_xgb.optimize(xgb_objective, n_trials=N_OPTUNA_XGB, show_progress_bar=True)

best_xgb_params = study_xgb.best_params
best_xgb_params.update({
    "objective"   : "binary:logistic",
    "eval_metric" : "logloss",
    "random_state": RANDOM_STATE,
})

print(f"\n      [✓] Best CV ROC-AUC (XGB): {study_xgb.best_value:.4f}")
print(f"      Best XGB params          : {best_xgb_params}\n")


# =============================================================================
# CELL 9 — Build & train the Soft-Voting Ensemble
# =============================================================================
rf_opt  = RandomForestClassifier(**best_rf_params)
xgb_opt = xgb.XGBClassifier(**best_xgb_params, verbosity=0)

ensemble = VotingClassifier(
    estimators=[
        ("rf",  rf_opt),
        ("xgb", xgb_opt),
    ],
    voting="soft",      # averages predicted probabilities from both models
    n_jobs=-1,
)

print("[5/5] Training Soft-Voting Ensemble on full training set …")
ensemble.fit(X_train_proc, y_train)
print("      [✓] Ensemble trained.\n")


# =============================================================================
# CELL 10 — Evaluate on the held-out test set
# =============================================================================
y_pred = ensemble.predict(X_test_proc)
y_prob = ensemble.predict_proba(X_test_proc)[:, 1]

acc  = accuracy_score (y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score   (y_test, y_pred)
f1   = f1_score       (y_test, y_pred)
auc  = roc_auc_score  (y_test, y_prob)
cm   = confusion_matrix(y_test, y_pred)

print("=" * 60)
print("  SOFT-VOTING ENSEMBLE — EVALUATION RESULTS")
print("  (XGBoost-Optuna  +  RandomForest-Optuna,  voting='soft')")
print("=" * 60)
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall*   : {rec:.4f}   ← critical for medical screening")
print(f"  F1-Score  : {f1:.4f}")
print(f"  ROC-AUC   : {auc:.4f}")
print()
print("  Confusion Matrix:")
print(f"    {cm}")
print()
print("  Classification Report:")
print(classification_report(y_test, y_pred,
                             target_names=["No Disease (0)", "Heart Disease (1)"]))
print("=" * 60)


# =============================================================================
# CELL 11 — Confusion-matrix plot  (displayed inline in Colab)
# =============================================================================
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(
    confusion_matrix = cm,
    display_labels   = ["No Disease", "Disease"],
).plot(ax=ax, colorbar=False, cmap="Blues")
ax.set_title("Soft-Voting Ensemble\nConfusion Matrix",
             fontsize=12, fontweight="bold")
ax.grid(False)
plt.tight_layout()
plt.savefig("ensemble_confusion_matrix.png", bbox_inches="tight", dpi=130)
plt.show()
print("[✓] Plot saved → ensemble_confusion_matrix.png")


# =============================================================================
# CELL 12 — Best hyperparameters (copy-paste ready)
# =============================================================================
sep = "-" * 60
print(sep)
print("  BEST HYPERPARAMETERS  (copy-paste ready)")
print(sep)

print("\n  ── Random Forest ──")
for k, v in best_rf_params.items():
    print(f"    {k:25s}: {v}")

print("\n  ── XGBoost ──")
for k, v in best_xgb_params.items():
    print(f"    {k:25s}: {v}")

print(sep)
