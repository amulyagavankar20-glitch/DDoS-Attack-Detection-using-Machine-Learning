import os
import time

import pandas as pd

from src.data_loader import load_and_merge
from src.preprocessing import split_data
from src.smote import apply_smote
from src.utils import save_model

from models.random_forest import train_rf
from models.xgboost_model import train_xgb
from models.lightgbm_model import train_lgb
from models.naive_bayes import train_naive_bayes
from models.knn import train_knn

from evaluation.feature_importance import plot_feature_importance
from evaluation.compare_models import evaluate

# Laptop-safe mode reduces training pressure by lowering search space and threads.
LAPTOP_SAFE_MODE = os.getenv("LAPTOP_SAFE_MODE", "1") == "1"

# Load and merge the raw train/test source CSV files.
df = load_and_merge()

# Split and preprocess once so every model receives the same feature space.
(X_train, X_test, y_train, y_test), le, prep_info = split_data(df)
print(f"Feature engineering complete: {prep_info['n_features']} features retained")
print(f"Target feature (y): {prep_info['target_feature']}")
print(f"Input features (X) count: {len(prep_info['feature_names'])}")
print(f"Input features (X): {prep_info['feature_names']}")

output_dir = os.path.join("outputs")
os.makedirs(output_dir, exist_ok=True)

pd.DataFrame({"feature": prep_info["feature_names"]}).to_csv(
    os.path.join(output_dir, "x_features.csv"),
    index=False,
)

with open(os.path.join(output_dir, "y_feature.txt"), "w", encoding="utf-8") as f:
    f.write(prep_info["target_feature"])

# Balance minority classes before training to reduce class-bias.
X_train, y_train = apply_smote(X_train, y_train)

trainers = [
    ("Random Forest", lambda X, y: train_rf(X, y, laptop_safe=LAPTOP_SAFE_MODE)),
    ("XGBoost GPU", lambda X, y: train_xgb(X, y, laptop_safe=LAPTOP_SAFE_MODE)),
    ("LightGBM GPU", lambda X, y: train_lgb(X, y, laptop_safe=LAPTOP_SAFE_MODE)),
    ("Naive Bayes", lambda X, y: train_naive_bayes(X, y, laptop_safe=LAPTOP_SAFE_MODE)),
    ("KNN", lambda X, y: train_knn(X, y, laptop_safe=LAPTOP_SAFE_MODE)),
]

models = {}
for model_name, trainer in trainers:
    # Print per-model timing so long runs are visible in terminal output.
    print(f"\n[TRAIN] Starting {model_name}...", flush=True)
    start_t = time.time()
    models[model_name] = trainer(X_train, y_train)
    elapsed = time.time() - start_t
    print(f"[TRAIN] Finished {model_name} in {elapsed:.2f}s", flush=True)

for model_name, model in models.items():
    filename = model_name.lower().replace(" ", "_")
    save_model(model, filename)

# Evaluate all trained models and write outputs/results.csv.
results = evaluate(models, X_test, y_test, class_names=le.classes_)

print(results)

feature_names = X_train.columns

plot_feature_importance(models["XGBoost GPU"], feature_names, "xgboost")
plot_feature_importance(models["Random Forest"], feature_names, "random_forest")
plot_feature_importance(models["LightGBM GPU"], feature_names, "lightgbm")
plot_feature_importance(models["Naive Bayes"], feature_names, "naive_bayes")
plot_feature_importance(models["KNN"], feature_names, "knn")

