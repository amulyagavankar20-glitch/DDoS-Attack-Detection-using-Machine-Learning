import os
import pandas as pd
import numpy as np
import joblib
import torch

# PATHS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# DEVICE (GPU CHECK)
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

# SAVE / LOAD MODELS
def save_model(model, name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"Model saved: {path}")


def load_model(name):
    path = os.path.join(MODELS_DIR, f"{name}.pkl")
    return joblib.load(path)

# LOAD DATA (FINAL CLEAN DATA)
def load_data():
    train_path = os.path.join(DATA_DIR, "processed", "train.parquet")
    test_path  = os.path.join(DATA_DIR, "processed", "test.parquet")

    train_df = pd.read_parquet(train_path)
    test_df  = pd.read_parquet(test_path)

    X_train = train_df.drop("label", axis=1)
    y_train = train_df["label"]

    X_test = test_df.drop("label", axis=1)
    y_test = test_df["label"]

    return X_train, X_test, y_train, y_test

# LOGGING
def print_header(title):
    print("\n" + "="*60)
    print(title)
    print("="*60)

# RESULTS FORMATTER
def format_results(results_list):
    return pd.DataFrame(results_list).sort_values(by="F1 Score", ascending=False)