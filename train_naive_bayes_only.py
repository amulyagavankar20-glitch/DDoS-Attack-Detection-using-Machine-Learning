import os
import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from models.naive_bayes import train_naive_bayes
from src.data_loader import load_and_merge
from src.preprocessing import split_data
from src.smote import apply_smote
from src.utils import save_model


LAPTOP_SAFE_MODE = os.getenv("LAPTOP_SAFE_MODE", "1") == "1"


def _build_result_row(model, X_test, y_test):
    start = time.time()
    preds = model.predict(X_test)
    inference_time = time.time() - start

    return {
        "Model": "Naive Bayes",
        "Accuracy": accuracy_score(y_test, preds),
        "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
        "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        "F1 Score": f1_score(y_test, preds, average="weighted", zero_division=0),
        "Time": inference_time,
    }


def _upsert_naive_bayes_result(results_path, row):
    columns = ["Model", "Accuracy", "Precision", "Recall", "F1 Score", "Time"]

    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        if "Model" in results_df.columns and (results_df["Model"] == "Naive Bayes").any():
            idx = results_df.index[results_df["Model"] == "Naive Bayes"][0]
            for col in columns:
                results_df.loc[idx, col] = row[col]
        else:
            results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)
    else:
        results_df = pd.DataFrame([row], columns=columns)

    # Keep a stable column order for compatibility with existing reporting.
    for col in columns:
        if col not in results_df.columns:
            results_df[col] = None
    results_df = results_df[columns]

    results_df.to_csv(results_path, index=False)
    return results_df


def main():
    df = load_and_merge()

    (X_train, X_test, y_train, y_test), _, prep_info = split_data(df)
    print(f"Feature engineering complete: {prep_info['n_features']} features retained")
    print(f"Target feature (y): {prep_info['target_feature']}")
    print(f"Input features (X) count: {len(prep_info['feature_names'])}")

    X_train, y_train = apply_smote(X_train, y_train)

    print("[TRAIN] Starting Naive Bayes...", flush=True)
    cv_folds = 3 if LAPTOP_SAFE_MODE else 5
    n_jobs = 2 if LAPTOP_SAFE_MODE else -1
    train_start = time.time()
    model = train_naive_bayes(
        X_train,
        y_train,
        cv_folds=cv_folds,
        n_jobs=n_jobs,
        laptop_safe=LAPTOP_SAFE_MODE,
    )
    train_elapsed = time.time() - train_start
    print(f"[TRAIN] Finished Naive Bayes in {train_elapsed:.2f}s", flush=True)

    save_model(model, "naive_bayes")

    row = _build_result_row(model, X_test, y_test)
    results_path = os.path.join("outputs", "results.csv")
    results_df = _upsert_naive_bayes_result(results_path, row)

    print("Updated Naive Bayes row in outputs/results.csv")
    print(results_df.loc[results_df["Model"] == "Naive Bayes"])


if __name__ == "__main__":
    main()
