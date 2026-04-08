import os
import time

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils import MODELS_DIR, OUTPUT_DIR, load_data, print_header


def _model_display_name(file_name):
    base_name = os.path.splitext(file_name)[0]
    mapping = {
        "random_forest": "Random Forest",
        "xgboost_gpu": "XGBoost GPU",
        "lightgbm_gpu": "LightGBM GPU",
        "naive_bayes": "Naive Bayes",
        "knn": "KNN",
    }
    return mapping.get(base_name, base_name.replace("_", " ").title())


def _load_feature_names():
    feature_file = os.path.join(OUTPUT_DIR, "x_features.csv")
    if not os.path.exists(feature_file):
        return None

    features_df = pd.read_csv(feature_file)
    if "feature" not in features_df.columns:
        return None

    return features_df["feature"].tolist()


def evaluate_saved_models():
    _, X_test, _, y_test = load_data()
    feature_names = _load_feature_names()

    if feature_names is not None:
        missing_features = [feature for feature in feature_names if feature not in X_test.columns]
        if missing_features:
            raise ValueError(f"Missing expected features in test split: {missing_features}")
        X_test = X_test[feature_names]

    model_files = sorted(
        file_name for file_name in os.listdir(MODELS_DIR) if file_name.endswith(".pkl")
    )

    if not model_files:
        raise FileNotFoundError(f"No .pkl files found in {MODELS_DIR}")

    rows = []
    print_header("Evaluating saved models on processed test data")

    for file_name in model_files:
        model_path = os.path.join(MODELS_DIR, file_name)
        model = joblib.load(model_path)
        model_name = _model_display_name(file_name)

        start_time = time.time()
        predictions = model.predict(X_test)
        inference_time = time.time() - start_time

        row = {
            "Model": model_name,
            "Accuracy": accuracy_score(y_test, predictions),
            "Precision": precision_score(y_test, predictions, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, predictions, average="weighted", zero_division=0),
            "F1 Score": f1_score(y_test, predictions, average="weighted", zero_division=0),
            "Time": inference_time,
        }
        rows.append(row)

        print(
            f"{model_name}: Accuracy={row['Accuracy']:.6f}, Precision={row['Precision']:.6f}, "
            f"Recall={row['Recall']:.6f}, F1={row['F1 Score']:.6f}, Time={row['Time']:.4f}s"
        )

    results_df = pd.DataFrame(rows).sort_values(by="F1 Score", ascending=False).reset_index(drop=True)
    results_path = os.path.join(OUTPUT_DIR, "actual_test_results.csv")
    results_df.to_csv(results_path, index=False)

    print_header("Sorted Results")
    print(results_df.to_string(index=False))
    print(f"\nSaved actual test results to: {results_path}")


if __name__ == "__main__":
    evaluate_saved_models()