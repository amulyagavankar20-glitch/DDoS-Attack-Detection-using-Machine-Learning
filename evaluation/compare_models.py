import os
import re
import time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score


def _safe_name(name):
    return re.sub(r"[^a-z0-9_]+", "_", name.lower().replace(" ", "_")).strip("_")


def _save_confusion_matrix(y_true, y_pred, model_name, output_dir, class_names=None):
    cm = confusion_matrix(y_true, y_pred)

    if class_names is None:
        labels = [str(v) for v in sorted(set(y_true) | set(y_pred))]
    else:
        labels = [str(v) for v in class_names]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.title(f"Confusion Matrix - {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if len(labels) <= 20:
        tick_positions = [i + 0.5 for i in range(len(labels))]
        plt.xticks(tick_positions, labels, rotation=45, ha="right")
        plt.yticks(tick_positions, labels, rotation=0)

    file_name = f"{_safe_name(model_name)}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()


def evaluate(models, X_test, y_test, class_names=None):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "outputs")
    plots_dir = os.path.join(output_dir, "plots")
    cm_dir = os.path.join(plots_dir, "confusion_matrices")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cm_dir, exist_ok=True)

    results = []

    for name, model in models.items():
        start = time.time()
        preds = model.predict(X_test)
        t = time.time() - start

        _save_confusion_matrix(y_test, preds, name, cm_dir, class_names=class_names)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Precision": precision_score(y_test, preds, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, preds, average='weighted', zero_division=0),
            "F1 Score": f1_score(y_test, preds, average='weighted', zero_division=0),
            "Time": t
        })

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "results.csv"), index=False)
    return df

def plot_results(df):
    df.set_index("Model")[["Accuracy","F1 Score"]].plot(kind="bar", figsize=(10,5))
    plt.title("Model Comparison")
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "model_comparison.png"))