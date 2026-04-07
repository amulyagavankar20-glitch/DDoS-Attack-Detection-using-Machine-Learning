import os

import matplotlib.pyplot as plt
import pandas as pd

def plot_feature_importance(model, feature_names, model_name):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, "outputs", "plots")
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_

    elif hasattr(model, "coef_"):
        importances = abs(model.coef_[0])

    else:
        print(f"{model_name} does not support feature importance")
        return

    df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(20)

    plt.figure(figsize=(10,6))
    plt.barh(df["Feature"], df["Importance"])
    plt.gca().invert_yaxis()
    plt.title(f"Top Features - {model_name}")
    plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance.png"))
    plt.show()