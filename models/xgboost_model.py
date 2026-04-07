from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from src.utils import get_device

def train_xgb(X_train, y_train, laptop_safe=False):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        stratify=y_train,
        random_state=42,
    )

    n_estimators = 200 if laptop_safe else 1200
    max_depth = 5 if laptop_safe else 8
    early_stopping_rounds = 15 if laptop_safe else 40

    params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=0.1,
        tree_method="hist",
        subsample=0.9,
        colsample_bytree=0.9,
        early_stopping_rounds=early_stopping_rounds,
        random_state=42,
    )

    if get_device() == "cuda":
        params["device"] = "cuda"

    model = XGBClassifier(**params)
    model.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
    return model