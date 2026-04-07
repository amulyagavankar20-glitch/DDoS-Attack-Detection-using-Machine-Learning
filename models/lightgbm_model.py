import lightgbm as lgb
from sklearn.model_selection import train_test_split

def train_lgb(X_train, y_train, laptop_safe=False):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.15,
        stratify=y_train,
        random_state=42,
    )

    n_estimators = 300 if laptop_safe else 1500
    num_leaves = 31 if laptop_safe else 63
    early_stopping_rounds = 15 if laptop_safe else 40

    params = {
        "n_estimators": n_estimators,
        "learning_rate": 0.05,
        "num_leaves": num_leaves,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "random_state": 42,
        "verbosity": -1,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_fit,
        y_fit,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model