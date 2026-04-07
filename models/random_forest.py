from sklearn.ensemble import RandomForestClassifier

def train_rf(X, y, laptop_safe=False):
    cpu_jobs = 2 if laptop_safe else -1
    
    # Use optimized parameters for fast, stable training
    n_estimators = 300 if laptop_safe else 500
    max_depth = 15 if laptop_safe else 20
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features="sqrt",
        criterion="gini",
        class_weight="balanced",
        n_jobs=cpu_jobs,
        random_state=42,
    )
    model.fit(X, y)
    return model