import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB


def train_naive_bayes(X_train, y_train, cv_folds=5, n_jobs=-1, laptop_safe=False):
    # Tune only var_smoothing; this is the key GaussianNB hyperparameter.
    smoothing_grid = np.logspace(-12, -6, 13) if laptop_safe else np.logspace(-12, -3, 25)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    search = GridSearchCV(
        estimator=GaussianNB(),
        param_grid={"var_smoothing": smoothing_grid},
        scoring="f1_weighted",
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_