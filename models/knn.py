from sklearn.neighbors import KNeighborsClassifier

def train_knn(X_train, y_train, n_neighbors=10, laptop_safe=False):
    n_neighbors = 5 if laptop_safe else n_neighbors

    model = KNeighborsClassifier(n_neighbors=n_neighbors, n_jobs=2)
    model.fit(X_train, y_train)
    return model