from imblearn.over_sampling import SMOTE

# This function applies the SMOTE (Synthetic Minority Over-sampling Technique) algorithm to the given feature matrix X and target vector y. SMOTE is used to address class imbalance by generating synthetic samples for the minority class. The function takes in the feature matrix X and target vector y, applies SMOTE to create a balanced dataset, and returns the resampled feature matrix and target vector.
def apply_smote(X, y):
    sm = SMOTE(random_state=42)
    return sm.fit_resample(X, y)