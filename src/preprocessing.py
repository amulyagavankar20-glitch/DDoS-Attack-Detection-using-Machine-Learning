import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from src.feature_engineering import remove_high_corr, remove_low_variance

# This function saves the preprocessed training and testing datasets to both Parquet and CSV formats in a designated "processed" directory. It constructs the file paths based on the current file's location, creates the necessary directories if they don't exist, and then saves the DataFrames with the appropriate filenames.
def save_preprocessed_data(X_train, X_test, y_train, y_test):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Combine the features and labels into single DataFrames for training and testing before saving
    train_df = X_train.copy()
    train_df["label"] = y_train

    # Combine the features and labels into a single DataFrame for testing before saving
    test_df = X_test.copy()
    test_df["label"] = y_test

    # Save the training and testing DataFrames to both Parquet and CSV formats in the processed directory
    train_df.to_parquet(os.path.join(processed_dir, "train.parquet"), index=False)
    test_df.to_parquet(os.path.join(processed_dir, "test.parquet"), index=False)

    train_df.to_csv(os.path.join(processed_dir, "train.csv"), index=False)
    test_df.to_csv(os.path.join(processed_dir, "test.csv"), index=False)

# This function takes a DataFrame as input and performs several preprocessing steps to prepare the data for machine learning models. It separates the features (X) from the target variable (y), encodes the target variable using LabelEncoder, and then splits the data into training and testing sets using train_test_split with stratification to maintain class distribution. After splitting, it applies feature engineering techniques to remove low variance features and highly correlated features from the training set, ensuring that the same transformations are applied to the testing set. Finally, it resets the indices of the resulting DataFrames and Series, saves the preprocessed data if specified, and returns the processed training and testing sets along with the LabelEncoder instance and a dictionary containing information about the preprocessing steps.
def split_data(
    df,
    test_size=0.2,
    random_state=42,
    variance_threshold=1e-5,
    corr_threshold=0.9,
    save_processed=True,
):
    X = df.drop("label", axis=1)
    y = df["label"]

    le = LabelEncoder()
    y = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    # Apply feature engineering to remove low variance and highly correlated features from the training set, and ensure the same transformations are applied to the testing set
    X_train = remove_low_variance(X_train, threshold=variance_threshold)
    X_test = X_test[X_train.columns]

    X_train, dropped_corr = remove_high_corr(X_train, threshold=corr_threshold)
    X_test = X_test.drop(columns=[c for c in dropped_corr if c in X_test.columns])

    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_test = pd.Series(y_test).reset_index(drop=True)

    # Save the preprocessed training and testing datasets to both Parquet and CSV formats in a designated "processed" directory
    if save_processed:
        save_preprocessed_data(X_train, X_test, y_train, y_test)

    # Compile information about the preprocessing steps, including the target feature name, the list of input feature names, the number of features retained after preprocessing, and the number of features dropped due to high correlation. This information is stored in a dictionary for easy reference and debugging purposes.
    info = {
        "target_feature": "label",
        "feature_names": X_train.columns.tolist(),
        "n_features": X_train.shape[1],
        "dropped_corr": len(dropped_corr),
    }

    return (X_train, X_test, y_train, y_test), le, info