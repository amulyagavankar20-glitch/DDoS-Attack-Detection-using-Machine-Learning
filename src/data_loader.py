import os

import pandas as pd

def load_and_merge():
    # Determine the base directory and construct paths to the training and testing CSV files
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = os.path.join(base_dir, "data", "merged_training.csv")
    test_path = os.path.join(base_dir, "data", "merged_testing.csv")

    # Load the training and testing datasets
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    # Concatenate the training and testing datasets into a single DataFrame
    df = pd.concat([train, test], ignore_index=True)

    # Clean column names by stripping leading and trailing whitespace
    df.columns = [column.strip() for column in df.columns]

    # Ensure the label column is consistently named 'label'
    if "Label" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"Label": "label"})

    if "label" not in df.columns:
        raise KeyError("Expected a label column named 'Label' or 'label'.")

    # Map the original attack labels to more concise names
    mapping = {
        'DrDoS_UDP': 'UDP',
        'UDP-lag': 'UDPLag',
        'DrDoS_MSSQL': 'MSSQL',
        'DrDoS_LDAP': 'LDAP',
        'DrDoS_NetBIOS': 'NetBIOS'
    }

    df["label"] = df["label"].replace(mapping)

    return df