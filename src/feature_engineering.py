import numpy as np

# Feature engineering functions
def remove_low_variance(df, threshold=1e-5):
    return df.loc[:, df.var() > threshold]

# This function identifies and removes features that are highly correlated with each other, based on a specified correlation threshold. It computes the correlation matrix of the DataFrame, extracts the upper triangle of the matrix to avoid redundant checks, and then identifies columns to drop if they have a correlation value greater than the threshold with any other column. Finally, it returns a new DataFrame with the identified columns removed, along with a list of the dropped columns for reference.
def remove_high_corr(df, threshold=0.9):
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return df.drop(columns=drop), drop