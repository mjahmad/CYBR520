import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau

def calculate_pearson_correlation(data, x_column, y_column):
    """
    Calculate the Pearson correlation coefficient between two continuous variables.

    Parameters:
    - data (pd.DataFrame): The dataset containing the variables to analyze.
    - x_column (str): The name of the first variable (independent).
    - y_column (str): The name of the second variable (dependent).

    Returns:
    - float: Pearson correlation coefficient.
    - float: p-value indicating significance of the correlation.

    Example Usage:
    >>> r, p = calculate_pearson_correlation(data, 'variable1', 'variable2')
    """
    r, p = pearsonr(data[x_column], data[y_column])
    print(f"Pearson Correlation: {r:.4f}, p-value: {p:.4f}")
    return r, p

def calculate_spearman_correlation(data, x_column, y_column):
    """
    Calculate the Spearman rank correlation coefficient between two variables.

    Parameters:
    - data (pd.DataFrame): The dataset containing the variables to analyze.
    - x_column (str): The name of the first variable (independent).
    - y_column (str): The name of the second variable (dependent).

    Returns:
    - float: Spearman correlation coefficient.
    - float: p-value indicating significance of the correlation.

    Example Usage:
    >>> r, p = calculate_spearman_correlation(data, 'variable1', 'variable2')
    """
    r, p = spearmanr(data[x_column], data[y_column])
    print(f"Spearman Correlation: {r:.4f}, p-value: {p:.4f}")
    return r, p

def calculate_kendall_correlation(data, x_column, y_column):
    """
    Calculate the Kendall tau correlation coefficient between two variables.

    Parameters:
    - data (pd.DataFrame): The dataset containing the variables to analyze.
    - x_column (str): The name of the first variable (independent).
    - y_column (str): The name of the second variable (dependent).

    Returns:
    - float: Kendall tau correlation coefficient.
    - float: p-value indicating significance of the correlation.

    Example Usage:
    >>> r, p = calculate_kendall_correlation(data, 'variable1', 'variable2')
    """
    r, p = kendalltau(data[x_column], data[y_column])
    print(f"Kendall Correlation: {r:.4f}, p-value: {p:.4f}")
    return r, p

def plot_correlation_heatmap(data):
    """
    Plot a heatmap showing the correlation coefficients of all numeric variables in the dataset.

    Parameters:
    - data (pd.DataFrame): The dataset containing numerical features.

    Returns:
    - None: Displays a heatmap of the correlation matrix.

    Example Usage:
    >>> plot_correlation_heatmap(data)
    """
    corr_matrix = data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap')
    plt.show()
