import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import correlation_analysis

def plot_histogram(data, column):
    """
    Plot a histogram of the specified column to visually inspect for normality.

    Parameters:
    - data (pd.DataFrame): The dataset containing the column.
    - column (str): The name of the column to plot.

    Returns:
    - None: Displays the histogram.

    Example Usage:
    >>> plot_histogram(data, 'variable_name')
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(f'Histogram of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def plot_qq_plot(data, column):
    """
    Plot a Q-Q plot to visually assess normality.

    Parameters:
    - data (pd.DataFrame): The dataset containing the column.
    - column (str): The name of the column to plot.

    Returns:
    - None: Displays the Q-Q plot.

    Example Usage:
    >>> plot_qq_plot(data, 'variable_name')
    """
    plt.figure(figsize=(10, 6))
    stats.probplot(data[column], dist="norm", plot=plt)
    plt.title(f'Q-Q Plot of {column}')
    plt.show()

def shapiro_test(data, column):
    """
    Perform the Shapiro-Wilk test for normality.

    Parameters:
    - data (pd.DataFrame): The dataset containing the column.
    - column (str): The name of the column to test.

    Returns:
    - tuple: Shapiro-Wilk test statistic and p-value.

    Example Usage:
    >>> stat, p = shapiro_test(data, 'variable_name')
    """
    stat, p = stats.shapiro(data[column])
    print(f'Shapiro-Wilk Test Statistic: {stat:.4f}, p-value: {p:.4f}')
    if p > 0.05:
        print("The data appears to be normally distributed (fail to reject H0).")
    else:
        print("The data does not appear to be normally distributed (reject H0).")
    return stat, p

def kolmogorov_smirnov_test(data, column):
    """
    Perform the Kolmogorov-Smirnov test for normality.

    Parameters:
    - data (pd.DataFrame): The dataset containing the column.
    - column (str): The name of the column to test.

    Returns:
    - tuple: Kolmogorov-Smirnov test statistic and p-value.

    Example Usage:
    >>> stat, p = kolmogorov_smirnov_test(data, 'variable_name')
    """
    stat, p = stats.kstest(data[column], 'norm', args=(np.mean(data[column]), np.std(data[column], ddof=1)))
    print(f'Kolmogorov-Smirnov Test Statistic: {stat:.4f}, p-value: {p:.4f}')
    if p > 0.05:
        print("The data appears to be normally distributed (fail to reject H0).")
    else:
        print("The data does not appear to be normally distributed (reject H0).")
    return stat, p

def anderson_darling_test(data, column):
    """
    Perform the Anderson-Darling test for normality.

    Parameters:
    - data (pd.DataFrame): The dataset containing the column.
    - column (str): The name of the column to test.

    Returns:
    - None: Displays the results of the Anderson-Darling test.

    Example Usage:
    >>> anderson_darling_test(data, 'variable_name')
    """
    result = stats.anderson(data[column], dist='norm')
    print(f'Anderson-Darling Test Statistic: {result.statistic:.4f}')
    print('Critical Values:')
    for i in range(len(result.critical_values)):
        significance_level = result.significance_level[i]
        critical_value = result.critical_values[i]
        print(f'  {significance_level:.1f}%: {critical_value:.4f}')
    if result.statistic < result.critical_values[2]:  # 5% significance level
        print("The data appears to be normally distributed (fail to reject H0).")
    else:
        print("The data does not appear to be normally distributed (reject H0).")

def check_normality_for_all_columns(data):
    """
    Check normality for all numeric columns in the DataFrame.

    Parameters:
    - data (pd.DataFrame): The dataset containing the columns to test.

    Returns:
    - None: Prints the results of normality tests for each numeric column.

    Example Usage:
    >>> check_normality_for_all_columns(data)
    """
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    for column in numeric_columns:
        print(f"\nChecking normality for column: {column}")
        shapiro_test(data, column)
        kolmogorov_smirnov_test(data, column)
        anderson_darling_test(data, column)
