from typing import Any

from scipy.signal import correlation_lags
from sklearn.preprocessing import LabelEncoder
import kagglehub
from pandas import read_csv, DataFrame
import seaborn as sns
import matplotlib.pyplot as plt
from pandas.io.parsers import TextFileReader
from sklearn.model_selection import train_test_split
import normality_checks
import correlation_analysis

import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split

import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_and_sample(file_path: str, sample_fraction: float) -> Any | None:
    """
    Function to load a dataset, handle encoding issues, drop NaN values,
    and subsample a fraction while maintaining class ratio.

    Parameters:
    - file_path (str): Path to the input CSV file.
    - sample_fraction (float): Fraction of the dataset to sample (default is 5%).

    Returns:
    - pd.DataFrame: Subsampled dataset.
    """
    try:
        # Step 1: Try loading the dataset with 'utf-8' encoding and handle encoding errors
        try:
            data = pd.read_csv(file_path, encoding='utf-8', low_memory=False)
        except UnicodeDecodeError:
            print("UTF-8 encoding failed, trying ISO-8859-1 (Latin-1) encoding.")
            data = pd.read_csv(file_path, encoding='ISO-8859-1', low_memory=False)

        # Step 2: Drop rows with NaN values
        data = data.dropna()

        # Step 3: Check the dependent variable (assumed to be the last column)
        dependent_var = data.columns[-1]

        # Step 4: Subsample the data while maintaining the class ratio
        data_sample, _ = train_test_split(
            data, test_size=1-sample_fraction, stratify=data[dependent_var], random_state=42
        )

        # Step 5: Return the subsampled dataset
        return data_sample

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def data_correlation(data):
    """
    Calculate and visualize the correlation matrix for a given dataset.

    This function performs the following steps:
    1. Computes the correlation matrix for the dataset using Pearson's method.
    2. Visualizes the correlation matrix using a heatmap for easier interpretation.
    3. Optionally prints the correlation of each feature with the last column in the dataset.

    Parameters:
    ----------
    data : pandas.DataFrame
        A DataFrame containing the dataset. All columns should be numerical.

    Returns:
    -------
    None
        The function does not return any value. It displays a heatmap of the correlation matrix
        and prints the correlation values of the last column.

    Example:
    --------
    # Assuming you have a DataFrame `df` with numerical columns:
    import pandas as pd

    df = pd.read_csv('your_dataset.csv')  # Load your dataset
    data_correlation(df)
    """

    # Calculate the correlation matrix
    correlation_matrix = data.corr()

    # Step 4: Visualize the correlation matrix
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix')
    plt.show()

    # Optionally, print the correlation with the last column
    label_column = data.columns[-1]  # Get the name of the last column
    print("Correlation with", label_column, ":")
    print(correlation_matrix[label_column])


def filter_features_correlation(data):
    """
    Filter features based on correlation with the target variable and visualize the results.

    This function performs the following steps:
    1. Converts the last column of the dataset (assumed to be the target variable) from a string
       format to numeric using Label Encoding.
    2. Computes the correlation matrix for the dataset.
    3. Filters the features based on a specified correlation threshold with the target variable.
    4. Visualizes the filtered correlation matrix using a heatmap.

    Parameters:
    ----------
    data : pandas.DataFrame
        A DataFrame containing the dataset. The last column is expected to be the target variable,
        which is converted to numeric format if it is of string type.

    Returns:
    -------
    None
        The function does not return any value. It displays a heatmap of the filtered correlation
        matrix for important features.

    Example:
    --------
    # Assuming you have a DataFrame `df` with numeric columns and a string target variable:
    import pandas as pd
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv('your_dataset.csv')  # Load your dataset
    filter_features(df)
    """
    # Convert the last string column to numeric using Label Encoding
    label_column = data.columns[-1]
    label_encoder = LabelEncoder()
    if data[label_column].dtype == 'object':
        data[label_column] = label_encoder.fit_transform(data[label_column])

    # Calculate the correlation matrix
    correlation_matrix = data.corr()

    # Set a threshold for correlation
    threshold = 0.2  # Adjust this value based on your needs

    # Filter the correlation matrix for strong correlations with the target variable
    strong_correlations = correlation_matrix[label_column].abs().sort_values(ascending=False)
    important_features = strong_correlations[strong_correlations > threshold].index.tolist()

    # Create a filtered correlation matrix for important features
    filtered_correlation_matrix = data[important_features].corr()

    # Visualize the filtered correlation matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(filtered_correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Filtered Correlation Matrix')
    plt.show()


# Download latest version
path = kagglehub.dataset_download("sweety18/cicids2017-full-dataset")

print("Path to dataset files:", path)

# Load the dataset
file_path = path+"\\"+"combine.csv"
print(file_path)

# Example usage:
sampled_data = preprocess_and_sample(file_path, 0.001)
print(sampled_data)

# Step 2: Convert the last string column to numeric using Label Encoding
label_column = sampled_data.columns[-1]  # Get the name of the last column
label_encoder = LabelEncoder()

# Check if the last column is of string type
if sampled_data[label_column].dtype == 'object':
    sampled_data[label_column] = label_encoder.fit_transform(sampled_data[label_column])

##############################################################################################
##############################################################################################
# When running  this codes, call methods from here. Doi not change anything else.
##############################################################################################
##############################################################################################


#data_correlation(sampled_data)
#filter_features_correlation(sampled_data)
#normality_checks.plot_histogram(sampled_data,' Label')
#normality_checks.plot_qq_plot(sampled_data,' Label')
#normality_checks.check_normality_for_all_columns(sampled_data)
#normality_checks.correlation_analysis(sampled_data)
#normality_checks.check_normality_for_all_columns(sampled_data)
#normality_checks.kolmogorov_smirnov_test(sampled_data, ' Label')
#normality_checks.anderson_darling_test(sampled_data,' Label')
#correlation_analysis.calculate_pearson_correlation(sampled_data,' Idle Min',' Label')
correlation_analysis.plot_correlation_heatmap(sampled_data)



