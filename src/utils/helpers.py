import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def fetch_dataset_from_bucket(gcs_dataset_path):
    """
    Fetches a dataset from Google Cloud Storage (GCS) and returns it as a pandas DataFrame.
    Args:
        gcs_dataset_path (str): The path to the dataset in Google Cloud Storage.
    Returns:
        pandas.DataFrame: The dataset fetched from the specified GCS path.
    """
    dataset = pd.read_csv(gcs_dataset_path)
    return dataset


def encoding_label(col, df):
    """
        :param col: The column to be encoded in the dataset.
        :param df: The input dataset in a Pandas DataFrame.
        :return: The dataset after performing label encoding on the specified column.
        """
    encoder = LabelEncoder()
    df[col] = encoder.fit_transform(df[col])
    return df


def drop_unused_feature(dataset, lists, axis):
    """
        :param dataset: The input dataset in a Pandas DataFrame.
        :param lists: A list of feature names to be dropped from the dataset.
        :param axis:
        :return: Return the modified dataset.
        """
    datasets = dataset.drop(lists, axis=axis)
    return datasets


def feature_selection(data_frame, selected_features, target_features):
    """
    getting the dataframe and doing feature selection
    :param: data_frame:
    :return: data_frame after doing feature selection
    """
    x_feature_dataframe = data_frame[selected_features]
    y_feature_dataframe = data_frame[target_features]
    return x_feature_dataframe, y_feature_dataframe


def standardize_feature(x_train, x_test):
    """
    :param x_train: The training set of feature variables.
    :param x_test: The testing set of feature variables.
    :return:
            x_train_scaled: The standardized training set of feature variables.
            x_test_scaled: The standardized testing set of feature variables.
    """
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    x_train_scaled = pd.DataFrame(x_train_scaled)
    x_test_scaled = pd.DataFrame(x_test_scaled)
    return x_train_scaled, x_test_scaled


def split_dataset(df_feature, df_targets, random_state, test_size):
    """
    :param df_feature: The DataFrame containing feature variables.
    :param df_targets: The DataFrame containing the target variable.
    :param random_state:
    :param test_size:
    :return:
            x_train: The training set of feature variables.
            x_test: The testing set of feature variables.
            y_train: The training set of target variable.
            y_test: The testing set of target variable.
    """
    x_train, x_test, y_train, y_test = \
        (train_test_split(df_feature, df_targets, random_state=random_state, test_size=test_size))
    x_train = pd.DataFrame(x_train)
    x_test = pd.DataFrame(x_test)
    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)
    return x_train, x_test, y_train, y_test



