from src.utils.helpers import (encoding_label, drop_unused_feature,
                               feature_selection, standardize_feature, split_dataset)
from src.utils.constant import col_name, drop_cols, selected_features, target_features, axis, random_state, test_size


def house_data_preprocessing(dataset):
    """
    Preprocesses the house dataset for machine learning tasks.
    Args:
        dataset (DataFrame): The input dataset containing house data.
    Returns:
        tuple: A tuple containing the preprocessed data.
            - x_train_scaled (DataFrame): Preprocessed and standardized features for training.
            - x_test_scaled (DataFrame): Preprocessed and standardized features for testing.
            - y_train (Dataframe): Target labels for training.
            - y_test (Dataframe): Target labels for testing.
    """
    # Step 1: Encoding Labels
    dataset = encoding_label(col_name, dataset)

    # Step 2: Dropping Unused Features
    dataset = drop_unused_feature(dataset, drop_cols, axis=axis)

    # Step 3: Feature Selection
    x, y = feature_selection(dataset, selected_features, target_features)

    # Step 4: Dataset Splitting
    x_train, x_test, y_train, y_test = split_dataset(x, y, random_state, test_size)

    # Step 5: Feature Standardization
    x_train_scaled, x_test_scaled = standardize_feature(x_train, x_test)

    return x_train_scaled, x_test_scaled, y_train, y_test

