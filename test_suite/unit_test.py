from src.utils.helpers import (drop_unused_feature, fetch_dataset_from_bucket, split_dataset,
                               standardize_feature, encoding_label, feature_selection, train_test_split)
import unittest
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.utils.constant import test_dataset_path, gcs_dataset_path


class TestHelpers(unittest.TestCase):

    def test_encoding_label(self):
        # Load the test dataset
        df = pd.read_csv(test_dataset_path)
        # Specify the column to be encoded
        col_to_encode = 'Neighborhood'
        # Encode the specified column
        encoded_df = encoding_label(col_to_encode, df)
        # Check if the column has been encoded
        self.assertTrue(encoded_df[col_to_encode].dtype == 'int64')

    @staticmethod
    def test_drop_unused_feature():
        # Define the features to drop
        test_dataset = pd.read_csv(test_dataset_path)
        features_to_drop = ['SquareFeet', 'Bedrooms']

        # Drop the specified features
        modified_dataset = drop_unused_feature(test_dataset, features_to_drop, axis=1)

        # Check if the dropped features are not in the modified dataset
        for feature in features_to_drop:
            assert feature not in modified_dataset.columns

    @staticmethod
    def test_feature_selection():
        test_dataset = pd.read_csv(test_dataset_path)
        # Define the selected and target features
        selected_features = ['SquareFeet', 'Bedrooms']
        target_features = ['Price']

        # Perform feature selection
        x_features, y_features = feature_selection(test_dataset, selected_features, target_features)

        # Check if the selected and target features are correct
        assert list(x_features.columns) == selected_features
        assert list(y_features.columns) == target_features

    @staticmethod
    def test_standardize_feature():
        # Unpack sample data
        data_train = {'feature1': [1, 2, 3],
                      'feature2': [4, 5, 6]}
        data_test = {'feature1': [7, 8, 9],
                     'feature2': [10, 11, 12]}
        x_train = pd.DataFrame(data_train)
        x_test = pd.DataFrame(data_test)

        # Standardize features
        x_train_scaled, x_test_scaled = standardize_feature(x_train, x_test)

        # Check if the scaled features have the correct shape
        assert x_train_scaled.shape == x_train.shape
        assert x_test_scaled.shape == x_test.shape

        # Check if the scaled features have the correct values
        scaler = StandardScaler()
        expected_x_train_scaled = scaler.fit_transform(x_train)
        expected_x_test_scaled = scaler.transform(x_test)
        pd.testing.assert_frame_equal(x_train_scaled, pd.DataFrame(expected_x_train_scaled))
        pd.testing.assert_frame_equal(x_test_scaled, pd.DataFrame(expected_x_test_scaled))

    def test_split_dataset(self):
        # Create sample data for testing
        data_feature = {'feature1': [1, 2, 3, 4, 5],
                        'feature2': [6, 7, 8, 9, 10]}
        data_targets = {'target': [11, 12, 13, 14, 15]}
        df_feature = pd.DataFrame(data_feature)
        df_targets = pd.DataFrame(data_targets)

        # Call the split_dataset function
        random_state = 42
        test_size = 0.2
        x_train, x_test, y_train, y_test = split_dataset(df_feature, df_targets, random_state, test_size)

        # Check if the split datasets have the correct shape
        self.assertEqual(x_train.shape[0], 4)  # 80% of 5 samples for training
        self.assertEqual(x_test.shape[0], 1)  # 20% of 5 samples for testing
        self.assertEqual(y_train.shape[0], 4)
        self.assertEqual(y_test.shape[0], 1)

        # Check if the split datasets have the correct values
        expected_x_train, expected_x_test, expected_y_train, expected_y_test = \
            train_test_split(df_feature, df_targets, random_state=random_state, test_size=test_size)

        pd.testing.assert_frame_equal(x_train, pd.DataFrame(expected_x_train))
        pd.testing.assert_frame_equal(x_test, pd.DataFrame(expected_x_test))
        pd.testing.assert_frame_equal(y_train, pd.DataFrame(expected_y_train))
        pd.testing.assert_frame_equal(y_test, pd.DataFrame(expected_y_test))

    def test_fetch_dataset_from_bucket(self):

        # Call the fetch_dataset_from_bucket function
        dataset = fetch_dataset_from_bucket(gcs_dataset_path)

        # Check if the returned dataset is a pandas DataFrame
        self.assertIsInstance(dataset, pd.DataFrame)


if __name__ == '__main__':
    unittest.main()
