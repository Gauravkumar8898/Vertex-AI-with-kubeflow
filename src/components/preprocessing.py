from src.utils.constant import base_image
from kfp import dsl


@dsl.component(
    base_image=base_image,
    packages_to_install=["pyarrow==15.0.0", "pandas==2.0.3", "scikit-learn==1.3.2", "fsspec==2024.2.0",
                         "gcsfs==2024.2.0", "google-cloud-storage==2.14.0"]
)
def data_preprocess(
        dataset: dsl.Input[dsl.Dataset],
        x_train_dataset: dsl.Output[dsl.Dataset],
        x_test_dataset: dsl.Output[dsl.Dataset],
        y_train_dataset: dsl.Output[dsl.Dataset],
        y_test_dataset: dsl.Output[dsl.Dataset]
):
    """
    Preprocesses a dataset for machine learning tasks and saves the preprocessed data.
    Args:
        dataset (dsl.Input[dsl.Dataset]): Input path to the dataset to preprocess.
        x_train_dataset (dsl.Output[dsl.Dataset]): Output path to save the preprocessed training features.
        x_test_dataset (dsl.Output[dsl.Dataset]): Output path to save the preprocessed testing features.
        y_train_dataset (dsl.Output[dsl.Dataset]): Output path to save the preprocessed training labels.
        y_test_dataset (dsl.Output[dsl.Dataset]): Output path to save the preprocessed testing labels.
    """
    from src.data_preprocess.preprocessing import house_data_preprocessing
    import logging
    import pandas as pd

    # Set up logging configuration
    logging.basicConfig(level=logging.INFO)

    try:
        logging.info("Dataset preprocessing.....")

        # Read the dataset for preprocessing
        dataset1 = pd.read_parquet(dataset.path)
        logging.info(f"Dataset columns: {dataset1.columns}")

        # Perform data preprocessing
        x_train, x_test, y_train, y_test = house_data_preprocessing(dataset1)
        logging.info("Dataset processed!")

        # Save the preprocessed data to Parquet files
        x_train.to_parquet(x_train_dataset.path)
        x_test.to_parquet(x_test_dataset.path)
        y_train.to_parquet(y_train_dataset.path)
        y_test.to_parquet(y_test_dataset.path)
        logging.info("Preprocessed data saved in Parquet files!!!")

    except Exception as e:
        logging.info("Preprocessing failed!")
        raise e
