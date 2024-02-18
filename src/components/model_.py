from src.utils.constant import base_image
from kfp import dsl


@dsl.component(
    base_image=base_image,
    packages_to_install=["pyarrow==15.0.0", "pandas==2.0.3", "scikit-learn==1.3.2", "fsspec==2024.2.0",
                         "gcsfs==2024.2.0", "google-cloud-storage==2.14.0", "joblib==1.3.2"]
)
def model_fitting(
        x_train_dataset: dsl.Input[dsl.Dataset],
        y_train_dataset: dsl.Input[dsl.Dataset],
        model_artifact_path: dsl.Output[dsl.Model]
):
    """
    Fits a machine learning model using training data and saves the trained model artifact.
    Args:
        x_train_dataset (dsl.Input[dsl.Dataset]): Input path to the training features dataset.
        y_train_dataset (dsl.Input[dsl.Dataset]): Input path to the training labels dataset.
        model_artifact_path (dsl.Output[dsl.Model]): Output path to save the trained model artifact.
    """
    import logging
    import pandas as pd
    from src.house_price_prediction_model.house_model import house_price_train_model

    # Set up logging configuration
    logging.basicConfig(level=logging.INFO)

    logging.info(f"Reading processed train data from: {x_train_dataset}")
    logging.info(f"Reading processed train data from: {y_train_dataset}")

    # Read the training features and labels datasets
    x_train_datasets = pd.read_parquet(x_train_dataset.path)
    y_train_datasets = pd.read_parquet(y_train_dataset.path)

    # Fit the machine learning model using the training data
    house_price_train_model(x_train_datasets, y_train_datasets, model_artifact_path)
