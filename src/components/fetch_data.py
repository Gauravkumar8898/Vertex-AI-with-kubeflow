from kfp import dsl
from src.utils.constant import base_image


@dsl.component(
    base_image=base_image,
    packages_to_install=["pyarrow==15.0.0", "pandas==2.0.3", "scikit-learn==1.3.2", "fsspec==2024.2.0",
                         "gcsfs==2024.2.0", "google-cloud-storage==2.14.0"]
)
def fetch_dataset_from_bucket(
        dataset: dsl.Output[dsl.Dataset]
):
    """
    Fetches a dataset from Google Cloud Storage (GCS) and saves it as a Parquet file.
    Args:
        dataset (dsl.Output[dsl.Dataset]): Output path to save the fetched dataset.
    """

    import logging
    from src.utils.constant import gcs_dataset_path
    from src.utils.helpers import fetch_dataset_from_bucket

    # Set up logging configuration
    logging.basicConfig(level=logging.INFO)

    logging.info("Dataset loading....!")

    # Fetch dataset from Google Cloud Storage
    dataset_house = fetch_dataset_from_bucket(gcs_dataset_path)

    logging.info("Dataset loaded from bucket...!")

    logging.info(f"Dataset saving at Parquet file {dataset.uri}")

    # Save the dataset as a Parquet file
    dataset_house.to_parquet(dataset.path)
