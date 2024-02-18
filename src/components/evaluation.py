from src.utils.constant import base_image
from kfp import dsl


@dsl.component(
    base_image=base_image,  # Base Docker image to use for the component
    packages_to_install=[  # List of Python packages to install in the Docker image
        "pyarrow==15.0.0",
        "pandas==2.0.3",
        "scikit-learn==1.3.2",
        "fsspec==2024.2.0",
        "gcsfs==2024.2.0",
        "google-cloud-storage==2.14.0",
        "joblib==1.3.2"
    ]
)
def model_evaluation(
        x_test_data_path: dsl.Input[dsl.Dataset],
        y_test_data_path: dsl.Input[dsl.Dataset],
        model_path: dsl.Input[dsl.Model],
        r2score: dsl.Output[dsl.Metrics]
):
    """
    Evaluates a trained machine learning model using test data and logs the R^2 score.
    Args:
        x_test_data_path (dsl.Input[dsl.Dataset]): Path to the test feature dataset.
        y_test_data_path (dsl.Input[dsl.Dataset]): Path to the test label dataset.
        model_path (dsl.Input[dsl.Model]): Path to the trained model.
        r2score (dsl.Output[dsl.Metrics]): Output metric for R^2 score.
    Raises:
        Exception: If an error occurs during model evaluation.
    """

    # Import necessary modules within the function to ensure they're available in the Docker image
    from src.house_price_prediction_model.house_model import \
        evaluation_model
    import logging
    logging.info("Reading model")
    import joblib
    import pandas as pd

    try:
        # Load the model from the provided model path
        model = joblib.load(model_path.path)

        # Read the test features (x_test) and labels (y_test) from the provided dataset paths
        x_test = pd.read_parquet(x_test_data_path.path)
        y_test = pd.read_parquet(y_test_data_path.path)

        # Evaluate the model using the evaluation_model function
        score = evaluation_model(model, x_test, y_test)
        # Calculating evaluation score

        logging.info(f"score model:{score}")
        # Log message indicating the evaluation score
        r2score.log_metric("r2score", score)
        # Log the R^2 score as a metric

    except Exception as e:
        raise e  # Raise any exceptions that occur during execution
