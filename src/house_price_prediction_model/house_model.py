import logging
from kfp import dsl
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def house_price_train_model(
        x_train,
        y_train,
        model_artifact_path: dsl.Output[dsl.Model]
):
    """
    Function to get dataset path,
    then fit model and save to artifact and giver artifact path as output.
    dataset_path: dataset parquet file path
    :model_artifact_path: model path saved at artifact registry and given path as output.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Task: Fitting linear regression model")
    model = LinearRegression()
    try:
        # Train Linear Regression model
        model = LinearRegression()
        model.fit(x_train, y_train)
        logging.info("Linear Regression model trained.")

        joblib.dump(model, model_artifact_path.path)
        logging.info("model dump in joblib format")

    except Exception as e:
        logging.error('An error occurred')
        raise e


def evaluation_model(model, x_test, y_test):
    logging.basicConfig(level=logging.INFO)
    logging.info("evaluation for model ......")
    y_pred = model.predict(x_test)
    score = r2_score(y_test, y_pred)
    logging.info(f"r2score of model: {score}")
