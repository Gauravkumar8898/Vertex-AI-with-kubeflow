import kfp.dsl as dsl
from src.components.fetch_data import fetch_dataset_from_bucket
from src.components.preprocessing import data_preprocess
from src.components.evaluation import model_evaluation
from src.utils.constant import pipeline_name, description, project, location
from src.components.model_ import model_fitting
from google.cloud import aiplatform
import logging


@dsl.pipeline(
    name=pipeline_name,
    description=description
)
def kube_pipeline():
    aiplatform.init(project=project, location=location)
    logging.info("Initializing AI Platform")
    dataset = fetch_dataset_from_bucket()
    data = data_preprocess(dataset=dataset.output).after(dataset)
    model = (model_fitting(x_train_dataset=data.outputs["x_train_dataset"],
                           y_train_dataset=data.outputs["y_train_dataset"]).after(data))

    model_evaluation(x_test_data_path=data.outputs["x_test_dataset"],
                     y_test_data_path=data.outputs["y_test_dataset"], model_path=model.output).after(model)
