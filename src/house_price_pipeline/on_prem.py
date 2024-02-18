from src.utils.helpers import fetch_dataset_from_bucket
from src.utils.constant import gcs_dataset_path
from src.data_preprocess.preprocessing import house_data_preprocessing
from src.house_price_prediction_model.house_model import house_price_train_model

def da():
    dataset = fetch_dataset_from_bucket(gcs_dataset_path)
    x,y,z,p=house_data_preprocessing(dataset)
    house_price_train_model(x, z, "xhs")


da()