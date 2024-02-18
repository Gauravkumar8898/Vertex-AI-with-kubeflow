from pathlib import Path

curr_path = Path(__file__).parents[1]
curr_path_prev = Path(__file__).parents[2]
data_directory = curr_path / 'data'

test_dataset_path = curr_path_prev / "test_suite/testdata.csv"

# column name change categorical into numerical
col_name = "Neighborhood"

# dropped features list
drop_cols = ['id']
axis = 1

# dataset path which exist on gcp bucket
gcs_dataset_path = "gs://marine-might-413408/house_price_predictions"

# feature selection
selected_features = ['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt']
target_features = "Price"
template_path = curr_path_prev / "pipeline.json"

package_path = 'pipeline.json'

# these parameters use for train test split
random_state = 42
test_size = 0.2

pipeline_name = "house_price_pipelines"
description = "house_price_prediction_test"
# pipeline staged path
staged_path = "gs://marine-might-413408/output"

# for  pipeline
pipeline_job_name = "house"
project = "marine-might-413408"
location = "us-central1"

# use for base image
bucket_name = "marine-might-413408"
PROJECT_ID = bucket_name

# for base images
REPO_NAME = "customrepo"
IMAGE_NAME = "house_price_kubeflow"
IMAGE_TAG = "latest-1.0"
# base_images for custom models
base_image = f'us-central1-docker.pkg.dev/{PROJECT_ID}/{REPO_NAME}/{IMAGE_NAME}:{IMAGE_TAG}'

model_name = "model_house"
