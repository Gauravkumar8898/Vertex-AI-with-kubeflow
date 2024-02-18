# set env variable
export PROJECT_ID=$(gcloud config list project --format "value(core.project)")
export REPO_NAME=customrepo
export IMAGE_NAME=house_price_kubeflow
export IMAGE_TAG=latest-1.0
export IMAGE_URI=us-central1-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${IMAGE_NAME}:${IMAGE_TAG}

# for build docker images
docker build -f Dockerfile -t ${IMAGE_URI} ./

# for push docker images to artifact registry
docker push ${IMAGE_URI}