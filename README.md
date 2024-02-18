# House Price Prediction Pipeline on Vertex AI with Kubeflow

This repository contains a Kubeflow pipeline component 
and pipeline for house price prediction using Vertex AI
on Google Cloud Platform.


## Overview
The pipeline consists of several components:

1. **Data Fetching Component**: Fetches the dataset from Google Cloud Storage (GCS) and returns it as a pandas DataFrame.
2. **Data Preprocessing Component**: Preprocesses the dataset by encoding labels, dropping unused features, and splitting the dataset into training and testing sets.
3. **Model Training Component**: Trains a machine learning model using the training data.
4. **Model Evaluation Component**: Evaluates the trained model using the testing data.

## Screenshot

![img.png](img.png)
