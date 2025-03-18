# End to End Machine Learning Pipeline for 🧠 Brain Stroke Prediction classfication Project

<div align="center">

![Brain Stroke Prediction](https://via.placeholder.com/800x200?text=Brain+Stroke+Prediction+Project)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-orange.svg)](https://scikit-learn.org/stable/)
[![Flask](https://img.shields.io/badge/Flask-Latest-lightgrey.svg)](https://flask.palletsprojects.com/)
[![MLflow](https://img.shields.io/badge/MLflow-Latest-blue.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

*A machine learning solution for predicting brain stroke risk using patient data*

</div>

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Data](#data)
- [Model](#model)
- [Monitoring](#monitoring)
- [API](#api)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Setup](#setup-1)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

**Stroke is a leading cause of death and disability worldwide.**  Rapid and accurate prediction of stroke risk is essential for enabling preventative measures and timely medical intervention. This project aims to develop a machine learning-based stroke prediction system to assist healthcare professionals in identifying individuals at high risk of stroke.

<div align="center">
  
![Stroke Risk Factors](https://via.placeholder.com/600x300?text=Stroke+Risk+Factors+Visualization)

</div>

By utilizing patient data and applying machine learning algorithms, this project seeks to:

*   **Improve Stroke Risk Assessment:** Provide a more accurate and data-driven approach to assess stroke risk compared to traditional methods.
*   **Enable Early Intervention:** Facilitate early identification of high-risk individuals, allowing for timely preventative interventions and lifestyle modifications.
*   **Enhance Clinical Decision Support:** Offer a valuable tool for clinicians to support their decision-making process in stroke prevention and management.

## Project Overview

The Brain Stroke Prediction project is structured as an end-to-end machine learning pipeline, encompassing the following key stages:

1.  **Data Ingestion**:
    *   Acquiring the dataset from its source and loading it into the project environment.
    *   Scripts for data ingestion are located in `src/data_processing/ingest.py`.

2.  **Data Exploration and Preprocessing**:
    *   In-depth analysis of the dataset to understand its structure, features, and potential issues such as missing values or outliers.
    *   Cleaning and preprocessing the data to handle missing values, inconsistencies, and prepare it for feature engineering and model training.
    *   Data preprocessing scripts can be found in `src/data_processing/`.

3.  **Feature Engineering**:(optional)(not using currently)
    *   Creating new features from existing data or transforming features to improve model performance.
    *   This may include scaling numerical features, encoding categorical variables, and generating interaction features.
    *   Feature engineering scripts are located in `src/feature_engineering/`.

4.  **Model Development and Training**:
    *   Selecting an appropriate machine learning model for stroke prediction for demo notenook I am using, Random Forest, XGBoosting model. but in final version we will use random forest which is more accurate prediction.
    *   Training the model using the preprocessed data and engineered features.
    *   Utilizing MLflow for experiment tracking and model management to ensure reproducibility and facilitate model comparison.
    *   Model training scripts are available in `src/model/train.py`.

5.  **Model Evaluation**:
    *   Rigorous evaluation of the trained model's performance using relevant metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.
    *   Generating comprehensive evaluation reports and visualizations to assess model strengths and weaknesses.
    *   Model evaluation scripts are located in `src/evaluation/evaluate.py`.

6.  **Monitoring and Observability**:
    *   Implementing monitoring systems to track model performance in production and detect data drift or concept drift.
    *   Utilizing Prometheus and Grafana for real-time monitoring and alerting.
    *   Exporting key metrics using `src/monitoring/metrics_exporter.py` and configuring Grafana dashboards in `grafana/dashboards/` and Prometheus alerts in `prometheus/alert_rules.yml`.

7.  **API Deployment**:
    *   Deploying the trained stroke prediction model as a RESTful API using Flask.
    *   Enabling seamless integration with other applications and systems to consume the model's predictions.
    *   API application code is located in `src/api/app.py`.

This project leverages a range of technologies and tools, including:

*   **Python**: The primary programming language for the project.
*   **Pandas**: For data manipulation and analysis.
*   **Scikit-learn**: For machine learning model development and evaluation.
*   **MLflow**: For experiment tracking, model management, and deployment.
*   **Prometheus and Grafana**: For monitoring and visualization.
*   **Docker**: For containerizing the application and ensuring reproducibility.
*   **Flask**: For building the REST API.

## Data

The dataset for this project is `brain.csv`, located in the `data/` directory. It contains anonymized patient data with the following features:

<div align="center">
  
![Data Distribution](https://via.placeholder.com/700x350?text=Data+Distribution+Visualization)

</div>

| Feature | Description | Type |
|---------|-------------|------|
| id | Unique identifier for each patient | Integer |
| gender | Patient's gender (Male, Female, Other) | Categorical |
| age | Patient's age in years | Numerical |
| hypertension | Whether the patient has hypertension (0 or 1) | Binary |
| heart_disease | Whether the patient has heart disease (0 or 1) | Binary |
| ever_married | Whether the patient has ever been married (No or Yes) | Categorical |
| work_type | Type of work (e.g., Private, Self-employed, Govt_job) | Categorical |
| Residence_type | Type of residence (Rural or Urban) | Categorical |
| avg_glucose_level | Average glucose level in blood | Numerical |
| bmi | Body mass index | Numerical |
| smoking_status | Patient's smoking status | Categorical |
| stroke | **(Target)** Whether the patient had a stroke (1) or not (0) | Binary |


## Model

The project utilizes a machine learning model to predict the likelihood of stroke based on the patient features described above.  The model is trained and evaluated using scripts in the `src/model/` directory. **[Specify Random forest Model Which is 0.1% better than XGBOOST  model is used for stroke prediction.]** MLflow is integrated to track experiments, manage model versions, and facilitate deployment.

## Monitoring

Model performance and data drift are continuously monitored using Prometheus and Grafana. Key metrics, such as accuracy, precision, recall, and data drift indicators, are exported using the `metrics_exporter.py` script located in `src/monitoring/`. Grafana dashboards, configured in the `grafana/dashboards/` directory, provide visualizations of these metrics. Prometheus alert rules, defined in `prometheus/alert_rules.yml`, are set up to trigger alerts in case of performance degradation or data drift.

## API

A REST API is provided to serve the trained stroke prediction model, making it accessible for integration with other healthcare applications and systems.

*   **Framework**: Built using Flask, a lightweight and flexible Python web framework.
*   **Code Location**: API application code is located in `src/api/app.py`.
*   **Functionality**: The API exposes an endpoint that accepts patient data in JSON format as input and returns the model's stroke prediction as a JSON response.
*   **Usage**:  Detailed instructions on how to interact with the API, including request formats and response examples.

# Stroke Prediction API - Production Deployment Guide

This guide outlines the steps to deploy and use the Stroke Prediction API in a production environment using Docker and Docker Compose.

## 1. Building the Docker Image for Production

To build the Docker image for production, navigate to the project root directory in your terminal and run the following command:

```bash
docker-compose -f docker/docker-compose.yml build
```

This command uses the `docker-compose.yml` file to build the Docker images for all services defined in the file, including the `stroke-api`, `model-monitor`, `prometheus`, and `grafana` services. Docker Compose will use the `Dockerfile` in the `docker/` directory to build the `stroke-api` image, incorporating all necessary dependencies and the application code. This command creates production-ready Docker images.

## 2. Running the API in Production

To run the API in detached mode (in the background), use the following command in the project root directory:

```bash
docker-compose -f docker/docker-compose.yml up -d
```

The `-d` flag runs the containers in detached mode, meaning they will run in the background. To verify that the containers are running, you can use:

```bash
docker-compose -f docker/docker-compose.yml ps
```

This command will show the status of all services defined in your `docker-compose.yml` file, confirming that the API and its related services are running.

## 3. Accessing the Prediction API

Once the Docker containers are running, the Stroke Prediction API will be accessible at `http://localhost:8000/predict`. You can send prediction requests using `curl` or any other HTTP client.

Here's an example `curl` command to send a prediction request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
      "gender": "Male",
      "age": 65.0,
      "hypertension": 1,
      "heart_disease": 0,
      "ever_married": "Yes",
      "work_type": "Private",
      "residence_type": "Urban",
      "avg_glucose_level": 100.0,
      "bmi": 28.0,
      "smoking_status": "formerly smoked"
    }'
```

This command sends a POST request to the `/predict` endpoint with patient data in JSON format. The API will respond with a JSON object containing the stroke probability and prediction.

## 4. Monitoring the API

The production setup includes monitoring capabilities using Prometheus and Grafana:

*   **Prometheus Metrics:** Access Prometheus metrics at `http://localhost:9090/metrics`. Prometheus collects metrics from the `stroke-api` and `model-monitor` services, providing insights into API performance and model behavior.
*   **Grafana Dashboards:** Access Grafana dashboards at `http://localhost:3001` (or `http://localhost:<your_grafana_port>`). Grafana provides pre-configured dashboards for visualizing the collected metrics. Key dashboards include:
    *   **Prediction API Dashboard:**  Visualizes API request rates, latency, and prediction probabilities.
    *   **Model Performance Dashboard:**  Monitors model accuracy, F1-score, ROC AUC, and other performance metrics.
    *   **Data Drift Dashboard:** Tracks data drift metrics to detect potential data quality issues.
    *   **Model Explainability Dashboard:** (If implemented) Provides insights into model predictions and feature importance.

These dashboards are crucial for monitoring the health and performance of your production API.

## 5. Scaling the API

To handle increased traffic, you can scale the Stroke Prediction API horizontally by increasing the number of `stroke-api` containers. Use the `docker-compose scale` command:


```bash
docker-compose -f docker/docker-compose.yml scale stroke-api=3
```

This command will scale the `stroke-api` service to 3 instances. Docker Compose will automatically handle load balancing between these instances. For more advanced load balancing in a production environment, you might consider using a dedicated load balancer service in front of your Docker containers.

## 6. Updating the Model

To update the model in production, follow these steps:

1.  **Retrain the model:** Run the training pipeline using: `python src/pipeline/train_pipeline.py`. This will train a new model and save it as `latest_model.joblib` in the `models/` directory.
2.  **Rebuild Docker images:** After retraining, rebuild the Docker images to include the new model: `docker-compose -f docker/docker-compose.yml build`.
3.  **Redeploy the API:**  Restart the Docker containers to apply the changes and load the new model: `docker-compose -f docker/docker-compose.yml up -d`.

This process ensures that your production API is updated with the latest trained model.


## Installation

To set up the project and run it locally, follow these steps:

1.  **Clone the repository** (if you haven't already):
    ```bash
    git clone https://github.com/QsingularityAi/MLOps_Classification.git
    cd MLOps_Classification
    ```

2.  **Navigate to the project directory**:
    ```bash
    cd MLOps_Classification
    ```

3.  **Create a virtual environment** (recommended):
    ```bash
    python3 -m venv venv
    ```

4.  **Activate the virtual environment**:
    *   **On macOS/Linux**:
        ```bash
        source venv/bin/activate
        ```
    *   **On Windows**:
        ```bash
        venv\Scripts\activate
        ```

5.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## 🚀 Usage

### 1. Run the API

*   **Start the Flask API server**:
    ```bash
    python src/api/app.py
    ```
    The API will be running at `http://localhost:8000`.

    
### Example API Request

```python
import requests
import json

url = "http://localhost:5000/predict"

patient_data = {
    "gender": "Male",
    "age": 67,
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked"
}

response = requests.post(url, json=patient_data)
prediction = response.json()

print(f"Stroke Probability: {prediction['stroke_probability']}")
print(f"Risk Category: {prediction['risk_category']}")
```


### 2. Run the Monitoring Stack (Prometheus & Grafana)

*   **Start with Docker Compose**:
    Ensure Docker and Docker Compose are installed on your system and run:
    ```bash
    docker-compose -f docker/docker-compose.yml up --build
    ```
    This command starts Prometheus and Grafana in detached mode.

*   **Access Prometheus**:
    Open your web browser and go to `http://localhost:9090` to access the Prometheus dashboard.

*   **Access Grafana**:
    Open your web browser and go to `http://localhost:3000`.
    *   **Login**: Use the default credentials: `admin` for username and `admin` for password.
    *   **Import Dashboards**: Import the provided Grafana dashboards from the `grafana/dashboards/` directory to visualize model performance and monitoring metrics.

### 3. Explore with Jupyter Notebook

*   **Start Jupyter Notebook**:
    For interactive experimentation, navigate to the `notebook/` directory and run:
    ```bash
    jupyter notebook demo.ipynb
    ```
    This will open the Jupyter Notebook in your browser, allowing you to explore the data, run model training experiments, and test the API.

Before running the notebook, ensure you have Jupyter installed:
    ```bash
    pip install jupyter
    ```

1. **Clone the repository:**
   ```bash
   git clone https://github.com/QsingularityAi/MLOps_Classification.git
   cd MLOps_Classification
   ```
  

2. **Set up Python environment:**
   It is recommended to use a virtual environment to manage project dependencies. You can create and activate a virtual environment using `venv` or `conda`:

   ```bash
   # Using venv
   python3 -m venv venv
   source venv/bin/activate  # On Linux/macOS
   venv\Scripts\activate  # On Windows

   # Using conda
   conda create -n stroke_prediction python=3.8  # or your preferred Python version
   conda activate stroke_prediction
   ```

3. **Install dependencies:**
   Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure

The project repository is organized as follows:

```
MLOps_Classification/
├── configs/                     # Configuration files for models and monitoring
├── data/                        # Dataset and data related files
├── docker/                      # Docker configurations for containerization
├── grafana/                     # Grafana dashboards and provisioning configurations
├── models/                      # Trained models and related artifacts
├── notebook/                    # Jupyter notebooks for experimentation and demo
├── prometheus/                  # Prometheus configurations and alerting rules
├── src/                         # Source code for API, data processing, model, monitoring, and pipelines
│   ├── api/                     # API application using Flask
│   ├── data_processing/        # Scripts for data ingestion and preprocessing
│   ├── evaluation/             # Scripts for model evaluation
│   ├── model/                  # Scripts for model training and MLflow integration
│   ├── monitoring/             # Scripts for metrics exporter
│   └── pipeline/               # Scripts for defining pipelines (training, monitoring)
├── README.md                    # Project README file
├── requirements.txt             # Python dependencies
└── setup.py                     # Setup script for packaging
```

## Setup

The project uses `setup.py` for packaging and installation. While direct installation via `setup.py` is possible, it is generally recommended to manage dependencies using `requirements.txt` as described in the Installation section.

## Features

* End-to-end machine learning pipeline for stroke prediction.
* Data ingestion, preprocessing, feature engineering, model training, and evaluation.
* Model monitoring and drift detection using Prometheus and Grafana.
* REST API for model serving and integration.
* Dockerized deployment for scalability and reproducibility.
* Jupyter notebooks for experimentation and development.
* Integration with MLflow for model management and experiment tracking.

## Contributing

Contributions to the Brain Stroke Prediction Project are welcome! If you want to contribute, please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Implement your changes and ensure they are well-tested.
4.  Submit a pull request with a clear description of your changes.


## Contact

For questions or issues related to this project, please contact me.

## License

This project is open-source and available under the `GPL-3.0 license`. You are free to use, modify, and distribute this project for commercial and non-commercial purposes.  See the `LICENSE` file for the full license text.

## Contributing

We welcome contributions to the Full End to End ML Based Brain Stroke Prediction Project!  If you're interested in contributing, please take the following steps:

1.  **Fork the Repository:** Start by forking the project repository to your own GitHub account.
2.  **Create a Branch:**  Create a new branch for your contribution (e.g., `feature/new-feature` or `bugfix/fix-issue`).
3.  **Implement Your Changes:**  Make your changes, adhering to the project's coding style and best practices.
4.  **Test Your Changes:** Ensure your changes are thoroughly tested and do not introduce regressions.
5.  **Submit a Pull Request:**  Submit a pull request to the main repository with a clear title and description of your contribution.  Be sure to reference any related issues.

### Contribution Guidelines

*   Follow the existing code style and conventions.
*   Write clear and concise commit messages.
*   Provide tests for new features and bug fixes.
*   Keep pull requests focused on a single feature or bug fix.

## Features

*   **End-to-end machine learning pipeline for stroke prediction:**  Provides a complete pipeline from data ingestion to API deployment, covering all stages of the machine learning lifecycle.
*   **Data ingestion, preprocessing, feature engineering, model training, and evaluation:** Includes comprehensive scripts and notebooks for each stage of the data processing and model development workflow.
*   **Model monitoring and drift detection using Prometheus and Grafana:** Implements robust monitoring and observability using industry-standard tools to ensure model reliability in production.
*   **REST API for model serving and integration:** Exposes the trained model as a RESTful API, enabling easy integration with other healthcare applications and systems.
*   **Dockerized deployment for scalability and reproducibility:**  Utilizes Docker for containerization, ensuring consistent and scalable deployment across different environments.
*   **Jupyter notebooks for experimentation and development:** Provides Jupyter notebooks for interactive data exploration, model development, and experimentation.
*   **Integration with MLflow for model management and experiment tracking:** Leverages MLflow to track experiments, manage models, and ensure reproducibility of results.
