# End to End Machine Learning Pipeline for Brain Stroke Prediction classfication Project

This project is aimed at predicting the likelihood of brain stroke based on patient data.

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Data](#data)
- [Model](#model)
- [Monitoring](#monitoring)
- [API](#api)
- [Setup](#setup)
- [Usage](#usage)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Setup](#setup-1)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Introduction

**Stroke is a leading cause of death and disability worldwide.**  Rapid and accurate prediction of stroke risk is essential for enabling preventative measures and timely medical intervention. This project aims to develop a machine learning-based stroke prediction system to assist healthcare professionals in identifying individuals at high risk of stroke.

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

3.  **Feature Engineering**:
    *   Creating new features from existing data or transforming features to improve model performance.
    *   This may include scaling numerical features, encoding categorical variables, and generating interaction features.
    *   Feature engineering scripts are located in `src/feature_engineering/`.

4.  **Model Development and Training**:
    *   Selecting an appropriate machine learning model for stroke prediction (e.g., Logistic Regression, Random Forest, Gradient Boosting Machines).
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

*   **id**: Unique identifier for each patient.
*   **gender**: Patient's gender (Male, Female, Other).
*   **age**: Patient's age in years.
*   **hypertension**: Whether the patient has hypertension (0 or 1).
*   **heart_disease**: Whether the patient has heart disease (0 or 1).
*   **ever_married**: Whether the patient has ever been married (No or Yes).
*   **work_type**: Type of work the patient does (e.g., Private, Self-employed, Govt_job).
*   **Residence_type**: Type of residence (Rural or Urban).
*   **avg_glucose_level**: Average glucose level in blood.
*   **bmi**: Body mass index.
*   **smoking_status**: Patient's smoking status (e.g., never smoked, formerly smoked, smokes).
*   **stroke (Target Variable)**: Whether the patient had a stroke (1) or not (0).

## Model

The project utilizes a machine learning model to predict the likelihood of stroke based on the patient features described above.  The model is trained and evaluated using scripts in the `src/model/` directory. **[Specify Random forest Model Which is 0.1% better than XGBOOST  model is used for stroke prediction.]** MLflow is integrated to track experiments, manage model versions, and facilitate deployment.

## Monitoring

Model performance and data drift are continuously monitored using Prometheus and Grafana. Key metrics, such as accuracy, precision, recall, and data drift indicators, are exported using the `metrics_exporter.py` script located in `src/monitoring/`. Grafana dashboards, configured in the `grafana/dashboards/` directory, provide visualizations of these metrics. Prometheus alert rules, defined in `prometheus/alert_rules.yml`, are set up to trigger alerts in case of performance degradation or data drift.

## API

A REST API is provided to serve the trained stroke prediction model, making it accessible for integration with other healthcare applications and systems.

*   **Framework**: Built using Flask, a lightweight and flexible Python web framework.
*   **Code Location**: API application code is located in `src/api/app.py`.
*   **Functionality**: The API exposes an endpoint that accepts patient data in JSON format as input and returns the model's stroke prediction as a JSON response.
*   **Usage**:  Detailed instructions on how to interact with the API, including request formats and response examples, can be found in the `src/api/README.md` file *(Note: API documentation file needs to be created)*.

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

## Usage

### 1. Run the API

*   **Start the Flask API server**:
    ```bash
    python src/api/app.py
    ```
    The API will be running at `http://localhost:5000`.

*   **API Documentation**:
    For detailed information on API endpoints, request formats, and response examples, refer to the `src/api/README.md` file. *(Note: API documentation file needs to be created)*

### 2. Run the Monitoring Stack (Prometheus & Grafana)

*   **Start with Docker Compose**:
    Ensure Docker and Docker Compose are installed on your system. Navigate to the `docker/` directory and run:
    ```bash
    docker-compose up -d
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
│   ├── feature_engineering/    # Scripts for feature engineering (if any)
│   ├── model/                  # Scripts for model training and MLflow integration
│   ├── monitoring/             # Scripts for metrics exporter
│   └── pipeline/               # Scripts for defining pipelines (training, monitoring)
├── tests/                       # Test scripts (if any)
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
