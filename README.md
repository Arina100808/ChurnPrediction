# Scalable Multi-Market Customer Churn Prediction
### End-to-End ML Engineering Solution

This project provides a scalable, containerized machine learning pipeline to predict customer churn across multiple markets. It is designed for automated deployment using cloud infrastructure, configuration-driven workflows, and continuous integration.

---

## Documentation Overview

- **README.md** – Provides setup instructions, infrastructure provisioning steps, and how to run the full pipeline.
- **[CLI Arguments & Data Requirements](docs/cli_arguments.md)** – Describes command-line arguments for training and prediction; and input data expectations.
- **[Training & Prediction Details](docs/training_prediction.md)** – Outlines the modeling pipeline, saved artifacts, and YAML configuration for per-market model customization.

---

## Project Structure

```
brenntag-churn/
├── artifacts/                 # Trained models, predictions, metrics per market
├── docker/
│   └── Dockerfile             # Container image definition
├── infra/                     # Infrastructure-as-code (IaC)
│   ├── ansible/
│   │   └── upload_model.yml   # Ansible playbook to upload model to S3
│   └── terraform/             # Terraform configuration for AWS infrastructure
│       ├── main.tf
│       ├── outputs.tf
│       ├── provider.tf
│       └── variables.tf       
├── scripts/                  
│   ├── train_default.sh       # Script to train model for default market
│   └── predict_default.sh     # Script to score model for default market
├── src/                       # Source code
│   ├── main.py                # CLI entry point for training/predicting
│   ├── data.py                # Data loading and preprocessing
│   ├── model.py               # Model training and evaluation
│   ├── predict.py             # Inference logic
│   ├── pipeline.py            # ML pipeline construction
│   ├── utils.py               # Shared utility functions
│   ├── config/
│   │   └── markets/           # YAML files with per-market parameters
│   └── data/
│       └── data.csv           # Input dataset
├── .github/workflows/ci.yml   # CI/CD pipeline config (GitHub)
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## Features

- This solution supports training and running models on a **per-market** basis (where "market" refers to a country or other regional identifier provided in the dataset). It is possible to train and evaluate models separately for each market, or to operate on the entire dataset without market-specific filtering when no market is specified.
- The code is organized into modular components and uses a command-line interface for training and prediction.
- A Dockerfile is provided to **containerize** the application for consistent execution across environments.
- Infrastructure for storing model artifacts is provisioned using **Terraform**, specifically by creating an S3 bucket on **AWS**.
- **Ansible** is used to automate the process of uploading trained model artifacts to the configured S3 bucket.
- A **CI/CD** pipeline is defined (GitHub) to run basic jobs on each commit, such as printing the current branch name and detecting secrets in the repository.
- The solution supports YAML-based configuration for market-specific model parameters such as number of estimators, depth, and learning rate.

---

## Getting Started


This section provides step-by-step instructions to:

- Run the solution using Docker
- Provision infrastructure on AWS using Terraform
- Upload model artifacts to S3 using Ansible
- Set up and verify the CI/CD pipeline with GitHub

All steps assume that the code has been cloned locally and required tools (e.g., Docker, Terraform, Ansible) are installed.

---

### Prerequisites

- Python 3.11+
- [Docker](https://www.docker.com/get-started/) installed and running locally
- [Terraform](https://developer.hashicorp.com/terraform/downloads)
- [Ansible](https://docs.ansible.com/) installed
- Git (for version control)
- An [AWS](https://console.aws.amazon.com/iam) account

---

### 1. Clone the Repository

```bash
git clone https://github.com/Arina100808/churn-prediction.git
cd churn-prediction
```

---

### 2. Train and Predict with Docker

The project can be run locally using Docker without needing to install Python dependencies manually.

#### Build the Docker Image

```bash
docker build -t churn-prediction -f docker/Dockerfile .
```

#### Train a Model

There are two options to train a model:

1. Run the provided shell script for all markets ("default" market):
  
```bash
sh scripts/train_default.sh
```

2. Run the Docker command directly, specifying the market (e.g., country "AB") and other parameters as needed:

```bash
docker run --rm -v "$PWD":/app churn-prediction \
  train --data src/data/data.csv --market AB
 ```

#### Predict Using the Trained Model

Analogously, there are two options for scoring new data with a trained model:

1. Use the provided shell script for the "default" market:
  
```bash
sh scripts/predict_default.sh
```

_For demonstration purposes, the same `data.csv` file is used during training and prediction. However, in practice, the prediction dataset should be a **separate file** containing **unseen data** (i.e., not used during training) to avoid data leakage._

2. Or run the Docker command directly, specifying the market and other parameters as needed:

```bash
docker run --rm -v "$PWD":/app churn-prediction \
  predict \
    --data src/data/data.csv \
    --model artifacts/CD/model_CD.joblib \
    --market CD
```

> * Refer to [CLI Arguments & Data Requirements](docs/cli_arguments.md) for a full description of available arguments and data formatting expectations for both training and prediction. 
> * For more details on the modeling pipeline, saved artifacts, and YAML configuration, refer to the [Training and Prediction Details](docs/training_prediction.md) file.
---

### 3. Provision AWS Infrastructure with Terraform

Terraform is used to create an S3 bucket for storing model artifacts.

#### Set Up AWS Credentials

Configure credentials using environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

#### Initialize Terraform

```bash
cd infra/terraform
terraform init
```

#### Apply Infrastructure

```bash
terraform apply
```

This will create an S3 bucket with the name defined in `variables.tf`. It outputs the name and ARN of the created S3 bucket to confirm successful provisioning.

---

### 4. Upload Model Artifacts with Ansible

Ansible is used to upload trained model artifacts to the S3 bucket provisioned by Terraform.

#### Modify the Upload Path (optional)

In `infra/ansible/upload_model.yml`, adjust the `src` parameter to point to the local model file, and the `object` parameter to specify the desired S3 object key (i.e., the file name in the S3 bucket).

#### Run the Playbook

```bash
cd ../../infra/ansible
ansible-playbook upload_model.yml
```

This uploads the trained model (e.g., `model_AB.joblib`) to the configured S3 bucket.

---

### 5. CI/CD Pipeline with GitHub

A `.github/workflows/ci.yml` file is included to trigger a pipeline on each commit. It contains:

- A job to print the branch name.
- Secret detection template.

To verify the pipeline:

1. Go to a GitHub project.
2. Click on **Actions > Commit name** to see the jobs.
3. Click into each job to see logs and results.

---

### Notes

- All secrets (e.g., AWS credentials) are excluded from version control via `.gitignore`.
- The trained models, predictions, metrics, and parameters are saved under `artifacts/` per market.
- Infrastructure is isolated per environment (default: `dev`), defined in `variables.tf`.


> * This project was created as part of a machine learning engineering assessment.  
> * The dataset and project structure are intended for evaluation purposes only and do not reflect production systems or real customer data.
