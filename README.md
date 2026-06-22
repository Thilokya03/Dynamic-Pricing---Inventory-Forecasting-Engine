# Dynamic Pricing & Inventory Forecasting Engine

A Python-based machine learning project for building a **Dynamic Pricing and Inventory Forecasting Engine** using historical retail sales data. The project is designed to forecast product demand, analyze inventory behavior, and support better pricing decisions.

This project uses the **M5 Forecasting - Accuracy** dataset from Kaggle.

---

## Project Overview

Retail businesses need to understand future demand before making decisions about stock levels, promotions, and prices. Poor demand forecasting can cause:

* Overstocking
* Stockouts
* Lost sales
* Unnecessary storage costs
* Poor pricing decisions

This project aims to solve that problem by building a data pipeline and forecasting workflow that can later be extended into a full dynamic pricing system.

---

## Main Objectives

* Download and manage the M5 retail forecasting dataset
* Store raw data locally without pushing large files to GitHub
* Prepare a clean project structure for machine learning development
* Build scripts for repeatable data processing
* Use GitHub Actions CI to check code quality and script execution
* Prepare the foundation for future demand forecasting and pricing optimization

---

## Dataset

This project uses the **M5 Forecasting - Accuracy** dataset.

The dataset includes retail sales information such as:

* Historical product sales
* Calendar/date information
* Product selling prices
* Store and item-level information

Raw dataset files are not included in this repository because they are large. Each user should download the dataset locally using the provided script.

---

## Project Structure

```text
Dynamic-Pricing---Inventory-Forecasting-Engine/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── notebooks/
│
├── scripts/
│   └── 0.download_dataset.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Important Note About Dataset Files

Raw dataset files must not be pushed to GitHub.

The following folders are ignored by Git:

```text
data/raw/
data/processed/
Data/raw/
Data/processed/
Dataset/
```

This avoids GitHub file size errors caused by large CSV files such as:

```text
sales_train_validation.csv
sales_train_evaluation.csv
sell_prices.csv
```

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Thilokya03/Dynamic-Pricing---Inventory-Forecasting-Engine.git
cd Dynamic-Pricing---Inventory-Forecasting-Engine
```

---

### 2. Create a Virtual Environment

For Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate
```

For macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Kaggle Authentication

To download the M5 competition dataset, you need Kaggle authentication.

### Option 1: KaggleHub Login

Run this once in Python:

```python
import kagglehub
kagglehub.login()
```

Then follow the login instructions.

### Option 2: Kaggle API Token

You can also use a Kaggle API token.

Download your Kaggle API token from your Kaggle account settings and configure it before running the dataset download script.

---

## Download Dataset

Run:

```bash
python scripts/0.download_dataset.py
```

After downloading, the raw data should be stored locally inside:

```text
data/raw/m5-forecasting-accuracy/
```

Expected files include:

```text
calendar.csv
sales_train_validation.csv
sales_train_evaluation.csv
sell_prices.csv
sample_submission.csv
```

---

## Run Python Syntax Check

To check whether Python files compile correctly:

```bash
python -m compileall scripts models notebooks
```

---

## GitHub Actions CI

This repository includes a CI workflow inside:

```text
.github/workflows/ci.yml
```

The CI workflow is used to:

* Set up Python
* Install dependencies
* Check Python syntax
* Make sure large raw dataset files are not committed
* Run available project scripts
* Run tests if a `tests/` folder is added later

The workflow can also use cache to speed up dataset-related tasks.

---

## Recommended Workflow

### Step 1: Download raw data

```bash
python scripts/0.download_dataset.py
```

### Step 2: Explore the dataset

Use notebooks inside the `notebooks/` folder for data understanding and experiments.

### Step 3: Preprocess data

Create preprocessing scripts and save cleaned outputs inside:

```text
data/processed/
```

### Step 4: Build forecasting models

Train forecasting models and save trained model files inside:

```text
models/
```

### Step 5: Add pricing logic

Use demand forecasts, stock levels, and price history to support dynamic pricing decisions.

---

## Planned Features

* Data preprocessing pipeline
* Exploratory data analysis notebooks
* Demand forecasting models
* Inventory forecasting
* Price elasticity analysis
* Dynamic pricing recommendation logic
* Model evaluation reports
* Automated CI checks

---

## Technologies Used

* Python
* KaggleHub
* Jupyter Notebook
* GitHub Actions
* Git
* Machine Learning / Forecasting libraries

---

## Current Status

This project is currently in the initial setup stage.

Completed:

* Basic repository structure
* Dataset download script
* `.gitignore` setup for large dataset files
* Python dependency file
* GitHub Actions CI setup

Next steps:

* Add data preprocessing scripts
* Add exploratory data analysis notebooks
* Build baseline forecasting model
* Add model evaluation metrics
* Extend the project toward dynamic pricing optimization

---

## License

This project is for academic and learning purposes. A license can be added later depending on the final project requirements.
