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
│   ├── raw/               ← downloaded M5 CSV files (git-ignored)
│   └── processed/         ← engineered feature table (git-ignored)
│
├── models/
│
├── notebooks/
│
├── scripts/
│   ├── 0.download_dataset.py   ← download raw M5 data from Kaggle
│   └── 1.build_features.py     ← build feature table → data/processed/features.parquet
│
├── src/
│   └── features/
│       ├── config.py            ← paths, column names, feature parameters
│       ├── data_loader.py       ← melt wide→long, merge calendar + prices
│       └── build_features.py    ← lag / rolling / price-elasticity / calendar features
│
├── tests/
│   └── test_build_features.py  ← unit tests for feature builders
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

After downloading, the raw data will be stored locally inside `data/raw/`:

```text
data/raw/
├── calendar.csv
├── sales_train_validation.csv
├── sales_train_evaluation.csv
├── sell_prices.csv
└── sample_submission.csv
```

---

## Feature Engineering (Pipeline 1)

This step reads the raw M5 data, engineers all features, and writes the processed
dataset to `data/processed/features.parquet`.

### What it produces

A single parquet file with one row per (product, store, day) and 46 columns:

| Feature group | Columns |
|---|---|
| Identifiers | `id`, `item_id`, `dept_id`, `cat_id`, `store_id`, `state_id` |
| Lag features | `sales_lag_1`, `sales_lag_7`, `sales_lag_14`, `sales_lag_28` |
| Rolling stats | `sales_roll_mean/std/max_{7,14,28}` |
| Price features | `sell_price`, `price_change`, `price_change_pct`, `price_roll_mean`, `price_discount_pct`, `price_elasticity` |
| Calendar | `date`, `day_of_week`, `is_weekend`, `week_of_year`, `month`, `year` |
| Event / SNAP | `is_event`, `event_name_1`, `event_type_1`, `snap_active` |

### Quick test (first 5 items, ~7 seconds)

Use this first to confirm everything works before running on the full dataset:

```bash
python scripts/1.build_features.py --sample-items 5
```

### Full dataset

```bash
python scripts/1.build_features.py
```

This processes all ~30,000 product-store series across ~1,900 days (~60 million rows).
Expect a few minutes and ~4–6 GB RAM usage.

The processed dataset will be saved to:

```text
data/processed/features.parquet
```

### Options

```bash
# custom output path
python scripts/1.build_features.py --output data/processed/my_features.parquet

# keep warm-up rows that have NaN lag/rolling values (dropped by default)
python scripts/1.build_features.py --keep-warmup
```

### Run tests

```bash
python -m pytest tests/
```

All 5 tests should pass. They verify lag correctness, no-leakage guarantees,
price feature logic, and per-series isolation.

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

### Step 1: Clone and set up

```bash
git clone https://github.com/Thilokya03/Dynamic-Pricing---Inventory-Forecasting-Engine.git
cd Dynamic-Pricing---Inventory-Forecasting-Engine
python -m venv .venv
# Windows:   .venv\Scripts\Activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### Step 2: Authenticate with Kaggle

```python
import kagglehub
kagglehub.login()
```

### Step 3: Download raw data

```bash
python scripts/0.download_dataset.py
```

### Step 4: Build the feature table  ← *you are here*

```bash
# quick test
python scripts/1.build_features.py --sample-items 5

# full dataset
python scripts/1.build_features.py
```

Output: `data/processed/features.parquet`

### Step 5: Train the forecasting model *(coming next)*

Train a LightGBM demand forecasting model using the feature table and track
experiments with MLflow.

### Step 6: Pricing optimization *(planned)*

Use demand forecasts, stock levels, and price history to recommend the best
price per product per day.

### Step 7: API and deployment *(planned)*

Expose predictions via a FastAPI REST API, containerize with Docker, and deploy
via GitHub Actions CI/CD.

---

## Technologies Used

| Tool | Purpose |
|---|---|
| Python 3.11 | Core language |
| pandas / numpy | Data processing and feature engineering |
| pyarrow | Parquet read/write |
| LightGBM *(planned)* | Demand forecasting model |
| MLflow *(planned)* | Experiment tracking |
| FastAPI *(planned)* | REST API |
| PostgreSQL *(planned)* | Inventory and sales database |
| Docker *(planned)* | Containerization |
| GitHub Actions | CI/CD |
| KaggleHub | Dataset download |

---

## Current Status

| Stage | Status |
|---|---|
| Repository setup | Done |
| Dataset download script | Done |
| Feature engineering pipeline | **Done** |
| LightGBM model training | Planned |
| MLflow experiment tracking | Planned |
| Pricing optimization engine | Planned |
| FastAPI backend | Planned |
| Docker containerization | Planned |
| Automated retraining (MLOps) | Planned |

---


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
