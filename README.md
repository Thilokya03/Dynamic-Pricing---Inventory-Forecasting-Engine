# Dynamic Pricing & Inventory Forecasting Engine

[![Python CI](https://github.com/Thilokya03/Dynamic-Pricing---Inventory-Forecasting-Engine/actions/workflows/ci.yml/badge.svg)](https://github.com/Thilokya03/Dynamic-Pricing---Inventory-Forecasting-Engine/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Polars](https://img.shields.io/badge/Data%20Processing-Polars-orange)
![License](https://img.shields.io/badge/License-MIT-green)

A retail data engineering and machine learning project built around the **M5 Forecasting - Accuracy** dataset. The long-term goal is to combine demand forecasting, inventory forecasting, price-response analysis, and dynamic pricing recommendations in one reproducible pipeline.

The project is currently focused on a scalable **Raw → Silver** data pipeline using **Polars LazyFrames** and Parquet.

---

## Project Goals

Retailers need accurate demand estimates before making inventory and pricing decisions. Poor forecasting can lead to stockouts, excess inventory, unnecessary holding costs, lost sales, ineffective promotions, and poor pricing decisions.

The long-term architecture is:

```text
Historical Sales
      +
Calendar / Events
      +
Selling Prices
      ↓
Data Engineering
      ↓
Demand Forecasting
      ↓
Inventory Forecasting
      ↓
Price Optimization
      ↓
Dynamic Pricing Recommendations
```

---

## Current Architecture

The current implemented pipeline creates a **Silver dataset** from the raw M5 competition files.

```text
data/raw/
├── sales_train_evaluation.csv
├── calendar.csv
└── sell_prices.csv
        │
        ▼
Polars LazyFrame Pipeline
        │
        ├── Wide → Long sales transformation
        ├── Calendar join
        ├── Selling-price join
        │
        ▼
data/silver/
└── sales.parquet
```

The transformation pipeline:

1. reads CSV files lazily with `pl.scan_csv()`,
2. converts `d_1 ... d_1941` sales columns from wide format to long format,
3. joins calendar data using `d`,
4. joins price data using `store_id`, `item_id`, and `wm_yr_wk`,
5. writes the resulting dataset as compressed Parquet.

---

## Dataset

This project uses the **M5 Forecasting - Accuracy** competition dataset.

The important source files used by the current Silver pipeline are:

| File | Purpose |
|---|---|
| `sales_train_evaluation.csv` | Daily unit sales by item and store |
| `calendar.csv` | Dates, weeks, events, and SNAP indicators |
| `sell_prices.csv` | Weekly selling prices by item and store |

Large dataset files are intentionally not committed to Git.

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
│   │   ├── calendar.csv
│   │   ├── sales_train_evaluation.csv
│   │   └── sell_prices.csv
│   │
│   └── silver/
│       └── sales.parquet
│
├── notebooks/
│
├── scripts/
│   ├── __init__.py
│   ├── download_dataset.py
│   └── create_silver_dataset.py
│
├── tests/
│   └── test_silver_dataset.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

> `data/raw/` and generated datasets should remain local and should not be committed to Git.

---

## Tech Stack

| Technology | Usage |
|---|---|
| Python | Main programming language |
| Polars | Lazy and memory-efficient data transformation |
| PyArrow | Parquet ecosystem support |
| DuckDB | Local analytical/query experimentation |
| Pandas / NumPy | Supporting data analysis |
| PySpark | Big-data/distributed-processing experimentation |
| Pytest | Unit testing |
| GitHub Actions | Continuous integration |
| KaggleHub | M5 competition dataset download |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Thilokya03/Dynamic-Pricing---Inventory-Forecasting-Engine.git
cd Dynamic-Pricing---Inventory-Forecasting-Engine
```

### 2. Create a virtual environment

#### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install project dependencies

```bash
pip install -r requirements.txt
```

The dataset download script also uses `kagglehub`. If it is not yet present in `requirements.txt`, install it with:

```bash
pip install kagglehub
```

---

## Kaggle Authentication

The raw dataset comes from the Kaggle **M5 Forecasting - Accuracy** competition.

Authenticate with KaggleHub before downloading the dataset:

```bash
python -c "import kagglehub; kagglehub.login()"
```

For CI, the repository workflow can use a `KAGGLE_API_TOKEN` GitHub Actions secret.

---

## Download the Dataset

Run:

```bash
python scripts/download_dataset.py
```

The Silver pipeline expects these files under `data/raw/`:

```text
data/raw/
├── calendar.csv
├── sales_train_evaluation.csv
└── sell_prices.csv
```

---

## Build the Silver Dataset

Run:

```bash
python scripts/create_silver_dataset.py
```

The current processing flow is:

```text
sales_train_evaluation.csv
        │
        ▼
transform_sales_data()
        │
        ├── keep identifier columns
        └── unpivot d_* columns
        │
        ▼
add_calendar_info()
        │
        └── left join on d
        │
        ▼
add_prices_info()
        │
        └── left join on
            store_id + item_id + wm_yr_wk
        │
        ▼
save_to_parquet()
        │
        ▼
Silver Parquet Dataset
```

The intended output is:

```text
data/silver/sales.parquet
```

---

## Silver Dataset Schema

The Silver dataset contains four main groups of columns.

### Product and location identifiers

```text
id
item_id
dept_id
cat_id
store_id
state_id
```

### Sales

```text
d
sales
```

Each original `d_*` column becomes a row in the long-format dataset.

### Calendar information

Examples include:

```text
date
wm_yr_wk
weekday
wday
month
year
event_name_1
event_type_1
event_name_2
event_type_2
snap_CA
snap_TX
snap_WI
```

### Price information

```text
sell_price
```

Price data is matched at the `(store_id, item_id, wm_yr_wk)` level.

---

## Missing Values

Some missing values in the Silver dataset are expected.

For example:

- event columns are null on normal non-event days,
- `sell_price` may be null for periods in which an item/store pair has no price record.

The Silver layer should preserve source information instead of automatically treating an unknown price as a real price of `0`.

Model-specific filtering and imputation should be performed later when building the Gold/modeling dataset.

---

## Running Tests

Run all tests with:

```bash
python -m pytest tests -v
```

The Silver-pipeline tests cover:

- sales wide-to-long transformation,
- preservation of product/store identifiers,
- calendar joins,
- missing calendar matches,
- price joins,
- missing price matches,
- Parquet writing, and
- Parquet null-value preservation.

---

## Continuous Integration

GitHub Actions is configured in:

```text
.github/workflows/ci.yml
```

The CI workflow runs on pushes and pull requests to `main` and performs checks such as:

- installing dependencies,
- Python syntax compilation,
- checking that raw M5 CSV files are not committed,
- restoring/downloading dataset cache when configured, and
- running the Pytest suite.

---

## Data Engineering Strategy

This repository follows a layered data approach.

### Raw / Bronze

Original source data downloaded from Kaggle.

```text
data/raw/
```

### Silver

Cleaned and joined analytical dataset.

```text
data/silver/
```

The current project work is focused here.

### Gold — Planned

Model-ready datasets with features such as:

- lagged demand,
- rolling demand statistics,
- calendar features,
- event features,
- price-change features,
- price elasticity features,
- availability indicators, and
- inventory-related features.

---

## Roadmap

| Stage | Status |
|---|---|
| Repository setup | ✅ Done |
| M5 dataset download pipeline | ✅ Implemented |
| Raw → Silver transformation | ✅ Implemented |
| Polars LazyFrame processing | ✅ Implemented |
| Silver-pipeline unit tests | ✅ Implemented |
| Gold feature engineering | 🚧 Next |
| Demand forecasting baseline | ⏳ Planned |
| Model evaluation | ⏳ Planned |
| Inventory forecasting / optimization | ⏳ Planned |
| Price elasticity modeling | ⏳ Planned |
| Dynamic pricing engine | ⏳ Planned |
| API service | ⏳ Planned |
| Containerization / deployment | ⏳ Planned |
| Automated model retraining | ⏳ Planned |

---

## Planned Modeling Workflow

```text
Silver Dataset
      ↓
Gold Feature Engineering
      ↓
Train / Validation Split
      ↓
Baseline Demand Forecast
      ↓
Model Evaluation
      ↓
Demand + Price Response Modeling
      ↓
Inventory Constraints
      ↓
Price Optimization
      ↓
Recommended Price
```

Forecasting features should be designed to avoid using future sales information and therefore prevent data leakage.

---

## Development Commands

### Run tests

```bash
python -m pytest tests -v
```

### Compile Python files

```bash
python -m compileall scripts notebooks
```

### Download data

```bash
python scripts/download_dataset.py
```

### Run the Silver pipeline

```bash
python scripts/create_silver_dataset.py
```

---

## Important Data Rules

Do not commit:

- raw M5 CSV files,
- generated Silver/Gold datasets,
- local virtual environments,
- credentials,
- Kaggle tokens, or
- `.env` files.

Large datasets should remain local or be stored in external data/artifact storage.

---

## Contributing

A typical contribution workflow is:

```bash
git checkout -b feature/my-feature
git add .
git commit -m "Add my feature"
git push origin feature/my-feature
```

Then create a pull request against `main`.

Before submitting a pull request, run:

```bash
python -m pytest tests -v
```

---

## Future Improvements

Planned improvements include:

- partitioned Parquet datasets for faster filtering,
- Gold-layer feature engineering,
- demand forecasting baselines,
- time-series-aware validation,
- price elasticity estimation,
- inventory-aware pricing rules,
- experiment tracking,
- model versioning,
- FastAPI prediction endpoints,
- Docker containerization,
- scheduled retraining, and
- production monitoring.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Author

**Thilokya03**

GitHub: [@Thilokya03](https://github.com/Thilokya03)

---

## Project Status

This project is under active development. The current focus is creating a reliable and scalable data foundation before moving to forecasting and dynamic pricing models.
