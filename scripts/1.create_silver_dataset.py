from pathlib import Path
import pandas as pd
import os


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# data/raw
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SILVER_DATA_DIR = PROJECT_ROOT / "data" / "silver"

def transform_sales_data(sales_df: pd.DataFrame) -> pd.DataFrame:
    # Product and location columns to keep
    id_columns = [
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
    ]

    # Select only columns such as d_1, d_2, d_3, ...
    day_columns = [
        column
        for column in sales_df.columns
        if column.startswith("d_")
    ]

    long_df = sales_df.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )

    return long_df


def add_calendar_info(sales_df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:

    result_df = sales_df.merge(
        calendar_df[["d", "date", "wm_yr_wk"]],
        on="d",
        how="left"
    )
    result_df["date"] = pd.to_datetime(result_df["date"])

    return result_df


def add_sales_info(sales_df: pd.DataFrame, sell_prices_df: pd.DataFrame) -> pd.DataFrame:

    result_df = sales_df.merge(
        sell_prices_df,
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left"
    )

    return result_df

def save_to_csv(df: pd.DataFrame, filename: str):
    SILVER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(SILVER_DATA_DIR / filename, index=False)
     

if __name__ == "__main__":
    print("Loading raw data...")
    calendar_df = pd.read_csv(RAW_DATA_DIR / "calendar.csv")
    sales_evaluation_df = pd.read_csv(RAW_DATA_DIR / "sales_train_evaluation.csv")
    sales_validation_df = pd.read_csv(RAW_DATA_DIR / "sales_train_validation.csv")
    sell_prices_df = pd.read_csv(RAW_DATA_DIR / "sell_prices.csv")
    
    print("Transforming sales data...")
    new_sales_validation_df = transform_sales_data(sales_validation_df)
    new_sales_evaluation_df = transform_sales_data(sales_evaluation_df)
    
    print("Adding calendar info...")
    new_sales_validation_df = add_calendar_info(new_sales_validation_df, calendar_df)
    new_sales_evaluation_df = add_calendar_info(new_sales_evaluation_df, calendar_df)
    
    print("Adding sales info...")
    new_sales_validation_df = add_sales_info(new_sales_validation_df, sell_prices_df)
    new_sales_evaluation_df = add_sales_info(new_sales_evaluation_df, sell_prices_df)
    
    print("Saving transformed data to silver dataset...")
    save_to_csv(new_sales_validation_df, "new_sales_validation.csv")
    save_to_csv(new_sales_evaluation_df, "new_sales_evaluation.csv")
    
    print("Silver dataset created successfully at:", SILVER_DATA_DIR)
    
    