from pathlib import Path
import polars as pl


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# data/raw
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SILVER_DATA_DIR = PROJECT_ROOT / "data" / "silver"

def transform_sales_data(sales_df: pl.LazyFrame) -> pl.LazyFrame:
    # Product and location columns to keep
    id_columns = [
        'id',
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

    sales_df = sales_df.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        variable_name="d",
        value_name="sales"
    )
    return sales_df


def add_calendar_info(sales_df: pl.LazyFrame, calendar_df: pl.LazyFrame) -> pl.LazyFrame:
    result_df = sales_df.join(
        calendar_df,
        on = ['d'],
        how = 'left'
        
    )
    return result_df


def add_prices_info(sales_df: pl.LazyFrame, prices_df: pl.LazyFrame) -> pl.LazyFrame:
    result_df = sales_df.join(
        prices_df,
        on = ['store_id','item_id', 'wm_yr_wk'],
        how = 'left'
    )
    return result_df


def save_to_parquet(df: pl.LazyFrame, filename: str):
    SILVER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = SILVER_DATA_DIR / f"{filename}.parquet"

    df.sink_parquet(
        output_path,
        compression="snappy",
        engine="streaming",
    )

def main():
    
    print("Loading raw data...")
    
    print("--" * 20)
    
    print("Loading raw data...")
    calendar_df = pl.scan_csv(str(RAW_DATA_DIR / "calendar.csv"))
    sales_eval_df = pl.scan_csv(str(RAW_DATA_DIR / "sales_train_evaluation.csv"))
    prices_df = pl.scan_csv(str(RAW_DATA_DIR / "sell_prices.csv"))
    print("Raw data loaded successfully.")
    
    print("--" * 20)
    
    print("Transforming sales data...")
    new_sales_eval_df = transform_sales_data(sales_eval_df)

    
    print("Adding calendar info...")
    new_sales_eval_df = add_calendar_info(new_sales_eval_df, calendar_df)
    
    print("Adding price info...")
    new_sales_eval_df = add_prices_info(new_sales_eval_df, prices_df)
    
    print("--" * 20)
    
    print("Saving transformed data to silver dataset...")
    save_to_parquet(
        new_sales_eval_df,
        "sales"
        )

    print("Silver dataset created successfully at:", SILVER_DATA_DIR)
    

    


if __name__ == "__main__":
    main()

