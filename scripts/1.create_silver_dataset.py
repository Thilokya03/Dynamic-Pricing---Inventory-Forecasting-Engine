from pathlib import Path
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# data/raw
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SILVER_DATA_DIR = PROJECT_ROOT / "data" / "silver"

def transform_sales_data(sales_df: DataFrame) -> DataFrame:
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

    sales_df = sales_df.unpivot(
        ids = id_columns,
        values = day_columns,
        variableColumnName = "d",
        valueColumnName = "sales"
    )
    return sales_df


def add_calendar_info(sales_df: DataFrame, calendar_df: DataFrame) -> DataFrame:
    result_df = sales_df.join(
        calendar_df,
        on = ['d'],
        how = 'left'
        
    )
    return result_df


def add_price_info(sales_df: DataFrame, sell_prices_df: DataFrame) -> DataFrame:
    result_df = sales_df.join(
        sell_prices_df,
        on = ['store_id','item_id', 'wm_yr_wk'],
        how = 'left'
    )
    

    return result_df

def save_to_parquet(df: DataFrame, filename: str):
    SILVER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SILVER_DATA_DIR / filename
    df.write.mode('overwrite').partitionBy('year').parquet(str(output_path))


def main():
    
    spark = SparkSession.builder.master("local[*]").appName("SilverDataset").getOrCreate()
    print("Spark session created.")
    
    print("--" * 20)
    print("Loading raw data...")
    
    print("--" * 20)
    
    print("Loading raw data...")
    calendar_df = spark.read.option("header", "true").csv(str(RAW_DATA_DIR / "calendar.csv"), inferSchema=True)
    sales_eval_df = spark.read.option("header", "true").csv(str(RAW_DATA_DIR / "sales_train_evaluation.csv"), inferSchema=True)

    prices_df = spark.read.option("header", "true").csv(str(RAW_DATA_DIR / "sell_prices.csv"), inferSchema=True)
    print("Raw data loaded successfully.")
    
    print("--" * 20)
    
    print("Transforming sales data...")
    new_sales_eval_df = transform_sales_data(sales_eval_df)

    
    print("Adding calendar info...")
    new_sales_eval_df = add_calendar_info(new_sales_eval_df, calendar_df)
    
    print("Adding price info...")
    new_sales_eval_df = add_price_info(new_sales_eval_df, prices_df)
    
    print("--" * 20)
    
    print("Saving transformed data to silver dataset...")
    save_to_parquet(new_sales_eval_df, "sales")

    print("Silver dataset created successfully at:", SILVER_DATA_DIR)
    
    print("--" * 20)
    spark.stop()
    print("Spark session stopped.")
    


if __name__ == "__main__":
    main()

