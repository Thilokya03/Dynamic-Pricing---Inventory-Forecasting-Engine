import polars as pl

import scripts.create_silver_dataset as csd


def test_add_prices_info():
    sales_df = pl.DataFrame({
        "store_id": [1, 2, 3],
        "item_id": [101, 102, 103],
        "wm_yr_wk": [202301, 202302, 202303],
        "sales": [10, 20, 30],
    }).lazy()

    prices_df = pl.DataFrame({
        "store_id": [1, 2, 3],
        "item_id": [101, 102, 103],
        "wm_yr_wk": [202301, 202302, 202303],
        "sell_price": [5.0, 10.0, 15.0],
    }).lazy()

    result_df = csd.add_prices_info(
        sales_df,
        prices_df,
    ).collect()

    assert result_df.shape == (3, 5)

    assert result_df.columns == [
        "store_id",
        "item_id",
        "wm_yr_wk",
        "sales",
        "sell_price",
    ]

    assert result_df["sell_price"].to_list() == [
        5.0,
        10.0,
        15.0,
    ]


def test_add_prices_info_missing_price():
    sales_df = pl.DataFrame({
        "store_id": [1, 2, 3],
        "item_id": [101, 102, 103],
        "wm_yr_wk": [202301, 202302, 202303],
        "sales": [10, 20, 0],
    }).lazy()

    # Price for item 103 is intentionally missing.
    prices_df = pl.DataFrame({
        "store_id": [1, 2],
        "item_id": [101, 102],
        "wm_yr_wk": [202301, 202302],
        "sell_price": [5.0, 10.0],
    }).lazy()

    result_df = csd.add_prices_info(
        sales_df,
        prices_df,
    ).collect()

    assert result_df.shape == (3, 5)

    assert result_df["sell_price"].to_list() == [
        5.0,
        10.0,
        None,
    ]


def test_add_calendar_info():
    sales_df = pl.DataFrame({
        "store_id": [1, 2, 3],
        "item_id": [101, 102, 103],
        "d": ["d_1", "d_2", "d_3"],
        "sales": [10, 20, 30],
    }).lazy()

    calendar_df = pl.DataFrame({
        "d": ["d_1", "d_2", "d_3"],
        "date": [
            "2023-01-01",
            "2023-01-02",
            "2023-01-03",
        ],
        "wm_yr_wk": [
            202301,
            202301,
            202301,
        ],
    }).lazy()

    result_df = csd.add_calendar_info(
        sales_df,
        calendar_df,
    ).collect()

    assert result_df.shape == (3, 6)

    assert result_df["date"].to_list() == [
        "2023-01-01",
        "2023-01-02",
        "2023-01-03",
    ]

    assert result_df["wm_yr_wk"].to_list() == [
        202301,
        202301,
        202301,
    ]


def test_add_calendar_info_missing_calendar_row():
    sales_df = pl.DataFrame({
        "store_id": [1, 2, 3],
        "item_id": [101, 102, 103],
        "d": ["d_1", "d_2", "d_3"],
        "sales": [10, 20, 30],
    }).lazy()

    calendar_df = pl.DataFrame({
        "d": ["d_1", "d_2"],
        "date": [
            "2023-01-01",
            "2023-01-02",
        ],
        "wm_yr_wk": [
            202301,
            202301,
        ],
    }).lazy()

    result_df = csd.add_calendar_info(
        sales_df,
        calendar_df,
    ).collect()

    assert result_df.shape == (3, 6)

    # d_3 has no matching calendar row.
    assert result_df["date"].to_list() == [
        "2023-01-01",
        "2023-01-02",
        None,
    ]


def test_transform_sales_data():
    sales_df = pl.DataFrame({
            "id": [
                "item_101",
                "item_102",
                "item_103",
            ],
            "item_id": [101, 102, 103],
            "dept_id": [1, 2, 3],
            "cat_id": [1, 2, 3],
            "store_id": [1, 2, 3],
            "state_id": ["CA", "TX", "NY"],
            "d_1": [1, 2, 3],
            "d_2": [4, 5, 6],
            "d_3": [7, 8, 9],
            "d_4": [10, 11, 12],
        }).lazy()

    result_df = csd.transform_sales_data(
        sales_df
    ).collect()

    # 3 products × 4 day columns = 12 rows.
    assert result_df.shape == (12, 8)

    assert result_df.columns == [
        "id",
        "item_id",
        "dept_id",
        "cat_id",
        "store_id",
        "state_id",
        "d",
        "sales",
    ]

    assert set(result_df["d"].to_list()) == {
        "d_1",
        "d_2",
        "d_3",
        "d_4",
    }

    assert sorted(
        result_df["sales"].to_list()
    ) == list(range(1, 13))


def test_transform_sales_data_preserves_item_information():
    sales_df = pl.DataFrame({
        "id": ["item_101"],
        "item_id": [101],
        "dept_id": [1],
        "cat_id": [10],
        "store_id": [100],
        "state_id": ["CA"],
        "d_1": [5],
        "d_2": [7],
    }).lazy()

    result_df = csd.transform_sales_data(
        sales_df
    ).collect()

    assert result_df.shape == (2, 8)

    assert result_df["item_id"].to_list() == [
        101,
        101,
    ]

    assert result_df["store_id"].to_list() == [
        100,
        100,
    ]

    assert set(result_df["sales"].to_list()) == {
        5,
        7,
    }


def test_save_to_parquet(tmp_path):
    expected_df = pl.DataFrame({
        "item_id": [101, 102, 103],
        "dept_id": [1, 2, 3],
        "cat_id": [1, 2, 3],
        "store_id": [1, 2, 3],
        "state_id": ["CA", "TX", "NY"],
        "d_1": [1, 2, 3],
        "d_2": [4, 5, 6],
        "d_3": [7, 8, 9],
        "d_4": [10, 11, 12],
    })

    output_path = tmp_path / "test.parquet"

    csd.save_to_parquet(
        expected_df.lazy(),
        output_path,
    )

    assert output_path.exists()

    actual_df = pl.read_parquet(
        output_path
    )

    assert actual_df.equals(expected_df)


def test_save_to_parquet_with_nulls(tmp_path):
    expected_df = pl.DataFrame({
        "item_id": [101, 102, 103],
        "sales": [10, 20, 0],
        "sell_price": [5.0, 10.0, None],
    })

    output_path = (
        tmp_path
        / "test_with_nulls.parquet"
    )

    csd.save_to_parquet(
        expected_df.lazy(),
        output_path,
    )

    assert output_path.exists()

    actual_df = pl.read_parquet(
        output_path
    )

    assert actual_df.equals(expected_df)