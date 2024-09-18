from unittest.mock import MagicMock

import polars as pl
import pytest

from data_access import BigQueryWrapper, MotherDuckWrapper

test_df = pl.DataFrame({"id": [118, 119], "qty": [10, 20], "usd": [1000, 2000]})


@pytest.fixture
def bigquery_client():
    client = BigQueryWrapper()
    client.get_token_distribution = MagicMock(return_value="mocked dataframe")
    client.get_protocol_data = MagicMock(return_value="mocked dataframe")
    client.compare_periods = MagicMock(return_value=test_df)
    return client


@pytest.fixture
def motherduck_client():
    client = MotherDuckWrapper()
    client.get_token_distribution = MagicMock(return_value="mocked dataframe")
    client.get_protocol_data = MagicMock(return_value="mocked dataframe")
    client.compare_periods = MagicMock(return_value=test_df)
    return client


def test_get_token_distribution_bigquery(bigquery_client):
    df = bigquery_client.get_token_distribution("USDC", "month")
    assert df == "mocked dataframe"


def test_get_protocol_data_bigquery(bigquery_client):
    df = bigquery_client.get_protocol_data("MakerDAO", "month")
    assert df == "mocked dataframe"


def test_compare_periods_bigquery(bigquery_client):
    df = bigquery_client.compare_periods("2023")
    assert df.equals(test_df, null_equal=True), "DataFrames are not the same"


def test_get_token_distribution_duckdb(motherduck_client):
    df = motherduck_client.get_token_distribution("USDC", "month")
    assert df == "mocked dataframe"


def test_get_protocol_data_duckdb(motherduck_client):
    df = motherduck_client.get_protocol_data("MakerDAO", "month")
    assert df == "mocked dataframe"


def test_compare_periods_duckdb(motherduck_client):
    df = motherduck_client.compare_periods("2023-01-24", "month")
    assert df.equals(test_df, null_equal=True), "DataFrames are not the same"


def test_compare_periods_consistency(bigquery_client, motherduck_client):
    df_big_query = bigquery_client.compare_periods("2024-04-10")
    df_mother_duck = motherduck_client.compare_periods("2024-04-10")

    sorted_df_big_query = df_big_query.sort("qty").filter(pl.col("id") == 118)
    sorted_df_mother_duck = df_mother_duck.sort("qty").filter(pl.col("id") == 118)

    assert sorted_df_mother_duck.equals(
        sorted_df_big_query, null_equal=True
    ), "DataFrames are not the same"
