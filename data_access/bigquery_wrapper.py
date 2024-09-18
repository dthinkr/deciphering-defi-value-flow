import pandas as pd
import polars as pl
import streamlit as st
from google.api_core import exceptions
from google.cloud import bigquery
from google.cloud.bigquery.client import Client
from google.cloud.bigquery.dataset import DatasetReference
from google.cloud.bigquery.table import Table
from google.cloud.exceptions import BadRequest
from google.oauth2 import service_account
from google.oauth2.service_account import Credentials

from utils import load_config

from . import BaseWrapper

CONFIG = load_config()


class BigQueryWrapper(BaseWrapper):
    def __init__(
        self, project: str = CONFIG.PROJECT_NAME, dataset: str = CONFIG.DATASET_NAME
    ) -> None:
        super().__init__()
        self.credentials: Credentials = (
            service_account.Credentials.from_service_account_info(
                st.secrets["gcp_service_account"]
            )
        )
        self.client: Client = bigquery.Client(
            # credentials=self.credentials,
            project=project
        )
        self.dataset_ref: DatasetReference = DatasetReference(project, dataset)

    def execute_query(self, query: str) -> pl.DataFrame:
        df = self.client.query(query).to_arrow().to_pandas()
        return pl.from_pandas(df)

    def _get_date_trunc_expr(self, granularity: str) -> str:
        expressions = {
            "day": "DATE(DATE(TIMESTAMP_SECONDS(CAST(ROUND(date) AS INT64))))",
            "week": "DATE_TRUNC(DATE(TIMESTAMP_SECONDS(CAST(ROUND(date) AS INT64))), WEEK(MONDAY))",
            "month": "DATE_TRUNC(DATE(TIMESTAMP_SECONDS(CAST(ROUND(date) AS INT64))), MONTH)",
            "year": "DATE_TRUNC(DATE(TIMESTAMP_SECONDS(CAST(ROUND(date) AS INT64))), YEAR)",
        }
        return expressions.get(granularity, expressions["day"])

    def _get_table(self, table_id: str) -> Table:
        table_ref = self.dataset_ref.table(table_id)
        return self.client._get_table(table_ref)

    def _get_table_schema(self, table_id: str) -> list:
        """Fetch the table schema"""
        table = self._get_table(table_id)
        return table.schema

    def _get_last_modified_time(self, table_id: str) -> str:
        """Get the last modified time of a table."""
        table = self._get_table(table_id)
        last_modified_time = table.modified
        # Convert to a more readable format, if desired
        formatted_time = last_modified_time.strftime("%Y-%m-%d %H:%M:%S %Z")
        return formatted_time

    def query_by_month(
        self, year: int, month: int, table: str = CONFIG.TABLES["C"]
    ) -> pd.DataFrame:
        """Query aggregated data by month."""
        # Construct the SQL query
        query = f"""
        SELECT
            id,
            chain_name,
            token_name,
            DATE_TRUNC(DATE(TIMESTAMP_SECONDS(CAST(ROUND(date) AS INT64))), MONTH) AS year_month,
            AVG(quantity) AS avg_quantity,
            AVG(value_usd) AS avg_value_usd
        FROM
            {self._table_ref(table)}
        WHERE
            EXTRACT(YEAR FROM TIMESTAMP_SECONDS(CAST(date AS INT64))) = {year} AND
            EXTRACT(MONTH FROM TIMESTAMP_SECONDS(CAST(date AS INT64))) = {month}
        GROUP BY
            id, chain_name, token_name, year_month
        """
        # Execute the query and return the DataFrame
        return self.execute_query(query)

    def get_protocol_tvl_distribution_on_day_with_names(
        self, unix_timestamp: int
    ) -> pl.DataFrame:
        # Convert the UNIX timestamp to a string representing the date in 'YYYY-MM-DD' format
        target_date = pd.Timestamp(unix_timestamp, unit="s").strftime("%Y-%m-%d")
        query = f"""
        SELECT
            c.id,
            a.name,
            SUM(c.value_usd) AS total_usd_value
        FROM {self._table_ref(CONFIG.TABLES["C"])} c
        JOIN {self._table_ref(CONFIG.TABLES["A"])} a ON c.id = a.id
        WHERE CAST(TIMESTAMP_SECONDS(c.date) AS DATE) = '{target_date}'
        GROUP BY c.id, a.name
        ORDER BY total_usd_value DESC
        """
        return self.execute_query(query)

    """REFACTORED IMPLEMENTATION"""

    def _table_ref(self, table_id: str) -> bigquery.table.TableReference:
        """Helper method to create a table reference from a table name."""
        return self.dataset_ref.table(table_id)

    def delete_table(self, table_id: str) -> None:
        """Deletes a table in BigQuery."""
        self.client.delete_table(self._table_ref(table_id))

    def get_table(self, table_id: str) -> Table:
        """Returns a Table instance for a given table_id."""
        return self.client.get_table(self._table_ref(table_id))

    def create_table(
        self, table_id: str, file_path=None, schema=None, clustering_fields=None
    ) -> Table:
        """Creates a new table in BigQuery from a Parquet file or a provided schema."""
        table_ref = self._table_ref(table_id)
        job_config = bigquery.LoadJobConfig()

        if file_path:
            job_config.source_format = bigquery.SourceFormat.PARQUET
            if schema:
                job_config.schema = schema
            else:
                job_config.autodetect = True
            job_config.write_disposition = bigquery.WriteDisposition.WRITE_EMPTY
        elif not schema:
            raise

        table = bigquery.Table(table_ref, schema=schema)
        if clustering_fields:
            table.clustering_fields = clustering_fields

        try:
            self.client.create_table(table)
        except BadRequest as e:
            raise e

        if file_path:
            with open(file_path, "rb") as source_file:
                job = self.client.load_table_from_file(
                    source_file, table_ref, job_config=job_config
                )
                job.result()

        return self.client.get_table(table_ref)

    def append_parquet(self, file_path: str, table_id: str) -> None:
        """Appends data from a Parquet file to a BigQuery table."""
        table_ref = self._table_ref(table_id)
        job_config = bigquery.LoadJobConfig()
        job_config.source_format = bigquery.SourceFormat.PARQUET
        job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND

        with open(file_path, "rb") as source_file:
            job = self.client.load_table_from_file(
                source_file, table_ref, job_config=job_config
            )
            job.result()

    def detect_duplicates(self, table_id: str = CONFIG.TABLES["C"]) -> pl.DataFrame:
        table_name = self._table_ref(table_id)
        query = f"""
        SELECT TO_JSON_STRING(t) AS json_representation, COUNT(*) AS duplicate_count
        FROM {table_name} t
        GROUP BY json_representation
        HAVING COUNT(*) > 1;
        """
        return self.execute_query(query)

    def remove_duplicates(
        self, table_id: str = CONFIG.TABLES["C"], limit: int = None
    ) -> None:
        raise NotImplementedError("Not implemented")
