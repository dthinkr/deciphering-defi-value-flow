import duckdb
import pandas as pd
import polars as pl

from utils import load_config

from . import BaseWrapper

CONFIG = load_config()


class MotherDuckWrapper(BaseWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.client = duckdb.connect(f"md:?motherduck_token={CONFIG.MD_KEY}")

    def execute_query(self, query: str) -> pl.DataFrame:
        df = self.client.execute(query).df()
        return pl.from_pandas(df)

    def _get_date_trunc_expr(self, granularity: str) -> str:
        expressions = {
            "day": "DATE_TRUNC('day', TIMESTAMP 'epoch' + date * INTERVAL '1 second')",
            "week": "DATE_TRUNC('week', TIMESTAMP 'epoch' + date * INTERVAL '1 second')",
            "month": "DATE_TRUNC('month', TIMESTAMP 'epoch' + date * INTERVAL '1 second')",
            "year": "DATE_TRUNC('year', TIMESTAMP 'epoch' + date * INTERVAL '1 second')",
        }
        return expressions.get(granularity, f"Unsupported granularity: {granularity}")

    def get_token_distribution(self, token_name: str, granularity: str) -> pl.DataFrame:
        """Retrieve the distribution of a specific token across protocols over time at specified granularity."""
        result = self.get_aggregated_data(
            granularity=granularity, token_name=token_name
        )
        return result

    def get_protocol_data(self, protocol_name: str, granularity: str) -> pl.DataFrame:
        """Retrieve data for a specific protocol with granularity, calculating the average of sums."""
        result = self.get_aggregated_data(
            granularity=granularity, protocol_name=protocol_name
        )
        return result

    def get_protocol_tvl_distribution_on_day_with_names(
        self, unix_timestamp: int
    ) -> pl.DataFrame:
        query = f"""
        SELECT
            c.id,
            a.name,
            SUM(c.value_usd) AS total_usd_value
        FROM {CONFIG.TABLES["C"]} c
        JOIN {CONFIG.TABLES["A"]} a ON c.id = a.id
        WHERE CAST(to_timestamp(c.date) AS DATE) = '{pd.Timestamp(unix_timestamp, unit="s").date()}'
        GROUP BY c.id, a.name
        ORDER BY total_usd_value DESC
        """
        result = self.execute_query(query)
        return result

    """NEW IMPLEMENTATION"""

    def create_table(
        self, table_id: str, file_path=None, schema=None, clustering_fields=None
    ):
        if file_path:
            create_table_query = (
                f"CREATE TABLE {table_id} AS SELECT * FROM read_parquet('{file_path}');"
            )
            self.execute_query(create_table_query)

        elif schema:
            column_order = [
                "id",
                "chain_name",
                "date",
                "token_name",
                "quantity",
                "value_usd",
            ]
            # Create a dictionary for quick access to field types by name
            schema_dict = {field[0]: field[1] for field in schema}

            schema_sql_parts = []
            for column_name in column_order:
                duckdb_type = schema_dict.get(
                    column_name, "VARCHAR"
                )  # default to VARCHAR
                field_name = f'"{column_name}"'
                schema_sql_parts.append(f"{field_name} {duckdb_type}")

            schema_sql = ", ".join(schema_sql_parts)
            create_table_query = f"CREATE TABLE {table_id} ({schema_sql});"
            try:
                self.execute_query(create_table_query)
            except Exception as e:
                if "already exists" in str(e):
                    pass  # Table {table_id} already exists.
                else:
                    raise
        else:
            raise ValueError("Either file_path or schema must be provided.")

    def delete_table(self, table_id: str):
        """Deletes a table in MotherDuck."""
        drop_table_query = f"DROP TABLE IF EXISTS {table_id}"
        self.execute_query(drop_table_query)

    def get_table(self, table_id: str) -> str:
        """Returns the existence of a table in MotherDuck."""
        exists_query = f"SELECT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_id = '{table_id}')"
        try:
            exists = self.execute_query(exists_query).fetchone()[0]
            if exists:
                return table_id
        except Exception as e:
            raise RuntimeError(f"Failed to get table {table_id}") from e

    def _table_ref(self, table_id: str) -> str:
        """Helper method to create a table reference from a table name."""
        return table_id

    def append_parquet(self, file_path: str, table_id: str) -> None:
        append_query = (
            f"INSERT INTO {table_id} SELECT * FROM read_parquet('{file_path}');"
        )
        try:
            self.execute_query(append_query)
        except Exception:
            raise

    def get_day_entry_counts(self, table_id: str) -> pl.DataFrame:
        query = f"""
        SELECT
            DATE_TRUNC('day', TIMESTAMP 'epoch' + date * INTERVAL '1 second') AS aggregated_date,
            COUNT(*) AS entry_count
        FROM {self._table_ref(table_id)}
        GROUP BY aggregated_date
        ORDER BY aggregated_date
        """
        return self.execute_query(query)

    def detect_duplicates(self, table_id: str = CONFIG.TABLES["C"]) -> pl.DataFrame:
        raise NotImplementedError("Not implemented")

    def remove_duplicates(self, table_id: str = CONFIG.TABLES["C"]) -> None:
        raise NotImplementedError("Not implemented")
