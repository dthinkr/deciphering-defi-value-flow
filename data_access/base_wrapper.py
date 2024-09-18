from datetime import datetime
from typing import List, Optional, Union

import polars as pl
from dateutil.relativedelta import relativedelta

from utils import load_config

CONFIG = load_config()


class BaseWrapper:
    def __init__(self):
        self.granularity_map = {
            "day": relativedelta(days=1),
            "week": relativedelta(weeks=1),
            "month": relativedelta(months=1),
            "year": relativedelta(years=1),
        }

    def get_dataframe(self, table_id: str, limit: Optional[int] = None) -> pl.DataFrame:
        table_id = self._table_ref(table_id)
        if limit is not None:
            query = f"SELECT * FROM {table_id} LIMIT {limit}"
        else:
            query = f"SELECT * FROM {table_id}"
        return self.execute_query(query)

    def get_all_protocol_slugs(self, desc: bool = True) -> pl.DataFrame:
        table_id = self._table_ref(CONFIG.TABLES["A"])
        order = "DESC" if desc else "ASC"
        query = f"""
        SELECT slug
        FROM (
            SELECT slug, MAX(tvl) as max_tvl
            FROM {table_id}
            GROUP BY slug
        ) AS subquery
        ORDER BY max_tvl {order}
        """
        return self.execute_query(query)

    def get_token_frequency(self, table_id: str) -> pl.DataFrame:
        """Retrieve the frequency of each token name from a specified table."""
        table_id = self._table_ref(table_id)
        query = f"""
        SELECT token_name, COUNT(*) as frequency
        FROM {table_id}
        GROUP BY token_name
        ORDER BY frequency DESC
        """
        return self.execute_query(query)

    def get_unique_token_names(
        self, table_id: str = CONFIG.TABLES["C"]
    ) -> pl.DataFrame:
        """Fetch unique token names from a specified table."""
        table_id = self._table_ref(table_id)
        query = f"SELECT DISTINCT token_name FROM {table_id}"
        return self.execute_query(query)

    def get_latest_dates_foreach_id_chain_token(self) -> pl.DataFrame:
        table_id = self._table_ref(CONFIG.TABLES["C"])
        query = f"""
        SELECT id, chain_name, token_name, MAX(date) AS latest_date
        FROM {table_id}
        GROUP BY id, chain_name, token_name
        """
        return self.execute_query(query)

    def get_data_by_protocol_ids(self, protocol_ids: list) -> pl.DataFrame:
        table_id = self._table_ref(CONFIG.TABLES["C"])
        ids_string = "', '".join(
            protocol_ids
        )  # Create a string of IDs separated by ', '
        query = f"""
        SELECT *
        FROM {table_id}
        WHERE id IN ('{ids_string}')
        """
        return self.execute_query(query)

    def get_protocol_info_by_id_or_name(
        self, identifier: Union[str, List[str]]
    ) -> pl.DataFrame:
        table_id = self._table_ref(CONFIG.TABLES["A"])
        if not isinstance(identifier, list):
            identifier = [identifier]

        filtered_identifiers = [str(id).lower() for id in identifier if id is not None]

        identifier_list = "', '".join(filtered_identifiers)

        query = f"""
        SELECT id, name, slug, *
        FROM {table_id}
        WHERE id IN ('{identifier_list}') OR LOWER(name) IN ('{identifier_list}')
        """
        return self.execute_query(query)

    def get_protocol_activity_counts(self, granularity: str = "month") -> pl.DataFrame:
        """Retrieve the count of entries for each protocol grouped by specified granularity."""
        table_id = self._table_ref(CONFIG.TABLES["C"])
        date_trunc_expr = self._get_date_trunc_expr(
            granularity
        )  # Use the subclass-specific method

        query = f"""
        SELECT
            {date_trunc_expr} AS period,
            id,
            COUNT(*) AS entry_count
        FROM {table_id}
        GROUP BY period, id
        ORDER BY period, id
        """
        return self.execute_query(query)

    def compare_periods(
        self,
        start_date: str,
        granularity: str = "daily",
        chain_filter: str = "exclude_borrowed",
    ) -> pl.DataFrame:
        """chain_filter := 'all', 'exclude_borrowed', 'only_borrowed', 'Ethereum'"""
        table_id_c = self._table_ref(CONFIG.TABLES["C"])
        table_id_a = self._table_ref(CONFIG.TABLES["A"])

        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = start_dt + self.granularity_map.get(granularity)

        start_timestamp = int(start_dt.timestamp())
        end_timestamp = int(end_dt.timestamp())

        query = f"""
        WITH FilteredData AS (
            SELECT
                C.id,
                C.chain_name,
                C.token_name,
                MIN(C.date) AS first_date,
                MAX(C.date) AS last_date
            FROM {table_id_c} C
            WHERE C.date BETWEEN {start_timestamp} AND {end_timestamp}
            GROUP BY C.id, C.chain_name, C.token_name
            HAVING COUNT(*) > 1
        ),
        FirstData AS (
            SELECT
                C.id,
                C.chain_name,
                C.token_name,
                C.date AS first_date,
                C.quantity AS qty,
                C.value_usd AS usd
            FROM {table_id_c} C
            JOIN FilteredData F ON C.id = F.id AND C.chain_name = F.chain_name AND C.token_name = F.token_name AND C.date = F.first_date
        ),
        LastData AS (
            SELECT
                C.id,
                C.chain_name,
                C.token_name,
                C.date AS last_date,
                C.quantity AS last_qty,
                C.value_usd AS last_usd
            FROM {table_id_c} C
            JOIN FilteredData F ON C.id = F.id AND C.chain_name = F.chain_name AND C.token_name = F.token_name AND C.date = F.last_date
        ),
        ProtocolDetails AS (
            SELECT
                A.id,
                A.name AS protocol_name,
                A.category,
                A.type
            FROM {table_id_a} A
        )
        SELECT
            F.id,
            PD.protocol_name,
            PD.category,
            PD.type,
            F.chain_name,
            F.token_name,
            FirstData.first_date,
            LastData.last_date,
            FirstData.qty,
            FirstData.usd,
            LastData.last_qty - FirstData.qty AS qty_change,
            LastData.last_usd - FirstData.usd AS usd_change
        FROM FilteredData F
        JOIN FirstData ON F.id = FirstData.id AND F.chain_name = FirstData.chain_name AND F.token_name = FirstData.token_name
        JOIN LastData ON F.id = LastData.id AND F.chain_name = LastData.chain_name AND F.token_name = LastData.token_name
        JOIN ProtocolDetails PD ON F.id = PD.id
        WHERE {self._filter_chain('F.chain_name', chain_filter)}
        """

        df = self.execute_query(query)
        return df

    def _filter_chain(self, column_name: str, filter_mode: str) -> str:
        if filter_mode == "exclude_borrowed":
            return f"{column_name} NOT LIKE '%borrowed%'"
        elif filter_mode == "only_borrowed":
            return f"{column_name} LIKE '%borrowed%'"
        elif filter_mode == "all":
            return "1=1"  # always true so does not filter out any rows
        else:  # if filter_mode is specified as a chain name, filter by that chain name
            return f"{column_name} = '{filter_mode}'"

    def detect_duplicates(self, table_id: str) -> pl.DataFrame:
        raise NotImplementedError(
            "Needs to be implemented in a database-specific wrapper class."
        )

    def remove_duplicates(self, table_id: str, limit: int = None) -> None:
        raise NotImplementedError(
            "Needs to be implemented in a database-specific wrapper class."
        )

    def get_token_distribution(self, token_name: str, granularity: str) -> pl.DataFrame:
        """Retrieve the distribution of a specific token across protocols over time at specified granularity."""

        date_trunc_expr = self._get_date_trunc_expr(granularity)
        table_id_c = self._table_ref(CONFIG.TABLES["C"])
        table_id_a = self._table_ref(CONFIG.TABLES["A"])

        query = f"""
        SELECT 
            {date_trunc_expr} as aggregated_date,
            C.id as id,
            A.name as protocol_name,
            A.category as category,
            A.type as type,
            C.chain_name,
            SUM(C.quantity) as total_quantity,
            SUM(C.value_usd) as total_value_usd
        FROM 
            {table_id_c} C
        INNER JOIN 
            {table_id_a} A ON C.id = A.id
        WHERE 
            C.token_name = '{token_name}' AND
            C.quantity > 0 AND
            C.value_usd > 0
        GROUP BY 
            aggregated_date,
            id,
            protocol_name,
            category,
            type,
            chain_name
        ORDER BY 
            aggregated_date,
            id
        """
        return self.execute_query(query)

    def get_aggregated_data(
        self,
        granularity: str,
        token_name: str = None,
        protocol_name: str = None,
        start_date: str = None,
        end_date: str = None,
    ) -> pl.DataFrame:
        table_id_c = self._table_ref(CONFIG.TABLES["C"])
        table_id_a = self._table_ref(CONFIG.TABLES["A"])
        date_trunc_expr = self._get_date_trunc_expr(granularity)
        token_filter = f"AND C.token_name = '{token_name}'" if token_name else ""
        protocol_filter = f"AND A.name = '{protocol_name}'" if protocol_name else ""
        date_filter = ""
        if start_date and end_date:
            start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())
            end_timestamp = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp())
            date_filter = f"AND CAST(C.date AS BIGINT) BETWEEN {start_timestamp} AND {end_timestamp}"

        query = f"""
        WITH AggregatedData AS (
            SELECT
                {date_trunc_expr} as aggregated_date,
                C.id as id,
                A.name as protocol_name,
                A.category as category,
                A.type as type,
                C.chain_name,
                C.token_name,
                C.quantity,
                C.value_usd
            FROM 
                {table_id_c} C
            INNER JOIN 
                {table_id_a} A ON C.id = A.id
            WHERE 
                C.quantity > 0 AND
                C.value_usd > 0
                {token_filter}
                {protocol_filter}
                {date_filter}
        )
        SELECT
            aggregated_date,
            id,
            protocol_name,
            category,
            type,
            chain_name,
            token_name,
            AVG(quantity) as qty,
            AVG(value_usd) as usd
        FROM AggregatedData
        GROUP BY
            aggregated_date,
            id,
            protocol_name,
            category,
            type,
            chain_name,
            token_name
        ORDER BY 
            aggregated_date, id
        """

        df = self.execute_query(query)
        df = self._map_category_to_type(df)
        return df

    def _map_category_to_type(self, df: pl.DataFrame) -> pl.DataFrame:
        """Maps categories using a predefined mapping and updates the DataFrame."""
        category_mapping = CONFIG.CATEGORY_MAPPING
        reverse_category_mapping = {
            v: k for k, values in category_mapping.items() for v in values
        }

        def map_categories(s: pl.Series) -> pl.Series:
            return s.map_elements(
                lambda x: reverse_category_mapping.get(x, "Unknown"),
                return_dtype=pl.Utf8,
            )

        return df.with_columns(
            pl.col("category").map_batches(map_categories).alias("type")
        )

    def get_historical_changes(self, granularity: str, start_date: str) -> pl.DataFrame:
        table_id_c = self._table_ref(CONFIG.TABLES["C"])
        table_id_a = self._table_ref(CONFIG.TABLES["A"])
        date_trunc_expr = self._get_date_trunc_expr(granularity)
        start_timestamp = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp())

        query = f"""
        WITH PeriodData AS (
            SELECT
                {date_trunc_expr} AS period,
                id,
                chain_name,
                token_name,
                MIN(date) AS first_date,
                MAX(date) AS last_date
            FROM {table_id_c}
            WHERE CAST(date AS BIGINT) >= {start_timestamp}
            GROUP BY period, id, chain_name, token_name
        ),
        FirstData AS (
            SELECT
                P.id,
                P.chain_name,
                P.token_name,
                P.period,
                C.date AS first_date,
                C.quantity AS first_qty,
                C.value_usd AS first_usd
            FROM PeriodData P
            JOIN {table_id_c} C ON P.id = C.id AND P.chain_name = C.chain_name AND P.token_name = C.token_name AND P.first_date = C.date
        ),
        LastData AS (
            SELECT
                P.id,
                P.chain_name,
                P.token_name,
                P.period,
                C.date AS last_date,
                C.quantity AS last_qty,
                C.value_usd AS last_usd
            FROM PeriodData P
            JOIN {table_id_c} C ON P.id = C.id AND P.chain_name = C.chain_name AND P.token_name = C.token_name AND P.last_date = C.date
        ),
        ProtocolDetails AS (
            SELECT
                id,
                name AS protocol_name,
                category,
                type
            FROM {table_id_a}
        )
        SELECT
            F.id,
            PD.protocol_name,
            PD.category,
            PD.type,
            F.chain_name,
            F.token_name,
            F.period,
            FirstData.first_date as date,
            LastData.last_date,
            FirstData.first_qty as qty,
            FirstData.first_usd as usd,
            LastData.last_qty - FirstData.first_qty AS qty_change,
            LastData.last_usd - FirstData.first_usd AS usd_change
        FROM PeriodData F
        JOIN FirstData ON F.id = FirstData.id AND F.chain_name = FirstData.chain_name AND F.token_name = FirstData.token_name AND F.period = FirstData.period
        JOIN LastData ON F.id = LastData.id AND F.chain_name = LastData.chain_name AND F.token_name = LastData.token_name AND F.period = LastData.period
        JOIN ProtocolDetails PD ON F.id = PD.id
        ORDER BY F.period, F.id
        """

        return self.execute_query(query)

    def get_all_unique_token_names(
        self, chains: Optional[List[str]] = None
    ) -> pl.DataFrame:
        """
        Fetch all unique token names from the main data table.

        Args:
            chains (Optional[List[str]]): List of chain names to filter by. If None, fetch for all chains.

        Returns:
            pl.DataFrame: DataFrame containing unique token names.
        """
        table_id = self._table_ref(CONFIG.TABLES["C"])

        query = f"""
        SELECT DISTINCT token_name, chain_name
        FROM {table_id}
        WHERE token_name IS NOT NULL AND token_name != ''
        """

        if chains:
            chain_list = ", ".join(f"'{chain}'" for chain in chains)
            query += f" AND chain_name IN ({chain_list})"

        query += " ORDER BY chain_name, token_name"

        return self.execute_query(query)
