from google.cloud.bigquery import SchemaField
from pydantic import BaseModel, Field


class ProtocolData(BaseModel):
    id: str = Field(..., description="Unique identifier for the protocol")
    chain_name: str = Field(..., description="Blockchain name")
    token_name: str = Field(..., description="Token name")
    date: int = Field(..., description="Unix timestamp of the data")
    quantity: float = Field(..., description="Quantity of the token")
    value_usd: float = Field(..., description="Value in USD")


def pydantic_model_to_schema(
    model_class: type[BaseModel], target: str = "BigQueryWrapper"
) -> list:
    """
    Converts a Pydantic model class to a schema for BigQuery or MotherDuck.

    Parameters:
    - model_class (type[BaseModel]): The Pydantic model class to convert.
    - target (str): The target database type ('bigquery' or 'motherduck').

    Returns:
    - list: A list of SchemaField objects for BigQuery or a list of tuples for MotherDuck.
    """
    schema = []
    for field_name, field in model_class.__fields__.items():
        field_type = field.annotation
        description = field.description

        if target == "BigQueryWrapper":
            if field_type == int:
                bq_type = "INT64"
            elif field_type == str:
                bq_type = "STRING"
            elif field_type == float:
                bq_type = "FLOAT64"
            else:
                bq_type = "STRING"
            schema.append(
                SchemaField(
                    field_name, bq_type, mode="NULLABLE", description=description
                )
            )
        elif target == "MotherDuckWrapper":
            if field_type == int:
                duckdb_type = "BIGINT"
            elif field_type == str:
                duckdb_type = "VARCHAR"
            elif field_type == float:
                duckdb_type = "DOUBLE"
            else:
                duckdb_type = "VARCHAR"
            schema.append((field_name, duckdb_type, description))

    return schema
