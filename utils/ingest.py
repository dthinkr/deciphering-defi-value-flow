import asyncio
import atexit
import json
import os
from typing import Tuple

import pandas as pd
import polars as pl
import psutil
import requests
from prefect import flow, get_run_logger, task

from data_access import BigQueryWrapper, ClientPool, MotherDuckWrapper
from data_analysis import ETLNetwork
from data_visualize import save_heatmap
from utils import ProtocolData, load_config, pydantic_model_to_schema

CONFIG = load_config()

os.makedirs(CONFIG.DATA_DIR, exist_ok=True)
PROTOCOL_HEADERS_JSON = os.path.join(CONFIG.BASE_DIR, "protocol_headers.json")
PROTOCOL_HEADERS_PARQUET = os.path.join(CONFIG.BASE_DIR, "protocol_headers.parquet")

if CONFIG.DATABASE_TYPE == "motherduck":
    pool = ClientPool(
        client_factory=MotherDuckWrapper,
        max_wrappers=CONFIG.MAX_CLIENTS,
    )
elif CONFIG.DATABASE_TYPE == "bigquery":
    pool = ClientPool(
        client_factory=BigQueryWrapper,
        max_wrappers=CONFIG.MAX_CLIENTS,
    )
else:
    raise NotImplementedError("Unsupported database type specified in configuration.")

schema = pydantic_model_to_schema(
    model_class=ProtocolData, target=pool.client_factory.__name__
)


@task(retries=5, retry_delay_seconds=5)
def fetch_data(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
    except Exception:
        raise


def save_data_to_file(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


@task
async def download_protocol_headers():
    url = f"{CONFIG.BASE_URL}protocols"
    data = fetch_data.fn(url)
    if data:
        save_data_to_file(data, PROTOCOL_HEADERS_JSON)
        df = pd.DataFrame(data)
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].astype(str)
        df["type"] = "default"
        df.to_parquet(PROTOCOL_HEADERS_PARQUET, index=False)
        os.remove(PROTOCOL_HEADERS_JSON)


def extract_token_tvl(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)

    results = []
    for chain_name, chain_data in data["chainTvls"].items():
        tokens_usd = chain_data["tokensInUsd"]
        tokens_quantity = chain_data["tokens"]

        for usd_entry, quantity_entry in zip(tokens_usd, tokens_quantity):
            date = usd_entry["date"]
            for token_name, value_usd in usd_entry["tokens"].items():
                quantity = quantity_entry["tokens"].get(token_name, 0)
                # large int handling
                value_usd_str = str(value_usd)
                quantity_str = str(quantity)
                results.append(
                    {
                        "id": data["id"],
                        "chain_name": chain_name,
                        "date": date,
                        "token_name": token_name,
                        "quantity": quantity_str,
                        "value_usd": value_usd_str,
                    }
                )

    df = pl.DataFrame(results)

    # convert large int back
    if "quantity" in df.columns:
        df = df.with_columns(pl.col("quantity").cast(pl.Float64).fill_null(0))
    if "value_usd" in df.columns:
        df = df.with_columns(pl.col("value_usd").cast(pl.Float64).fill_null(0))

    return df


@task
async def process_and_filter_file(json_file_path, parquet_file_path, latest_dates):
    df = extract_token_tvl(json_file_path)
    if not df.is_empty():
        if not latest_dates.is_empty():
            df = df.join(
                latest_dates, on=["id", "chain_name", "token_name"], how="left"
            )
            filtered_rows = df.filter(
                (pl.col("date") > pl.col("latest_date"))
                | pl.col("latest_date").is_null()
            )
            filtered_rows = filtered_rows.drop("latest_date")
        else:
            filtered_rows = df

        if not filtered_rows.is_empty():
            filtered_rows.write_parquet(parquet_file_path)
            get_run_logger().warning(
                "Prepared %s lines of new data for %s",
                filtered_rows.shape[0],
                json_file_path,
            )
        else:
            get_run_logger().info("No new data to process for %s.", json_file_path)
    else:
        get_run_logger().critical("No data found in llama data %s.", json_file_path)


@task(retries=2, retry_delay_seconds=[10, 30])
async def upload_parquet(file_path, table_id):
    wrapper = await pool.get_wrapper()
    wrapper.append_parquet(file_path=file_path, table_id=table_id)
    await pool.release_wrapper(wrapper)


@task
async def download_and_process_single_protocol(slug, latest_dates):
    url = f"{CONFIG.BASE_URL}protocol/{slug}"
    data = fetch_data.fn(url)
    if data:
        json_file_path = os.path.join(CONFIG.DATA_DIR, f"{slug}.json")
        parquet_file_path = os.path.join(CONFIG.DATA_DIR, f"{slug}.parquet")
        save_data_to_file(data, json_file_path)
        await process_and_filter_file.fn(
            json_file_path, parquet_file_path, latest_dates
        )
        os.remove(json_file_path)
        return parquet_file_path


@task(retries=3, retry_delay_seconds=[10, 30, 300])
async def prepare_table_a(file_path):
    table_id = CONFIG.TABLES["A"]
    wrapper = await pool.get_wrapper()
    try:
        try:
            wrapper.delete_table(table_id)
        except Exception:
            get_run_logger().warning("Table %s does not exist, creating it.", table_id)
        wrapper.create_table(table_id=table_id, file_path=file_path)
        os.remove(file_path)
    finally:
        await pool.release_wrapper(wrapper)


@task(retries=3, retry_delay_seconds=[10, 30, 300])
async def prepare_table_c(clustering_fields):
    table_id = CONFIG.TABLES["C"]
    wrapper = await pool.get_wrapper()
    try:
        try:
            wrapper.get_table(table_id)
        except:
            try:
                wrapper.create_table(
                    table_id=table_id,
                    schema=schema,
                    clustering_fields=clustering_fields,
                )
            except Exception as e:
                if "already exists" in str(e):
                    pass
                else:
                    raise
    finally:
        await pool.release_wrapper(wrapper)


@task
async def update_mapping():
    wrapper = await pool.get_wrapper()
    etl = ETLNetwork(wrapper=wrapper)
    etl.update_mapping()
    await pool.release_wrapper(wrapper)


@task
async def plot_heatmap():
    wrapper = await pool.get_wrapper()
    df = wrapper.get_protocol_activity_counts()
    save_heatmap(df)
    await pool.release_wrapper(wrapper)


def _get_system_memory_info_gb():
    mem = psutil.virtual_memory()
    return mem.total / (1024.0**3)


def _calculate_concurrent_tasks(
    memory_per_task_gb=20 / 200, safety_factor=CONFIG.SAFETY_FACTOR
):
    total_memory_gb = _get_system_memory_info_gb()
    usable_memory_gb = total_memory_gb * safety_factor
    max_concurrent_tasks_based_on_memory = int(usable_memory_gb / memory_per_task_gb)
    return max_concurrent_tasks_based_on_memory


@task
async def get_latest_dates_foreach_id_chain_token():
    wrapper = await pool.get_wrapper()
    latest_dates = wrapper.get_latest_dates_foreach_id_chain_token()
    await pool.release_wrapper(wrapper)
    return latest_dates


@task
async def calculate_protocol_batches() -> Tuple[pl.DataFrame, int, int, int]:
    wrapper = await pool.get_wrapper()

    all_protocol_slugs = wrapper.get_all_protocol_slugs()  # this is polars df
    await pool.release_wrapper(wrapper)

    max_slugs = (
        len(all_protocol_slugs) if CONFIG.MAX_SLUGS is None else CONFIG.MAX_SLUGS
    )

    # Limit the number of slugs if necessary
    all_protocol_slugs = all_protocol_slugs.slice(0, max_slugs)

    max_concurrent_tasks = _calculate_concurrent_tasks()
    total_slugs_to_process = len(all_protocol_slugs)
    total_batches = (
        total_slugs_to_process + max_concurrent_tasks - 1
    ) // max_concurrent_tasks  # rounding up, this has the same effect as math.ceil

    return (
        all_protocol_slugs,
        max_concurrent_tasks,
        total_slugs_to_process,
        total_batches,
    )


@task
async def combine_parquet_files(batch_files, batch_index, total_batches):
    combined_file_path = (
        f"{CONFIG.DATA_DIR}/combined_batch_{batch_index}_of_{total_batches}.parquet"
    )

    # Filter out None values and check file existence
    valid_files = [
        file for file in batch_files if file is not None and os.path.exists(file)
    ]

    if not valid_files:
        get_run_logger().warning(
            "No valid Parquet files to combine for batch %s/%s.",
            batch_index,
            total_batches,
        )
        return None

    dfs = [pl.read_parquet(file) for file in valid_files]
    combined_df = pl.concat(dfs)

    combined_df.write_parquet(combined_file_path)

    for file_path in valid_files:
        os.remove(file_path)  # Delete the Parquet file

    get_run_logger().warning(
        "Combined %s files into %s for batch %s/%s",
        len(batch_files),
        combined_file_path,
        batch_index,
        total_batches,
    )


# Global variable to track the status of the ingest flow
INGEST_FLOW_ACTIVE = False


@flow
async def ingest_llama():
    global INGEST_FLOW_ACTIVE
    try:
        print("Starting ingest_llama function")
        await download_protocol_headers()
        print("Protocol headers downloaded")
        await prepare_table_a(PROTOCOL_HEADERS_PARQUET)
        print("Table A prepared")
        await prepare_table_c(CONFIG.CLUSTERING_FIELDS)
        print("Table C prepared")
        latest_dates = await get_latest_dates_foreach_id_chain_token()
        print("Latest dates retrieved")

        INGEST_FLOW_ACTIVE = True
        print("INGEST_FLOW_ACTIVE set to True")

        (
            all_protocol_slugs,
            max_concurrent_tasks,
            total_slugs_to_process,
            total_batches,
        ) = await calculate_protocol_batches()
        print(
            f"Calculated batches: {total_batches} batches, {total_slugs_to_process} slugs"
        )

        for i in range(0, total_slugs_to_process, max_concurrent_tasks):
            batch_slugs = all_protocol_slugs.slice(i, max_concurrent_tasks)[
                "slug"
            ].to_list()
            print(f"Processing batch {i // max_concurrent_tasks + 1}/{total_batches}")

            tasks = [
                download_and_process_single_protocol(slug, latest_dates)
                for slug in batch_slugs
            ]
            if tasks:
                batch_files = await asyncio.gather(*tasks)
                print(f"Processed {len(batch_files)} protocols in this batch")
                await combine_parquet_files(
                    batch_files, i // max_concurrent_tasks, total_batches
                )

        print("All batches processed")
        await update_mapping()
        print("Mapping updated")
        await plot_heatmap()
        print("Heatmap plotted")

        latest_dates.to_pandas().to_csv(
            f"data/latest_dates/{CONFIG.DATABASE_TYPE}_latest_dates.csv"
        )
        print("Latest dates saved to CSV")

        INGEST_FLOW_ACTIVE = False
        print("INGEST_FLOW_ACTIVE set to False")

    except Exception as e:
        INGEST_FLOW_ACTIVE = False
        print(f"Error in ingest_llama: {e}")


@flow
async def upload_combined_files():
    print("Starting upload_combined_files function")
    while not INGEST_FLOW_ACTIVE:
        print("Waiting for ingest flow to activate...")
        await asyncio.sleep(CONFIG.CHECK_INTERVAL_SECONDS)

    print("Ingest flow active, starting upload process")
    while INGEST_FLOW_ACTIVE:
        combined_files = [
            f for f in os.listdir(CONFIG.DATA_DIR) if f.startswith("combined_batch_")
        ]
        if combined_files:
            print(f"Found {len(combined_files)} combined files to upload")
            for file_path in combined_files:
                full_path = os.path.join(CONFIG.DATA_DIR, file_path)
                print(f"Uploading {full_path}")
                await upload_parquet(full_path, CONFIG.TABLES["C"])

                lzdf = pl.scan_parquet(full_path)
                row_count = lzdf.select(pl.len()).collect()["len"][0]

                print(f"Uploaded {full_path} with {row_count} rows")
                os.remove(full_path)
                print(f"Removed {full_path}")
            await asyncio.sleep(CONFIG.CHECK_INTERVAL_SECONDS)
        else:
            print("No combined files found, waiting...")
            await asyncio.sleep(CONFIG.CHECK_INTERVAL_SECONDS)

    print("Ingest flow no longer active, ending upload process")


def cleanup_data_directory():
    data_dir = CONFIG.DATA_DIR
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        if os.path.isfile(file_path):
            os.remove(file_path)


atexit.register(cleanup_data_directory)


async def main():
    try:
        print("Starting ingestion...")
        ingest_task = asyncio.create_task(ingest_llama())
        print("Ingestion task created")
        upload_task = asyncio.create_task(upload_combined_files())
        print("Upload task created")
        await asyncio.gather(upload_task, ingest_task)
        print("All tasks completed")
    except Exception as e:
        print(f"Error in main function: {e}")
        cleanup_data_directory()
        print("Data directory cleaned up")


if __name__ == "__main__":
    asyncio.run(main())
