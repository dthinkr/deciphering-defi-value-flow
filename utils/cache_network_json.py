import json
import os
from datetime import datetime

import requests
from dateutil.relativedelta import relativedelta
from prefect import flow, task

from utils import load_config

CONFIG = load_config()


@task(retries=5, retry_delay_seconds=5)
def fetch_and_save_data(date, granularity):
    url = CONFIG.ENDPOINT_URL + "network-json/"
    formatted_date = date.strftime("%Y-%m-%d")
    params = {"granularity": granularity, "chain_filter": "exclude_borrowed"}
    response = requests.get(url + formatted_date, params=params)
    if response.status_code == 200:
        data = response.json()
        with open(f"data/network_data/{formatted_date}.json", "w") as f:
            json.dump(data, f)
        return {formatted_date: data}


@flow
def run_data_collection_flow(start_date, end_date, granularity="day"):
    data_collection = {}
    os.makedirs("data/network_data", exist_ok=True)
    current_date = start_date
    while current_date <= end_date:
        data = fetch_and_save_data(current_date, granularity)
        data_collection.update(data)
        if granularity == "day":
            current_date += relativedelta(days=1)
        elif granularity == "week":
            current_date += relativedelta(weeks=1)
        elif granularity == "month":
            current_date += relativedelta(months=1)
        elif granularity == "year":
            current_date += relativedelta(years=1)
    return data_collection


if __name__ == "__main__":
    run_data_collection_flow(
        datetime(
            datetime.now().year, 1, 1
        ),  # start from the first day of the current year
        datetime.now(),
        granularity="week",
    )
