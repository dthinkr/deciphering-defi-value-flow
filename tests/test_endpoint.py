from unittest.mock import AsyncMock, patch

import polars as pl
from fastapi.testclient import TestClient

from utils.endpoint import app

client = TestClient(app)


@patch(
    "utils.endpoint.ETLNetwork.calculate_network", return_value={"data": "processed"}
)
@patch(
    "utils.endpoint.BigQueryWrapper.compare_periods", return_value="mocked dataframe"
)
def test_get_network_data(mock_compare_periods, mock_calculate_network):
    response = client.get(
        "/network-json/2024-01-01?granularity=day&chain_filter=exclude_borrowed"
    )
    assert response.status_code == 200
    assert response.json() == {"data": "processed"}
