import logging
from contextlib import asynccontextmanager
from datetime import datetime, date
from functools import wraps
from inspect import signature

from fastapi import Depends, FastAPI, HTTPException, Path, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse

from data_access import BigQueryWrapper, ClientPool, MotherDuckWrapper
from data_analysis import ETLNetwork
from data_visualize import visualize_top_nodes
from utils import load_config

CONFIG = load_config()

app = FastAPI()

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


def get_client_pool():
    return pool


@asynccontextmanager
async def managed_wrapper(pool):
    wrapper = await pool.get_wrapper()
    try:
        yield wrapper
    finally:
        await pool.release_wrapper(wrapper)


def handle_context(f):
    """ "handles both error and getting and releasing wrapper"""

    @wraps(f)
    async def decorator(*args, **kwargs):
        pool = kwargs.get("pool", None)
        if not pool:
            raise ValueError("ClientPool instance required for this operation.")
        try:
            async with managed_wrapper(pool) as wrapper:
                sig = signature(f)
                if "wrapper" in sig.parameters:
                    kwargs["wrapper"] = wrapper
                return await f(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {f.__name__}: {str(e)}", exc_info=True)
            raise HTTPException(status_code=500, detail=str(e))

    return decorator


"""endpoints"""


@app.get("/network-json/{date_input}", summary="Network Data")
@handle_context
async def get_network_data(
    date_input: str = Path(..., description="Date in 'YYYY-MM-DD' format."),
    granularity: str = Query("day", description="Daily, month, or year granularity."),
    chain_filter: str = Query("exclude_borrowed", description="Filter for chain data."),
    pool: ClientPool = Depends(get_client_pool),
    wrapper=None,
):
    etl_network = ETLNetwork(wrapper=wrapper)
    C = wrapper.compare_periods(
        date_input, granularity=granularity, chain_filter=chain_filter
    )
    network_data = etl_network.calculate_network(C)
    return network_data


@app.get("/render-network/{date_input}", response_class=HTMLResponse)
@handle_context
async def render_network(
    date_input: str = Path(..., description="Date in 'YYYY-MM-DD' format."),
    top_x: int = Query(CONFIG.TOP_X, description="Number of top nodes to display."),
    granularity: str = Query("day", description="Granularity of the data."),
    chain_filter: str = Query("exclude_borrowed", description="Filter for chain data."),
    pool: ClientPool = Depends(get_client_pool),
    wrapper=None,
):
    network_data = await get_network_data(
        date_input, granularity, chain_filter, pool=pool
    )
    node_ids = [node["id"] for node in network_data["nodes"]]
    protocol_info_df = wrapper.get_protocol_info_by_id_or_name(node_ids)
    id_to_name = (
        {row["id"]: row["name"] for row in protocol_info_df.to_dicts()}
        if not protocol_info_df.height == 0
        else None
    )
    net = visualize_top_nodes(network_data, id_to_name=id_to_name, top_x=top_x)

    # Generate HTML content directly
    html_content = net.generate_html()

    return HTMLResponse(content=html_content, status_code=200)


@app.get("/token-distribution/{token_name}/{granularity}", summary="Token Distribution")
@handle_context
async def token_distribution(
    token_name: str,
    granularity: str,
    pool: ClientPool = Depends(get_client_pool),
    wrapper=None,
):
    df = wrapper.get_token_distribution(token_name, granularity)
    csv_data = df.write_csv()
    return Response(content=csv_data, media_type="text/csv")


@app.get("/protocol-data/{protocol_name}/{granularity}", summary="Protocol Data")
@handle_context
async def protocol_data(
    protocol_name: str,
    granularity: str,
    pool: ClientPool = Depends(get_client_pool),
    wrapper=None,
):
    df = wrapper.get_protocol_data(protocol_name, granularity)
    csv_data = df.write_csv()
    return Response(content=csv_data, media_type="text/csv")


@app.get("/unique-token-names", summary="Unique Token Names")
@handle_context
async def unique_token_names(pool: ClientPool = Depends(get_client_pool), wrapper=None):
    """
    Returns a list of unique token names from the default table.
    """
    df = wrapper.get_unique_token_names()  # No table name passed, uses default
    csv_data = df.write_csv()
    return Response(content=csv_data, media_type="text/csv")


@app.get("/historical-network", summary="Historical Network Data")
@handle_context
async def get_historical_network(
    time_granularity: str = Query("day", description="Time granularity for the data."),
    start_date: str = Query(
        "2024-04-01", description="Start date for the data in 'YYYY-MM-DD' format."
    ),
    pool: ClientPool = Depends(get_client_pool),
    wrapper=None,
):
    df = wrapper.get_historical_changes(time_granularity, start_date)
    etl_network = ETLNetwork(wrapper)
    data = etl_network.calculate_historical_network(df)
    if isinstance(data, dict):
        data = {
            key.isoformat() if isinstance(key, (date, datetime)) else key: value
            for key, value in data.items()
        }
    return JSONResponse(content={"data": data})


@app.get("/historical-network-edges", summary="Filtered Network Edges CSV")
@handle_context
async def get_historical_network_edge(
    time_granularity: str = Query("day", description="Time granularity for the data."),
    start_date: str = Query(
        "2024-04-01", description="Start date for the data in 'YYYY-MM-DD' format."
    ),
    size_threshold: int = Query(
        1000, description="Size threshold for filtering links."
    ),
    pool: ClientPool = Depends(get_client_pool),
    wrapper=None,
):
    etl_network = ETLNetwork(wrapper)
    csv_data = etl_network.generate_historical_network_edges(
        time_granularity, start_date, size_threshold
    )
    return Response(content=csv_data, media_type="text/csv")


@app.get("/historical-network-nodes", summary="Network Nodes CSV")
@handle_context
async def get_historical_network_nodes(
    time_granularity: str = Query("day", description="Time granularity for the data."),
    start_date: str = Query(
        "2024-04-01", description="Start date for the data in 'YYYY-MM-DD' format."
    ),
    size_threshold: int = Query(
        1000, description="Size threshold for filtering nodes."
    ),
    pool: ClientPool = Depends(get_client_pool),
    wrapper=None,
):
    etl_network = ETLNetwork(wrapper)
    csv_data = etl_network.generate_historical_network_nodes(
        time_granularity, start_date, size_threshold
    )
    return Response(content=csv_data, media_type="text/csv")
