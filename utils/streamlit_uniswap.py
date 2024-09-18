import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import os
import plotly.express as px
from streamlit_extras.add_vertical_space import add_vertical_space
import numpy as np
from dotenv import load_dotenv
import plotly.graph_objects as go
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from utils.plot import create_uniswap_v3_tvl_chart

st.set_page_config(layout="wide")
color_theme = [
    "#FFAA00",
    "#497EB9",
    "#94C3E9",
    "#7AB4A3",
    "#94BFBD",
    "#FFCF47",
    "#FE4A49",
    "#2AB7CA",
    "#E6E6EA",
    "#4A4E4D",
    "#3DA4AB",
    "#F6CD61",
    "#FE8A71",
    "#D11141",
    "#00B159",
    "#00AEDB",
    "#F37735",
    "#F4F1BB",
    "#9BC1BC",
    "#5CA4A9",
    "#D6A184",
    "#63474D",
    "#AA767C",
    "#373F51",
    "#A9BCD0",
    "#457B9D",
]


@st.cache_data
def read_data():
    data = pd.read_parquet("data/bsheet/uniswap_balance_sheet_v2.parquet")
    data = data.reset_index()
    return data


@st.cache_data
def data_transformation(data):
    numeric_col = ["amount", "usd_value"]
    for col in numeric_col:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data["date_str"] = pd.to_datetime(data["date"], unit="s")
    data["date_str"] = data["date_str"].dt.date

    ### exclude outliers for asset
    asset = data[data["type"] == "asset"]
    exclude_date_list = list(
        set(
            asset.sort_values("usd_value", ascending=False)
            .head(15)["date_str"]
            .to_list()
        )
    )
    exclude_token_list = list(
        set(asset.sort_values("usd_value", ascending=False).head(15)["token"].to_list())
    )
    filt_1 = [x in exclude_date_list for x in data["date_str"]]
    exclude_data = data.loc[filt_1]
    filt_2 = [x in exclude_token_list for x in exclude_data["token"]]
    exclude_data = exclude_data.loc[filt_2]
    exclude_index = exclude_data[exclude_data["type"] == "asset"].index
    data = data.drop(exclude_index)

    # Modified groupby operation
    data = data.groupby(
        ["type", "liquidity", "source", "token", "date_str"], as_index=False
    ).agg({"usd_value": "sum", "amount": "sum"})

    return data


@st.cache_data
def token_balance_analysis(bs_data):
    df = bs_data.copy()
    analyze_type = ["asset", "liability"]
    results = []
    for index, item_type in enumerate(analyze_type):
        tempt = df[df["type"] == item_type]
        results.append(tempt)
        results[index] = results[index].rename(
            {
                "usd_value": analyze_type[index] + "_usd_value",
                "amount": analyze_type[index] + "_amount",
            },
            axis=1,
        )

    token_bs = pd.merge(results[0], results[1], how="outer", on=["date_str", "token"])
    reserved_col = [
        "date_str",
        "token",
        "asset_amount",
        "asset_usd_value",
        "liability_amount",
        "liability_usd_value",
    ]

    token_bs = token_bs[reserved_col]
    token_bs = token_bs.fillna(0)
    token_bs["usd_value_balance"] = (
        token_bs["asset_usd_value"] - token_bs["liability_usd_value"]
    )
    token_bs["amount_balance"] = token_bs["asset_amount"] - token_bs["liability_amount"]
    token_bs = token_bs.sort_values(["token", "date_str"])
    return token_bs


def overview_uniswap_bs(data):
    summary_type = (
        data.groupby(["date_str", "type"])["usd_value"].sum().unstack().fillna(0)
    )
    summary_type["liability_ma"] = summary_type["liability"].rolling(7).mean()
    summary_type["asset_ma"] = summary_type["asset"].rolling(7).mean()
    summary_type["equity_ma"] = summary_type["equity"].rolling(7).mean()
    summary_type["equity+liability"] = (
        summary_type["equity_ma"] + summary_type["liability_ma"]
    )
    return summary_type


def area_chart_1(summary_type):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=summary_type.index,
            y=summary_type.asset_ma,
            fill="tozeroy",
            name="asset",
            fillcolor=color_theme[0],
            mode="none",
        )
    )  # fill down to xaxis
    # fill to trace0 y
    fig.add_trace(
        go.Scatter(
            x=summary_type.index,
            y=-summary_type["equity+liability"],
            fill="tozeroy",
            mode="none",
            name="equity",
            fillcolor=color_theme[1],
        )
    )  # fill to trace0 y
    fig.add_trace(
        go.Scatter(
            x=summary_type.index,
            y=-summary_type.liability_ma,
            fill="tozeroy",
            name="liability",
            fillcolor=color_theme[2],
            mode="none",
        )
    )
    return fig


def token_balance_df(token_df, token):
    results = token_df.copy()
    charts_df = results[results["token"] == token]
    window_size = 7  # This could be adjusted as needed
    charts_df["moving_average"] = (
        charts_df["usd_value_balance"].rolling(window=window_size).mean()
    )

    return charts_df


#### app

# ## overview
# data = read_data()
# print("Successfully load the data")

# data = data_transformation(data)
# print("Successfully transform the data")

# summary_type = overview_uniswap_bs(data)
# fig1 = area_chart_1(summary_type)

st.title("Uniswap Balance Sheet Analysis")

st.subheader("Uniswap V3 Pools")
pool_chart = create_uniswap_v3_tvl_chart()
st.altair_chart(pool_chart, use_container_width=True)

# st.subheader("Protocol Balance Sheet Overview")
# st.plotly_chart(fig1, use_container_width=True)


# ## pool token analysis
# token_bs = token_balance_analysis(data)

# st.subheader('Uniswap Token Balance Analysis')

# token_list = ['USDT','USDC','WETH','DAI','WBTC','MKR','COMP','UNI']
# #### add token selection
# row11, row12 = st.columns(
#     (0.3, 0.7)
# )
# token  = row11.selectbox('Please select a token',token_list)

# charts_df = token_balance_df(token_bs, token)

# fig2 = px.line(charts_df , x="date_str", y="moving_average", title=f'{token} Token Balance',labels={
#                      "date_str": "Date",
#                      "moving_average": "Token Balance in USD"})

# fig2.add_hline(y = 0, annotation_text ="Neutral Balance")
# st.plotly_chart(fig2, use_container_width=True)
