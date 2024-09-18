import sys
import os
from pathlib import Path

# Add the project root directory to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import atexit
import tempfile
from datetime import datetime

import pandas as pd
import polars as pl
import streamlit as st

from data_access import MotherDuckWrapper
from data_visualize import (
    generate_cumulation_plot,
    generate_line_plot,
    genereate_treemap_plot,
)
from utils import load_config

temp_files = []

CONFIG = load_config()


def get_temp_file_path(suffix=".html"):
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    temp_file_path = temp_file.name
    temp_file.close()
    temp_files.append(temp_file_path)
    return temp_file_path


def cleanup_temp_files():
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.unlink(file_path)
    print("Cleaned up all temporary files.")


atexit.register(cleanup_temp_files)


def generate_first_days_of_years(years=5):
    return [
        datetime(year, 1, 1).timestamp()
        for year in range(datetime.now().year - years + 1, datetime.now().year + 1)
    ]


@st.cache_data
def prepare_historical_data(_wrapper, granularity: str = "month"):
    df = _wrapper.get_aggregated_data(granularity=granularity)

    df_global = (
        df.group_by(["type", "aggregated_date"])
        .agg(pl.sum("usd").alias("usd"))
        .sort(["type", "aggregated_date"])
    )

    protocol_avg = df.group_by(["protocol_name"]).agg(avg_usd=pl.mean("usd"))

    # Filter to keep only the top protocols
    top_protocols = protocol_avg.sort("avg_usd", descending=True).head(CONFIG.TOP_X)[
        "protocol_name"
    ]

    # Filter the original dataframe to include only the top protocols
    df_top_protocols = df.filter(pl.col("protocol_name").is_in(top_protocols))

    # Group and aggregate the data for each type and protocol
    df_aggregated = (
        df_top_protocols.group_by(["type", "protocol_name", "aggregated_date"])
        .agg(pl.sum("usd").alias("usd"))
        .sort(["type", "protocol_name", "aggregated_date"])
    )

    return df_global, df_aggregated


@st.cache_data
def prepare_top_data(_wrapper):
    timestamps = generate_first_days_of_years()
    all_years_df = pl.DataFrame()
    for unix_timestamp in timestamps:
        df = _wrapper.get_protocol_tvl_distribution_on_day_with_names(unix_timestamp)
        if df is not None:
            df = df.sort("total_usd_value", descending=True)
            total_tvl_day = df["total_usd_value"].sum()
            df = df.with_columns(
                [
                    (df["total_usd_value"].cum_sum() / total_tvl_day).alias(
                        "cumulative_share"
                    ),
                    (df["total_usd_value"] / total_tvl_day).alias("individual_share"),
                    ((df["total_usd_value"].cum_sum() / total_tvl_day) * 100).alias(
                        "cumulative_share_percentage"
                    ),
                    ((df["total_usd_value"] / total_tvl_day) * 100).alias(
                        "individual_share_percentage"
                    ),
                    pl.lit(pd.to_datetime(unix_timestamp, unit="s").year).alias("year"),
                ]
            )
            all_years_df = pl.concat([all_years_df, df])

    all_years_df = all_years_df.with_columns(all_years_df["year"].cast(pl.Utf8))

    all_years_df = (
        all_years_df.sort(["year", "cumulative_share_percentage"])
        .group_by("year")
        .map_groups(
            lambda group: group.with_columns(
                pl.arange(0, group.height, eager=True).alias("protocol_amount")
            )
        )
    )
    return all_years_df


def process_top_data(wrapper):
    all_years_df = prepare_top_data(wrapper)
    percentage_line = st.slider(
        "Percentage Threshold", min_value=0, max_value=100, value=90, step=1
    )
    all_years_df = all_years_df.with_columns(
        (all_years_df["cumulative_share_percentage"] < percentage_line).alias(
            "is_below_threshold"
        )
    )

    summary = all_years_df.group_by("year").agg(
        [
            pl.sum("is_below_threshold").alias("below_threshold"),
            pl.count("is_below_threshold").alias("total"),
        ]
    )

    summary = summary.with_columns(
        (
            pl.col("year").cast(pl.Utf8)
            + " ("
            + pl.col("below_threshold").cast(pl.Utf8)
            + "/"
            + pl.col("total").cast(pl.Utf8)
            + ")"
        ).alias("legend_label")
    )
    all_years_df = all_years_df.join(summary, on="year", how="left")
    return all_years_df, percentage_line


def main():
    st.title("Protocol TVL Distribution Analysis")
    wrapper = MotherDuckWrapper()
    all_years_df, percentage_line = process_top_data(wrapper)

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "Data Overview",
            "Cumulative Distribution Plot",
            "Distribution Treemap",
            "Experimental",
        ]
    )

    with tab1:
        st.dataframe(all_years_df)

    with tab2:
        if st.button("Show Cumulative Distribution Plot", key="cumulative"):
            html_content = generate_cumulation_plot(
                all_years_df, get_temp_file_path(), percentage_line
            )
            st.markdown(html_content, unsafe_allow_html=True)

    with tab3:
        if st.button("Show Distribution Treemap", key="treemap"):
            fig = genereate_treemap_plot(all_years_df)
            st.plotly_chart(fig)

    with tab4:
        if st.button("Show Experimental Plots", key="experimental"):
            df_global, df_aggregated = prepare_historical_data(wrapper)
            html_global = generate_line_plot(
                df_global,
                "Total Value Locked (USD)",
                get_temp_file_path(),
                stroke_field="type",
            )
            st.markdown("TVL for all types")
            st.markdown(html_global, unsafe_allow_html=True)

            for type_group in df_aggregated["type"].unique():
                st.markdown(type_group)
                df_type = df_aggregated.filter(pl.col("type") == type_group)
                html_type = generate_line_plot(
                    df_type,
                    f"{type_group} Value Locked (USD)",
                    get_temp_file_path(),
                    stroke_field="protocol_name",
                )
                st.markdown(html_type, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
