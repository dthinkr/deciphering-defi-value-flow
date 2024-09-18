import pandas as pd
import altair as alt
from utils import load_config

CONFIG = load_config()


def calculate_ratio(df: pd.DataFrame) -> pd.DataFrame:
    required_columns = ["time", "type", "inflow", "outflow"]
    if not all(column in df.columns for column in required_columns):
        raise ValueError(
            f"DataFrame must include {', '.join(required_columns)} columns."
        )

    df["ratio"] = float("nan")
    unique_pairs = df.groupby(["time", "type"])

    for (time, type_), group in unique_pairs:
        if group.shape[0] == 2:
            values = list(group["inflow"]) + list(group["outflow"])
            filtered_values = [value for value in values if not pd.isna(value)]

            if len(filtered_values) == 2:
                inflow, outflow = filtered_values
                if outflow != 0:
                    ratio = abs(inflow / outflow)
                    df.loc[
                        (df["time"] == time)
                        & (df["type"] == type_)
                        & ~df["inflow"].isna(),
                        "ratio",
                    ] = ratio

    return df


def create_plots_for_incidents() -> None:
    for incident, incident_date in CONFIG.SHOCKS.items():
        new_nodes_df = pd.read_csv(f"{CONFIG.NETWORK_DIR}filtered_nodes_{incident}.csv")
        new_edges_df = pd.read_csv(f"{CONFIG.NETWORK_DIR}filtered_edges_{incident}.csv")
        new_nodes_df["time"] = pd.to_datetime(new_nodes_df["time"])
        new_edges_df["time"] = pd.to_datetime(new_edges_df["time"])
        new_nodes_df = new_nodes_df[
            ~new_nodes_df["type"].isin(["Privacy & Security", "Unknown"])
        ]

        new_edges_with_types = new_edges_df.merge(
            new_nodes_df[["id", "type"]], left_on="source", right_on="id", how="left"
        ).rename(columns={"type": "source_type"})
        new_edges_with_types = new_edges_with_types.merge(
            new_nodes_df[["id", "type"]], left_on="target", right_on="id", how="left"
        ).rename(columns={"type": "target_type"})

        new_inflow_summary = (
            new_edges_with_types.groupby(["target_type", "time"])
            .agg(inflow=("size", "sum"))
            .unstack(fill_value=0)
        )
        new_outflow_summary = (
            new_edges_with_types.groupby(["source_type", "time"])
            .agg(outflow=("size", "sum"))
            .unstack(fill_value=0)
        )
        new_outflow_summary = -new_outflow_summary

        inflow_long = (
            new_inflow_summary.stack(future_stack=True)
            .reset_index()
            .rename(columns={0: "inflow"})
        )
        outflow_long = (
            new_outflow_summary.stack(future_stack=True)
            .reset_index()
            .rename(columns={0: "outflow"})
        )

        inflow_long = inflow_long.rename(columns={"target_type": "type"})
        outflow_long = outflow_long.rename(columns={"source_type": "type"})

        combined_flow = pd.concat([inflow_long, outflow_long])

        combined_flow = calculate_ratio(combined_flow)
        combined_flow["flow"] = combined_flow["inflow"].fillna(0) + combined_flow[
            "outflow"
        ].fillna(0)
        combined_flow["time"] = pd.to_datetime(combined_flow["time"]).dt.strftime(
            "%Y-%m-%d"
        )
        combined_flow["Sector"] = combined_flow["type"].replace(
            "Asset Management", "Liquid Staking & Asset Management"
        )
        combined_flow["flow"] = combined_flow["flow"] / 1e10

        bars = (
            alt.Chart(combined_flow)
            .mark_bar()
            .encode(
                x=alt.X("time:O", axis=alt.Axis(title="Time", labelAngle=-45)),
                y=alt.Y(
                    "flow:Q",
                    axis=alt.Axis(
                        title="Flow (in billions $)", format="$,.0f", grid=True
                    ),
                ),
                color=alt.condition(
                    alt.datum.time == incident_date, alt.value("red"), "Sector:N"
                ),
                tooltip=[
                    "time",
                    "flow",
                    "Sector",
                    alt.Tooltip("ratio:Q", title="Inflow/Outflow Ratio", format=".2f"),
                ],
            )
        )

        text = bars.mark_text(align="center", baseline="middle", dy=-10).encode(
            text=alt.condition(
                alt.datum.ratio != None,
                alt.Text("ratio:Q", format=".2f"),
                alt.value(""),
            )
        )

        layered_chart = (
            alt.layer(bars, text)
            .properties(width=200, height=300)
            .facet(
                column="Sector:N",
                title={
                    "text": f"Value Flow near {incident.upper()} Collapse ({incident_date})",
                    "subtitle": "Analysis of inflow and outflow by sector. Values show inflow vs. outflow ratio.",
                    "color": "black",
                    "fontSize": 20,
                    "subtitleColor": "gray",
                    "subtitleFontSize": 15,
                },
            )
            .resolve_scale(x="independent", y="shared")
            .configure_axis(gridColor="grey", gridDash=[6, 4], gridWidth=1)
        )

        layered_chart.save(f"{CONFIG.FIG_DIR}{incident}.html")
        layered_chart.save(f"{CONFIG.FIG_DIR}{incident}.svg")
        layered_chart.save(f"doc/figs/value_flow_{incident}.pdf")

        print(f"Plot for {incident} saved successfully.")


if __name__ == "__main__":
    create_plots_for_incidents()
