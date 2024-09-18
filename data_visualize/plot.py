import plotly.express as px
import polars as pl
from pyobsplot import Obsplot, Plot
from pyvis.network import Network


def save_heatmap(df: pl.DataFrame, plot_path: str = "data/protocol_activity_heatmap"):
    op = Obsplot(renderer="jsdom")
    plot_spec = {
        "height": 600,
        "grid": True,
        "marks": [
            Plot.dot(
                df,
                {
                    "x": "period",
                    "y": "id",
                    "fill": "entry_count",
                    "r": 2,
                    "fillOpacity": 0.5,
                },
            )
        ],
        "x": {"grid": True, "label": "Period"},
        "y": {"grid": True, "label": "ID", "sort": "ascending", "type": "linear"},
        "color": {"scale": "sqrt", "legend": False},
    }

    Plot.plot(plot_spec, path=plot_path + ".html")
    op(plot_spec, path=plot_path + ".svg")


def generate_cumulation_plot(
    all_years_df: pl.DataFrame, plot_path: str, percentage_line: int
):
    op = Obsplot(renderer="jsdom")
    op(
        {
            "marks": [
                Plot.dot(
                    all_years_df.filter(pl.col("year") == year),
                    {
                        "x": "protocol_amount",
                        "y": "cumulative_share_percentage",
                        "fill": "legend_label",
                    },
                )
                for year in all_years_df["year"].unique().to_list()
            ]
            + [
                Plot.ruleY(
                    [percentage_line],
                    {"stroke": "red", "strokeWidth": 2, "strokeDasharray": "10,10"},
                )
            ],
            "grid": True,
            "color": {"legend": True},
        },
        path=plot_path,
    )
    with open(plot_path, "r") as file:
        html_content = file.read()
    return html_content


def generate_line_plot(data, label, plot_path, stroke_field="type", legend=True):
    op = Obsplot(renderer="jsdom")
    plot_spec = {
        "marginLeft": 50,
        "width": 928,
        "height": 600,
        "y": {"grid": True, "label": label},
        "color": {"legend": legend},
        "marks": [
            Plot.lineY(
                data,
                {
                    "x": "aggregated_date",
                    "y": "usd",
                    "stroke": stroke_field,
                    "tip": True,
                },
            ),
            Plot.ruleY([0]),
        ],
    }
    op(plot_spec, path=plot_path)
    with open(plot_path, "r") as file:
        html_content = file.read()
    return html_content


def genereate_treemap_plot(all_years_df: pl.DataFrame):
    all_years_df_pandas = all_years_df.to_pandas()
    all_years_df_pandas = all_years_df_pandas[
        all_years_df_pandas["individual_share_percentage"].notna()
        & (all_years_df_pandas["individual_share_percentage"] > 0)
    ]
    fig = px.treemap(
        all_years_df_pandas,
        path=["year", "name"],
        values="total_usd_value",
        color="individual_share_percentage",
        color_continuous_scale="Greys",
    )
    fig.update_layout(coloraxis_showscale=False)
    return fig


def visualize_top_nodes(
    data,
    id_to_name=None,
    top_x=100,
    min_visual_node=10,
    max_visual_node=30,
    min_visual_edge=1,
    max_visual_edge=5,
):
    net = Network(notebook=False, height="750px", width="100%", directed=True)
    net.use_CDN = True
    if id_to_name:
        for node in data["nodes"]:
            node["id"] = id_to_name.get(str(node["id"]), node["id"])
        for link in data["links"]:
            link["source"] = id_to_name.get(str(link["source"]), link["source"])
            link["target"] = id_to_name.get(str(link["target"]), link["target"])

    top_nodes = sorted(data["nodes"], key=lambda x: x["size"], reverse=True)[:top_x]
    top_node_ids = {node["id"] for node in top_nodes}

    def normalize_size(size, min_size, max_size, min_visual, max_visual):
        return (
            ((size - min_size) / (max_size - min_size)) * (max_visual - min_visual)
            + min_visual
            if max_size > min_size
            else min_visual
        )

    min_node_size = min((node["size"] for node in top_nodes), default=0)
    max_node_size = max((node["size"] for node in top_nodes), default=1)

    all_edge_sizes = [
        link["size"]
        for link in data["links"]
        if link["source"] in top_node_ids and link["target"] in top_node_ids
    ]
    min_edge_size = min(all_edge_sizes, default=1)
    max_edge_size = max(all_edge_sizes, default=1)

    for node in top_nodes:
        normalized_node_size = normalize_size(
            node["size"], min_node_size, max_node_size, min_visual_node, max_visual_node
        )
        net.add_node(node["id"], label=node["id"], size=normalized_node_size)

    for link in data["links"]:
        if link["source"] in top_node_ids and link["target"] in top_node_ids:
            normalized_edge_size = normalize_size(
                link["size"],
                min_edge_size,
                max_edge_size,
                min_visual_edge,
                max_visual_edge,
            )
            net.add_edge(link["source"], link["target"], value=normalized_edge_size)

    return net
