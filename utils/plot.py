import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

# plt.style.use("science")
import os
from datetime import datetime, timedelta

import matplotlib.dates as mdates
import pandas as pd
import requests

# print(os.environ["PATH"])
# os.environ["PATH"] += os.pathsep + "/usr/local/bin/"
# os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin/"

network_translation = {
    "erdos_renyi": "Random",
    "watts_strogatz": "Small-World",
    "barabasi_albert": "Scale-Free",
    "complete": "Fully Connected",
    "star": "Star",
}


def generate_network_with_authority(
    network_type, n, clusters=0, authority_per_cluster=1
):
    # Generate the base network
    G = None
    if network_type == "erdos_renyi":
        G = nx.erdos_renyi_graph(n, 0.1)
    elif network_type == "watts_strogatz":
        G = nx.watts_strogatz_graph(n, 4, 0.1)
    elif network_type == "barabasi_albert":
        G = nx.barabasi_albert_graph(n, 2)
    elif network_type == "complete":
        G = nx.complete_graph(n)
    elif network_type == "star":
        G = nx.star_graph(n - 1)
    else:
        raise ValueError("Unknown network type")

    # Skip clustering logic if clusters <= 1
    if clusters >= 1:
        nodes_per_cluster = n // clusters
        for cluster_index in range(clusters):
            start_node = cluster_index * nodes_per_cluster
            end_node = start_node + nodes_per_cluster
            cluster_nodes = list(range(start_node, end_node))

            # Add authority nodes to the cluster
            for _ in range(authority_per_cluster):
                authority_node = max(G.nodes) + 1
                edges_to_add = [(authority_node, node) for node in cluster_nodes]
                G.add_edges_from(edges_to_add)
                G.nodes[authority_node][
                    "authority"
                ] = True  # Tag the node as an authority node

    return G


def simulate_SI_model_with_authority(G, initial_infected):
    infected = set(initial_infected)
    propagation_steps = [infected.copy()]
    recovered = set()
    total_messages_sent = 0  # Initialize the counter for messages sent

    while len(infected) < len(G.nodes()):
        new_infected = set()
        for node in infected:
            if node in recovered:
                continue  # Skip recovered nodes
            for neighbor in G.neighbors(node):
                total_messages_sent += 1  # Increment for each neighbor checked
                if neighbor not in infected and neighbor not in recovered:
                    new_infected.add(neighbor)
                    if (
                        "authority" in G.nodes[neighbor]
                    ):  # Check if neighbor is an authority node
                        recovered.add(neighbor)  # Recover authority nodes immediately
                        for n in G.neighbors(
                            neighbor
                        ):  # Infect all neighbors of the authority node
                            total_messages_sent += 1  # Increment for each neighbor of the authority node checked
                            if n not in infected:
                                new_infected.add(n)
        infected.update(new_infected)
        propagation_steps.append(infected.copy())

    return (
        propagation_steps,
        total_messages_sent,
    )  # Return both propagation steps and total messages sent


def plot_all_steps_single_page(network_types, node_size, steps, clusters=0):
    # Calculate grid size
    rows = len(network_types)
    cols = steps
    fig, axs = plt.subplots(
        rows, cols, figsize=(cols * 2, rows * 2)
    )  # Adjust subplot size here

    for i, network_type in enumerate(network_types):
        G = generate_network_with_authority(network_type, node_size, clusters=clusters)
        initial_infected = [list(G.nodes())[0]]
        propagation_steps, _ = simulate_SI_model_with_authority(G, initial_infected)

        for step_number in range(steps):
            ax = axs[i, step_number] if rows > 1 else axs[step_number]
            if step_number < len(propagation_steps):
                step = set(propagation_steps[step_number])
            else:
                step = set(propagation_steps[-1])

            pos = nx.spring_layout(G, seed=42)
            non_infected_nodes = set(G.nodes()) - step
            infected_nodes = step

            node_sizes_non_infected = [
                100 if G.nodes[n].get("authority") else 10 for n in non_infected_nodes
            ]
            node_sizes_infected = [
                100 if G.nodes[n].get("authority") else 10 for n in infected_nodes
            ]

            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=non_infected_nodes,
                node_color="blue",
                node_size=node_sizes_non_infected,
                ax=ax,
            )
            nx.draw_networkx_nodes(
                G,
                pos,
                nodelist=infected_nodes,
                node_color="red",
                node_size=node_sizes_infected,
                ax=ax,
            )
            nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)

            ax.set_title(
                f"{network_translation[network_type]} (Step {step_number})", fontsize=10
            )
            ax.axis("off")

    plt.tight_layout()
    return fig


# Example usage
node_size = 200
network_types = ["erdos_renyi", "watts_strogatz", "barabasi_albert", "complete", "star"]
steps = 5
clusters = 2

fig = plot_all_steps_single_page(network_types, node_size, steps, clusters)

with PdfPages(f"network_propagation_clusters_{clusters}.pdf") as pdf:
    pdf.savefig(fig)


def collect_messages_data(
    network_types, node_sizes, clusters_options=[1], simulations_per_setup=10
):
    results = []
    for network_type in network_types:
        for n in node_sizes:
            for clusters in clusters_options:
                messages = []
                for _ in range(simulations_per_setup):
                    # Always use generate_network_with_authority for consistency
                    G = generate_network_with_authority(
                        network_type, n, clusters=clusters
                    )
                    _, total_messages = simulate_SI_model_with_authority(
                        G, [list(G.nodes())[0]]
                    )
                    messages.append(total_messages)
                avg_messages = np.mean(messages)
                results.append(
                    {
                        "Network Type": network_type,
                        "Nodes": n,
                        "Clusters": clusters,
                        "Average Messages": avg_messages,
                    }
                )
    return results


def plot_messages_vs_nodes(results, filename="plot.png"):
    df_results = pd.DataFrame(results)
    plt.figure(figsize=(10, 8))
    for network_type in network_types:
        for clusters in df_results["Clusters"].unique():
            subset = df_results[
                (df_results["Network Type"] == network_type)
                & (df_results["Clusters"] == clusters)
            ]
            plt.plot(
                subset["Nodes"],
                subset["Average Messages"],
                marker="o",
                linestyle="-",
                label=f"{network_translation[network_type]}, Clusters: {clusters}",
            )

    plt.xlabel("Number of Nodes")
    plt.ylabel("Average Number of Messages Sent")
    plt.title("Average Number of Messages Sent vs. Number of Nodes")
    plt.legend(loc="upper left", frameon=True)
    plt.grid(True)

    # Save the figure to a file
    plt.savefig(filename)
    # Optionally display the plot
    plt.show()


def fetch_prices(coin_id, days=30):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}&interval=day"
    data = requests.get(url).json()["prices"]
    timestamps, prices = zip(
        *[(datetime.fromtimestamp(p[0] / 1000), p[1]) for p in data]
    )
    return timestamps, prices


# uniswap pools
import duckdb
import polars as pl
import altair as alt
from . import load_config


def create_uniswap_v3_tvl_chart():
    CONFIG = load_config()

    query = """
    WITH pool_avg_tvl AS (
        SELECT pool_id, AVG(tvl_usd) as avg_tvl
        FROM uniswap_v3_pool_data
        GROUP BY pool_id
    ),
    top_10_pools AS (
        SELECT pool_id
        FROM pool_avg_tvl
        ORDER BY avg_tvl DESC
        LIMIT 10
    ),
    weekly_data AS (
        SELECT 
            DATE_TRUNC('week', date) as week,
            CASE 
                WHEN u.pool_id IN (SELECT pool_id FROM top_10_pools)
                THEN token0 || '/' || token1
                ELSE 'Other'
            END as pool_category,
            AVG(tvl_usd) as avg_tvl_usd
        FROM uniswap_v3_pool_data u
        GROUP BY 1, 2
    ),
    stats AS (
        SELECT
            PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY avg_tvl_usd) AS Q1,
            PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY avg_tvl_usd) AS Q3
        FROM weekly_data
    )
    SELECT 
        week as date,
        pool_category,
        avg_tvl_usd / 1e9 as tvl_usd_billions
    FROM weekly_data, stats
    WHERE avg_tvl_usd BETWEEN (Q1 - 1.5 * (Q3 - Q1)) AND (Q3 + 1.5 * (Q3 - Q1))
    ORDER BY week, pool_category
    """

    with duckdb.connect(
        f"md:{CONFIG.CONSTRUCT_TVL_FOLDER_NAME}?motherduck_token={CONFIG.MD_KEY}"
    ) as conn:
        df = conn.execute(query).pl()
    # Calculate category totals for sorting
    category_total = (
        df.group_by("pool_category")
        .agg(pl.col("tvl_usd_billions").mean())
        .sort("tvl_usd_billions", descending=True)
    )

    # Define color palette (FT-inspired)
    color_palette = [
        "#0f5499",
        "#990f3d",
        "#f2dfce",
        "#000000",
        "#8400cd",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
    ]

    # Create the chart using Altair
    chart = (
        alt.Chart(df.to_pandas())
        .mark_area()
        .encode(
            x=alt.X(
                "date:T",
                axis=alt.Axis(
                    title="",
                    format="%b %Y",
                    labelAngle=0,
                    tickCount=10,
                    labelOverlap=True,
                ),
            ),
            y=alt.Y(
                "tvl_usd_billions:Q",
                stack="zero",
                title="Total Value Locked (Billion USD)",
            ),
            color=alt.Color(
                "pool_category:N",
                sort=category_total["pool_category"].to_list(),
                scale=alt.Scale(
                    domain=category_total["pool_category"].to_list(),
                    range=color_palette,
                ),
                legend=alt.Legend(title="Pool", orient="top", columns=3),
            ),
            order=alt.Order("pool_category:N", sort="descending"),
            opacity=alt.condition(
                alt.datum.pool_category == "Other", alt.value(0.9), alt.value(0.8)
            ),
            tooltip=[
                "date:T",
                "pool_category:N",
                alt.Tooltip("tvl_usd_billions:Q", format="$.2f"),
            ],
        )
        .properties(width=800, height=400)
    )

    # Add gridlines
    gridlines = (
        alt.Chart(df.to_pandas())
        .mark_rule(color="#d9d9d9", strokeDash=[1, 1])
        .encode(x="date:T", y="tvl_usd_billions:Q")
    )

    # Combine area chart and gridlines
    final_chart = (
        (gridlines + chart)
        .properties(
            title={
                "text": "Uniswap V3: Top 10 Pools by Total Value Locked",
                "subtitle": [
                    "Weekly average data, ranked by all-time average TVL",
                    "Source: The Graph, UniSwap | Chart: dFMI",
                ],
                "color": "black",
                "fontSize": 28,
                "subtitleFontSize": 18,
                "subtitleColor": "#666666",
                "font": "outfit",
                "subtitleFont": "outfit",
            }
        )
        .configure_axis(
            labelFont="outfit", titleFont="outfit", labelFontSize=14, titleFontSize=16
        )
        .configure_legend(
            labelFont="outfit", titleFont="outfit", labelFontSize=14, titleFontSize=16
        )
        .configure_title(font="outfit", fontSize=24)
        .configure_view(strokeWidth=0)
    )

    return final_chart


# # Usage
# alt.data_transformers.enable('default', max_rows=None)
# chart = create_uniswap_v3_tvl_chart()
# chart.save('uniswap_v3_top_pools_tvl.png', scale_factor=2.0)
# chart
