import pandas as pd
import networkx as nx
import infomap
import numpy as np
import matplotlib.pyplot as plt
from utils import load_config

CONFIG = load_config()


TOP_X_COMMUNITY = 3
TOP_X_NODE = 50
VISUAL_SCALER = 10000


def process_network_data(shock):
    edges_df = pd.read_csv(f"{CONFIG.NETWORK_DIR}filtered_edges_{shock}.csv")
    nodes_df = pd.read_csv(f"{CONFIG.NETWORK_DIR}filtered_nodes_{shock}.csv")

    edges_df["time"] = pd.to_datetime(edges_df["time"]).astype("int64")
    nodes_df["time"] = pd.to_datetime(nodes_df["time"]).astype("int64")

    def get_node_size(node_id, time):
        node_info = nodes_df[(nodes_df["id"] == node_id) & (nodes_df["time"] == time)]
        return node_info.iloc[0]["size"] if not node_info.empty else None

    time_slices = edges_df["time"].unique()
    communities_with_sizes = {}

    for t in time_slices:
        current_edges = edges_df[edges_df["time"] == t]
        G = nx.DiGraph()
        for index, row in current_edges.iterrows():
            G.add_edge(row["source"], row["target"], weight=np.log1p(row["size"]))

        im = infomap.Infomap("--directed --silent")
        node_to_id = {node: idx for idx, node in enumerate(G.nodes())}
        for source, target, data in G.edges(data=True):
            im.add_link(node_to_id[source], node_to_id[target], data["weight"])
        im.run()

        communities = {}
        for node in im.tree:
            if node.is_leaf:
                original_node_id = list(node_to_id.keys())[
                    list(node_to_id.values()).index(node.node_id)
                ]
                node_size = get_node_size(original_node_id, t)
                communities.setdefault(node.module_id, []).append(
                    (original_node_id, node_size)
                )
        communities_with_sizes[t] = list(communities.values())

    return communities_with_sizes, edges_df


# Visualization function
def visualize_communities(communities_with_sizes, network_type, edges_df):
    num_times = len(communities_with_sizes)
    num_columns = 3
    num_rows = (num_times + 1) // num_columns

    fig, axes = plt.subplots(
        nrows=num_rows, ncols=num_columns, figsize=(48, 16 * num_rows)
    )
    axes = axes.flatten()

    for ax, (time, communities) in zip(axes, communities_with_sizes.items()):
        G_vis = nx.DiGraph()
        top_communities = sorted(
            communities,
            key=lambda x: sum(size for _, size in x if size is not None),
            reverse=True,
        )[:TOP_X_COMMUNITY]
        cmap = plt.cm.get_cmap("tab10", len(top_communities))
        colors = [cmap(i / len(top_communities)) for i in range(len(top_communities))]

        for i, community in enumerate(top_communities):
            top_nodes = sorted(
                community, key=lambda x: x[1] if x[1] is not None else 0, reverse=True
            )[:TOP_X_NODE]
            node_list = [node for node, size in top_nodes if size is not None]
            for node, size in top_nodes:
                if size is not None:
                    G_vis.add_node(
                        node,
                        label=node,
                        size=np.sqrt(size) / VISUAL_SCALER,
                        color=colors[i],
                    )
            for node in node_list:
                relevant_edges = edges_df[
                    (edges_df["time"] == time)
                    & (edges_df["source"] == node)
                    & (edges_df["target"].isin(node_list))
                ]
                for _, edge in relevant_edges.iterrows():
                    G_vis.add_edge(
                        edge["source"],
                        edge["target"],
                        weight=np.sqrt(edge["size"]) / VISUAL_SCALER * 2,
                    )

        pos = nx.kamada_kawai_layout(G_vis, scale=2)
        nx.draw_networkx_nodes(
            G_vis,
            pos,
            ax=ax,
            node_color=[G_vis.nodes[node]["color"] for node in G_vis.nodes],
            node_size=[G_vis.nodes[node]["size"] * 100 for node in G_vis.nodes],
        )
        nx.draw_networkx_edges(
            G_vis,
            pos,
            ax=ax,
            edgelist=G_vis.edges(),
            width=[G_vis[u][v]["weight"] for u, v in G_vis.edges()],
            alpha=0.5,
            arrowstyle="-",
            arrowsize=10,
        )
        nx.draw_networkx_labels(
            G_vis,
            pos,
            ax=ax,
            labels={node: node for node in G_vis.nodes()},
            font_size=10,
        )
        title_color = (
            "red"
            if pd.to_datetime(time, unit="ns").strftime("%Y-%m-%d")
            == CONFIG.SHOCKS[network_type]
            else "black"
        )
        ax.set_title(
            f"Community Structure at Time {pd.to_datetime(time, unit='ns').strftime('%Y-%m-%d')}",
            color=title_color,
        )
        ax.axis("off")

    for i in range(len(communities_with_sizes), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(f"{CONFIG.FIG_DIR}{network_type}.png", dpi=300)


if __name__ == "__main__":
    for network in CONFIG.SHOCKS:
        communities_with_sizes, edges_df = process_network_data(network)
        visualize_communities(communities_with_sizes, network, edges_df)
