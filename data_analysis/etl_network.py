import ast
import json
import os

import numpy as np
import pandas as pd
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import load_config

CONFIG = load_config()

try:
    NUM_CORES = max(1, int(os.cpu_count() * 0.8))
except:
    NUM_CORES = 1


class ETLNetwork:
    def __init__(self, wrapper):
        self.wrapper = wrapper
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        self.mapping_path = os.path.join(project_root, "data", "mapping")
        self.rev_map, self.categories, self.id_to_info = {}, {}, {}
        required_files = ["rev_map.json", "token_to_protocol.json", "id_to_info.json"]
        missing_files = [
            file
            for file in required_files
            if not os.path.exists(os.path.join(self.mapping_path, file))
        ]
        if missing_files:
            print(f"Missing files: {missing_files}")
            self.update_mapping()
        self._load_mappings()

    def _retrieve_update_date(self, tables=[CONFIG.TABLES["A"], CONFIG.TABLES["C"]]):
        for table in tables:
            self.wrapper._get_last_modified_time(table)
        raise NotImplementedError

    def _load_mappings(self):
        self.rev_map = self._load_json("rev_map.json")
        self.categories = self._load_json("token_to_protocol.json")
        self.id_to_info = self._load_json("id_to_info.json")

    def _load_json(self, filename):
        with open(os.path.join(self.mapping_path, filename), "r") as file:
            return json.load(file)

    def _default_handler(self, obj: object) -> object:
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise TypeError(
                f"Object of type {obj.__class__.__name__} is not JSON serializable"
            )

    def _save_json(self, data: dict, filename: str):
        path = os.path.join(self.mapping_path, filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as json_file:
            json.dump(data, json_file, indent=4, default=self._default_handler)

    def _similarity_scores(self, categories, A):
        """
        Calculates similarity scores between tokens and protocols to map each token to the most similar protocol.

        This method uses TF-IDF vectorization and cosine similarity to compare the textual representation of tokens
        against a set of protocols. Each token is then mapped to the protocol with the highest similarity score,
        provided the score exceeds a predefined threshold.

        Parameters:
        - categories (dict): A dictionary containing token categories with their respective tokens and metadata.
        - A (pd.DataFrame): A DataFrame containing protocol information, including textual descriptions.

        Returns:
        - dict: A mapping of token names to their most similar protocol, including the protocol's ID and name.
        """
        input_tokens = {}
        for category_name, tokens in categories.items():
            for token, data in tokens.items():
                if isinstance(data["id"], str):
                    input_tokens[token] = data

        token_names = list(input_tokens.keys())
        combined_texts = token_names + A["all_text"].tolist()
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(combined_texts)
        tfidf_tokens = tfidf_matrix[: len(token_names)]
        tfidf_A_texts = tfidf_matrix[len(token_names) :]
        similarity_scores = cosine_similarity(tfidf_tokens, tfidf_A_texts)

        best_matches_indices = similarity_scores.argmax(axis=1)
        best_scores = similarity_scores.max(axis=1)
        threshold = CONFIG.SIMILARIY_THRESHOLD
        token_to_protocol = {}

        for i, token_name in enumerate(token_names):
            if best_scores[i] >= threshold:
                best_match_index = best_matches_indices[i]
                matched_id = A.iloc[best_match_index]["id"]
                matched_name = A.iloc[best_match_index]["name"]
                token_to_protocol[token_name] = {"id": matched_id, "name": matched_name}

        return token_to_protocol

    def update_mapping(self):
        """
        Updates the internal mappings of tokens to protocols based on the latest data from BigQuery.

        This method fetches the latest data from BigQuery, processes it to update the reverse mapping (rev_map),
        token to protocol mapping (categories), and the ID to information mapping (id_to_info). It handles the
        extraction, transformation, and loading (ETL) of token and protocol data to ensure the mappings reflect
        the current state of the blockchain data. The updated mappings are then saved to local JSON files.

        The process involves:
        - Fetching the latest protocol and token data from BigQuery.
        - Calculating total TVLs for protocols and sorting them.
        - Updating the reverse mapping for tokens to their respective protocol IDs.
        - Classifying tokens into categories and updating the token to protocol mappings based on similarity scores and predefined rules.
        - Saving the updated mappings to local JSON files for use in data processing and analysis.
        """
        A = self.wrapper.get_dataframe(CONFIG.TABLES["A"]).to_pandas()
        tvl_column = (
            "currentChainTvls" if "currentChainTvls" in A.columns else "chainTvls"
        )
        A["totalTvls"] = A[tvl_column].apply(
            lambda x: sum(ast.literal_eval(x).values())
            if isinstance(x, str) and x.strip() != ""
            else 0
        )

        A = A.sort_values(by="totalTvls", ascending=False)

        # Update Rev Map
        rev_map = {}
        for idx, row in A.iterrows():
            if pd.notna(row["assetToken"]) and row["assetToken"] != "-":
                rev_map[row["assetToken"]] = row["id"]
            if pd.notna(row["symbol"]) and row["symbol"] != "-":
                rev_map[row["symbol"]] = row["id"]

        self._save_json(rev_map, "rev_map.json")

        unique_token_names = (
            self.wrapper.get_unique_token_names(CONFIG.TABLES["C"]).to_pandas().dropna()
        )
        frequency_df = self.wrapper.get_token_frequency(CONFIG.TABLES["C"]).to_pandas()
        frequency_dict = frequency_df.set_index("token_name")["frequency"].to_dict()
        categories = {"MAP": {}, "LP": {}, "UNKNOWN": {}, "PRIMARY": {}, "OTHER": {}}
        for idx, row in unique_token_names.iterrows():
            token_name = row["token_name"]
            category = ""
            if token_name in rev_map:
                category = "MAP"
            elif "LP" in token_name:
                category = "LP"
            elif "UNKNOWN" in token_name:
                category = "UNKNOWN"
            elif "-" in token_name:
                category = "OTHER"
            else:
                category = "PRIMARY"
            categories[category][token_name] = {
                "id": rev_map.get(token_name, None),
                "frequency": frequency_dict.get(
                    token_name, 0
                ),  # Default to 0 if not found
            }

        """
            - update the PRIMARY category and mark the rest as either MAP, LP, UNKNOWN, or OTHER
            - this is done using a manually defined mapping in config.config
        """
        sorted_primary = sorted(
            categories["PRIMARY"].items(), key=lambda x: x[1]["frequency"], reverse=True
        )
        categories["PRIMARY"] = {token: data for token, data in sorted_primary}

        for token, protocol_name in CONFIG.PRIMARY_TOKEN_TO_PROTOCOL.items():
            matching_row = A[A["name"] == protocol_name]
            if not matching_row.empty:
                matching_id = matching_row.iloc[0]["id"]
                if token in categories["PRIMARY"]:
                    categories["PRIMARY"][token]["id"] = matching_id

        for category_name, tokens in categories.items():
            for token, data in tokens.items():
                if data["id"] is None:
                    if category_name == "PRIMARY":
                        categories[category_name][token]["id"] = token
                    else:
                        categories[category_name][token]["id"] = category_name
        """ 
            - further update using similarity scores
            - read ummapped tokens
            - find the most similar token in the mapping
            - if the similarity score is high, we can use the mapping
            - update and overwrite the old mapping
        """
        A["all_text"] = A.apply(
            lambda x: " ".join(x.fillna("").replace("None", "").astype(str)), axis=1
        )
        mapped_tokens = self._similarity_scores(categories, A)

        for token_name, mapping_info in mapped_tokens.items():
            for category_name, tokens in categories.items():
                if token_name in tokens:
                    categories[category_name][token_name]["id"] = mapping_info["id"]
                    break

        self._save_json(categories, "token_to_protocol.json")

        # Update ID to Info
        detailed_to_broad = {
            det: broad
            for broad, dets in CONFIG.CATEGORY_MAPPING.items()
            for det in dets
        }
        id_to_info = {
            row["id"]: {
                "name": row["name"],
                "category": detailed_to_broad.get(row["category"], "Unknown"),
            }
            for index, row in A.iterrows()
        }

        self._save_json(id_to_info, "id_to_info.json")

    def calculate_network(self, df: pl.DataFrame) -> dict:
        processed_df = self.process_data(df)
        network_data = self.generate_network_data(processed_df)
        return network_data

    def process_data(self, df: pl.DataFrame) -> pl.DataFrame:
        # Initialize the new columns with None for 'from' and 'to' values
        df = df.with_columns(
            [
                pl.lit(None).alias("qty_from"),
                pl.lit(None).alias("qty_to"),
                pl.lit(None).alias("usd_from"),
                pl.lit(None).alias("usd_to"),
            ]
        )

        # Assign 'from' and 'to' values based on the sign of 'usd_change'
        df = df.with_columns(
            [
                pl.when(df["usd_change"] < 0).then(df["qty"]).alias("qty_from"),
                pl.when(df["usd_change"] < 0).then(df["usd"]).alias("usd_from"),
                pl.when(df["usd_change"] >= 0).then(df["qty"]).alias("qty_to"),
                pl.when(df["usd_change"] >= 0).then(df["usd"]).alias("usd_to"),
            ]
        )

        # Apply the function to find token_id for each row in DataFrame 'df'
        df = df.with_columns(
            [
                pl.col("token_name")
                .apply(
                    lambda token_name: self._find_token_id(token_name, self.categories),
                    return_dtype=pl.Utf8,
                )
                .alias("token_id"),
                pl.col("id").cast(pl.Utf8).alias("id"),
            ]
        )

        # Vectorized operation to set from_node and to_node based on usd_change
        df = df.with_columns(
            [
                pl.when(df["usd_change"] < 0)
                .then(df["id"])
                .otherwise(df["token_id"])
                .alias("from_node"),
                pl.when(df["usd_change"] < 0)
                .then(df["token_id"])
                .otherwise(df["id"])
                .alias("to_node"),
            ]
        )

        # Ensure token_name is included in the final DataFrame
        df = df.with_columns([pl.col("token_name")])

        # Select only the required columns and adjust 'qty_change' and 'usd_change' to be absolute values
        df = df.select(
            [
                "from_node",
                "to_node",
                "chain_name",
                "qty_from",
                "qty_to",
                "usd_from",
                "usd_to",
                "token_name",  # Include token_name in the output
                pl.col("qty_change").abs().alias("qty_flow"),
                pl.col("usd_change").abs().alias("usd_flow"),
            ]
        )

        return df

    def _find_token_id(self, token_name: str, categories: dict) -> str:
        for category, tokens in categories.items():
            if token_name in tokens:
                return str(tokens[token_name].get("id", None))
        return None

    def calculate_historical_network(self, df: pl.DataFrame) -> dict:
        # extract unique periods
        unique_periods = df["period"].unique().to_list()

        # hold network json for each period
        network_data_by_period = {}

        for period in unique_periods:
            period_df = df.filter(pl.col("period") == period)
            network_data = self.calculate_network(period_df)
            network_data_by_period[period] = network_data
        return network_data_by_period

    def generate_network_data(self, df: pl.DataFrame) -> dict:
        # Identify all unique nodes
        all_nodes = pl.concat(
            [
                df.select(pl.col("from_node").alias("node")),
                df.select(pl.col("to_node").alias("node")),
            ]
        ).unique()

        # Aggregate USD values for nodes appearing as 'from_node' with token details
        from_agg = (
            df.filter(pl.col("usd_from").is_not_null())
            .group_by(["from_node", "token_name"])
            .agg(usd_from_total=pl.col("usd_from").sum())
            .rename(
                {"from_node": "node", "token_name": "token", "usd_from_total": "amount"}
            )
        )

        # Aggregate USD values for nodes appearing as 'to_node' with token details
        to_agg = (
            df.filter(pl.col("usd_to").is_not_null())
            .group_by(["to_node", "token_name"])
            .agg(usd_to_total=pl.col("usd_to").sum())
            .rename(
                {"to_node": "node", "token_name": "token", "usd_to_total": "amount"}
            )
        )

        # Combine the two aggregations
        node_composition = (
            from_agg.vstack(to_agg)
            .group_by(["node", "token"])
            .agg(total_amount=pl.col("amount").sum())
        )

        # Calculate total size for each node
        node_sizes = node_composition.group_by("node").agg(
            total_size=pl.col("total_amount").sum()
        )

        # Ensure all nodes are included
        node_sizes = all_nodes.join(node_sizes, on="node", how="left").fill_null(0)

        # Prepare data for link aggregation
        df_links = (
            df.with_columns(
                [
                    pl.when(pl.col("from_node") < pl.col("to_node"))
                    .then(
                        pl.struct(
                            [
                                pl.col("from_node").alias("source"),
                                pl.col("to_node").alias("target"),
                                pl.col("usd_flow").alias("flow"),
                                pl.col("token_name").alias("token"),
                            ]
                        )
                    )
                    .otherwise(
                        pl.struct(
                            [
                                pl.col("to_node").alias("source"),
                                pl.col("from_node").alias("target"),
                                -pl.col("usd_flow").alias("flow"),
                                pl.col("token_name").alias("token"),
                            ]
                        )
                    )
                    .alias("link_info")
                ]
            )
            .select("link_info")
            .unnest("link_info")
        )

        # Aggregate links to ensure uniqueness and calculate net flow
        links_agg = df_links.group_by(["source", "target", "token"]).agg(
            net_flow=pl.col("flow").sum()
        )

        # Create links with token details and composition
        links = links_agg.with_columns(
            [
                pl.col("net_flow").alias("size"),
                pl.struct(
                    [pl.col("token").alias("token"), pl.col("net_flow").alias("size")]
                ).alias("composition"),
            ]
        )

        # Convert to JSON format
        nodes_json = node_sizes.to_dicts()
        links_json = links.to_dicts()

        # Format nodes for output, including total size and composition
        nodes_output = [
            {"id": node["node"], "size": node["total_size"], "composition": {}}
            for node in nodes_json
        ]

        # Update composition for nodes
        for node in nodes_output:
            if node["id"] is not None:
                compositions = node_composition.filter(
                    pl.col("node") == node["id"]
                ).to_dicts()
            else:
                compositions = node_composition.filter(
                    pl.col("node").is_null()
                ).to_dicts()
            node["composition"] = {
                comp["token"]: comp["total_amount"] for comp in compositions
            }

        # Format links for output, including token details and composition
        links_output = [
            {
                "source": link["source"],
                "target": link["target"],
                "size": link["size"],
                "composition": {
                    link["composition"]["token"]: link["composition"]["size"]
                },
            }
            for link in links_json
        ]

        links_output = self._merge_links(links_output)

        network_data = {"nodes": nodes_output, "links": links_output}

        return network_data

    def _merge_links(self, links_output: list) -> list:
        merged_links = {}
        for link in links_output:
            key = (link["source"], link["target"])
            reverse_key = (link["target"], link["source"])

            # Determine whether to use the normal or reversed key based on existing entries and size
            if key in merged_links:
                target_key = key
            elif reverse_key in merged_links and link["size"] < 0:
                target_key = reverse_key
            else:
                target_key = key

            if target_key not in merged_links:
                merged_links[target_key] = {
                    "source": target_key[0],
                    "target": target_key[1],
                    "size": 0,
                    "composition": {},
                }

            merged_link = merged_links[target_key]

            # Adjust size and composition based on the direction of the link
            if target_key == key:
                merged_link["size"] += link["size"]
                composition_update = link["composition"]
            else:
                merged_link["size"] -= link["size"]
                composition_update = {
                    token: -size for token, size in link["composition"].items()
                }

            # Update composition
            for token, size in composition_update.items():
                if token in merged_link["composition"]:
                    merged_link["composition"][token] += size
                else:
                    merged_link["composition"][token] = size

        # Final adjustment: check for negative sizes and swap if necessary
        for link in list(merged_links.values()):
            if link["size"] < 0:
                link["size"] = -link["size"]
                link["source"], link["target"] = link["target"], link["source"]
                link["composition"] = {
                    token: -size for token, size in link["composition"].items()
                }

        return list(merged_links.values())

    def generate_historical_network_edges(
        self, time_granularity: str, start_date: str, size_threshold: int = None
    ) -> str:
        """
        Generates a filtered network CSV that contains the entire history based on granularity.
        If a size threshold is provided, only links with a size greater than the threshold are included.
        """

        df = self.wrapper.get_historical_changes(time_granularity, start_date)
        historical_network_data = self.calculate_historical_network(df)

        data = []
        for date, content in historical_network_data.items():
            for link in content["links"]:
                row = {
                    "time": date,
                    "source": self._map_id_to_name(link["source"]),
                    "target": self._map_id_to_name(link["target"]),
                    "size": link["size"],
                }
                data.append(row)

        df = pl.DataFrame(data)
        if size_threshold is not None:
            df = df.filter(pl.col("size") > size_threshold)

        return df.write_csv(file=None)

    def _map_id_to_name(self, id):
        return self.id_to_info.get(str(id), {}).get("name", str(id))

    def generate_historical_network_nodes(
        self, time_granularity: str, start_date: str, size_threshold: int = None
    ) -> str:
        """
        Generates a filtered network CSV that contains node information based on granularity,
        including a 'type' column that represents the category of each node.
        If a size threshold is provided, only nodes with a size greater than the threshold are included.
        """

        df = self.wrapper.get_historical_changes(time_granularity, start_date)
        historical_network_data = self.calculate_historical_network(df)

        data = []
        for date, content in historical_network_data.items():
            for node in content["nodes"]:
                node_id = node["id"]
                node_info = self.id_to_info.get(str(node_id), {})
                node_size = node["size"]
                if size_threshold is None or node_size > size_threshold:
                    row = {
                        "time": date,
                        "id": node_info.get("name", "Unknown"),
                        "size": node_size,
                        "type": node_info.get("category", "Unknown"),
                    }
                    data.append(row)

        df = pl.DataFrame(data)
        return df.write_csv(file=None)
