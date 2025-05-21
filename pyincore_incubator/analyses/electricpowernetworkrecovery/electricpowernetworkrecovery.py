__author__ = "Kooshan Amini"

import random
import networkx as nx
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point
from scipy.spatial import cKDTree
from collections import deque, defaultdict, OrderedDict
import concurrent.futures
import warnings
from pyincore import BaseAnalysis

warnings.filterwarnings("ignore")


class ElectricPowerNetworkRecovery(BaseAnalysis):
    """ElectricPowerNetworkRecovery class for power network recovery."""

    def __init__(self, incore_client):
        super(ElectricPowerNetworkRecovery, self).__init__(incore_client)
        self.crs = "epsg:4326"
        self.sample_num = 10
        self.num_crew = 20
        self.seed_num = 1234

    def run(self):
        """Executes power network analysis."""
        self.sample_num = self.get_parameter("num_samples")
        self.num_crew = self.get_parameter("num_crews")
        node_data = self.get_input_dataset("nodes").get_dataframe_from_shapefile()
        edge_data = self.get_input_dataset("edges").get_dataframe_from_shapefile()
        building_data = self.get_input_dataset(
            "buildings"
        ).get_dataframe_from_shapefile()
        pole_damage_mcs_samples_df = self.get_input_dataset(
            "pole_damage"
        ).get_dataframe_from_csv()

        self.create_graph(node_data=node_data, edge_data=edge_data)

        (
            power_final_result_average,
            power_final_result_all_samples,
        ) = self.recovery_analysis(
            building_shapefile=building_data,
            pole_damage_mcs_samples_df=pole_damage_mcs_samples_df,
            num_thread=self.get_parameter("num_thread"),
            time_step=self.get_parameter("time_step"),
            consider_protective_devices=self.get_parameter(
                "consider_protective_devices"
            ),
        )

        # Convert GeoDataFrame to DataFrame without geometry
        result_df = power_final_result_average.drop(columns="geometry")
        samples_df = power_final_result_all_samples.drop(columns="geometry")

        # Convert DataFrame to list of OrderedDict
        result_list = result_df.to_dict(orient="records", into=OrderedDict)
        samples_list = samples_df.to_dict(orient="records", into=OrderedDict)

        # Generate final outputs
        self.set_result_csv_data(
            "result",
            samples_list,
            name=self.get_parameter("result_name") + "_all_samples",
        )
        self.set_result_csv_data(
            "result",
            result_list,
            name=self.get_parameter("result_name") + "_average_return_time_in_days",
        )

        return True

    def get_spec(self):
        """Get specifications of the power network analysis.

        Returns:
            obj: A JSON object of specifications of the power network analysis result.

        """
        return {
            "name": "powernetworkanalysis",
            "description": "Power Network Analysis for damage and recovery",
            "input_parameters": [
                {
                    "id": "result_name",
                    "required": True,
                    "description": "Result dataset name",
                    "type": str,
                },
                {
                    "id": "num_thread",
                    "required": True,
                    "description": "Number of threads to use",
                    "type": int,
                },
                {
                    "id": "num_samples",
                    "required": True,
                    "description": "Number of samples to run",
                    "type": int,
                },
                {
                    "id": "num_crews",
                    "required": True,
                    "description": "Number of crews for power pole restoration",
                    "type": int,
                },
                {
                    "id": "time_step",
                    "required": True,
                    "description": "Time step for recovery analysis",
                    "type": int,
                },
                {
                    "id": "consider_protective_devices",
                    "required": True,
                    "description": "Consider protective devices in the analysis",
                    "type": bool,
                },
            ],
            "input_datasets": [
                {
                    "id": "nodes",
                    "required": True,
                    "description": "Node dataset",
                    "type": "incore:epfVer2",
                },
                {
                    "id": "edges",
                    "required": True,
                    "description": "Edge dataset",
                    "type": "incore:epfVer2",
                },
                {
                    "id": "buildings",
                    "required": True,
                    "description": "Building dataset",
                    "type": "ergo:buildingInventoryVer7",
                },
                {
                    "id": "pole_damage",
                    "required": True,
                    "description": "Pole damage samples dataset",
                    "type": "incore:sampleFailureState",
                },
            ],
            "output_datasets": [
                {
                    "id": "result",
                    "parent_type": "buildings",
                    "description": "GeoDataFrame of power back time results",
                    "type": "ergo:buildingPowerRecoveryVer1",
                }
            ],
        }

    def graph_nodes_to_gdf(self, graph=None):
        # create a geopandas dataframe for the nodes
        nodes = []
        for node in graph.nodes():
            point = graph.nodes[node]["geometry"]
            attrs = {
                attr: graph.nodes[node][attr]
                for attr in graph.nodes[node]
                if attr != "geometry"
            }
            nodes.append({"geometry": point, **attrs})

        nodes_gdf = gpd.GeoDataFrame(nodes, crs=self.crs)

        return nodes_gdf

    def graph_edges_to_gdf(self, graph=None):
        # create a geopandas dataframe for the edges
        edges = []
        for edge in graph.edges():
            line = graph.edges[edge]["geometry"]
            edges.append({"geometry": line})

        edges_gdf = gpd.GeoDataFrame(edges, crs=self.crs)

        # add edge attributes to the geopandas dataframe
        for attr in graph.edges[edge]:
            edges_gdf[attr] = pd.Series(
                {edge: graph.edges[edge][attr] for edge in graph.edges()}
            ).to_numpy()

        return edges_gdf

    def create_graph(self, node_data=None, edge_data=None):
        # Read node shapefile
        self.node_gdf = node_data
        self.node_gdf = self.node_gdf[["id", "utilfcltyc", "guid", "geometry"]]

        # Rename columns
        self.node_gdf.columns = ["node_id", "node_type", "guid_pole", "geometry"]

        # Convert node_id to integer
        self.node_gdf["node_id"] = self.node_gdf["node_id"].astype(int)

        # Update node_type values
        self.node_gdf["node_type"] = self.node_gdf["node_type"].map(
            {
                "1": "wood",
                "2": "concrete",
                "3": "tower",
                "10": "substation",
                "20": "substation",
                "40": "substation",
            }
        )

        # Read edge shapefile
        self.edge_gdf = edge_data

        # Convert "source" and "target" columns to integer
        self.edge_gdf["source"] = self.edge_gdf["source"].astype(int)
        self.edge_gdf["target"] = self.edge_gdf["target"].astype(int)

        # Generate networkx graph
        self.G = nx.Graph()
        # self.G = nx.DiGraph()

        # Add nodes and their attributes
        for index, row in self.node_gdf.iterrows():
            self.G.add_node(
                row["node_id"],
                node_id=row["node_id"],
                node_type=row["node_type"],
                guid_pole=row["guid_pole"],
                geometry=row["geometry"],
            )

        # Add edges and their attributes
        for index, row in self.edge_gdf.iterrows():
            edge_attributes = {
                k: v for k, v in row.to_dict().items() if k != "geometry"
            }
            self.G.add_edge(
                row["source"],
                row["target"],
                **edge_attributes,
                geometry=LineString(
                    [
                        self.G.nodes[row["source"]]["geometry"],
                        self.G.nodes[row["target"]]["geometry"],
                    ]
                ),
            )

        # assign 77 protective devices to the network
        # Filter nodes based on the required 'node_type' values
        filtered_nodes = [
            node
            for node, attr in self.G.nodes(data=True)
            if attr["node_type"] in ["concrete", "wood"]
        ]

        # Randomly select 77 nodes from the filtered_nodes list
        selected_nodes = random.sample(filtered_nodes, 77)

        # Set the 'protective_device' attribute to 'False' for all nodes first
        for node in self.G.nodes:
            self.G.nodes[node]["protective_device"] = False

        # Set the 'protective_device' attribute to 'True' for the selected nodes
        for node in selected_nodes:
            self.G.nodes[node]["protective_device"] = True

        # save the graph nodes and edges into geodataframe
        self.node_gdf = self.graph_nodes_to_gdf(graph=self.G)
        self.edge_gdf = self.graph_edges_to_gdf(graph=self.G)

    def get_nearest_node(self, building_point, graph_nodes, graph_kdtree):
        """Returns the nearest node in graph G to the given point."""
        # Use the KD-tree to find the nearest node
        _, node_idx = graph_kdtree.query([building_point.x, building_point.y])
        return graph_nodes[node_idx]

    def repair_time_power(self, road_num_pr, num_crew):
        # repair time model inspired from a matlab code by Yousef Darestani
        if road_num_pr.size == 0:
            return np.array([])
        else:
            pr = road_num_pr.reshape((-1, 1))
            t_repair = np.zeros((int(np.ceil(pr.size / num_crew)) + 1, num_crew))
            j = num_crew

            for i in range(int(np.ceil(pr.size / num_crew))):
                # Define an empty numpy array to store the lengths of the matched roads
                lengths_temp = np.array([])

                start_index = i * num_crew
                end_index = (i + 1) * num_crew
                indices = pr[start_index:end_index]

                # evaluate the repair time
                t_repair[i + 1, :j] = np.sort(
                    (
                        np.maximum(5 + 2.5 * np.random.randn(j), np.zeros(j))
                        + np.maximum(4 + 2 * np.random.randn(j), np.zeros(j))
                    )
                    / 24
                    + t_repair[i, :j],
                    axis=0,
                )

                prr = pr[(i) * j + 1 :]
                j = min(num_crew, prr.size)

            t_repair2 = t_repair[1:, :].T.reshape((-1, 1))
            Output1 = pr.reshape((-1, 1))
            Output2 = t_repair2[: Output1.size]
            ind = np.argsort(Output2, axis=0).ravel()
            Output11 = Output1[ind]
            Output22 = Output2[ind]
            Output222 = np.concatenate(
                [Output22[:1], Output22[1:] - Output22[:-1]], axis=0
            )
            Output = np.concatenate([Output11, Output222, Output22], axis=1)
            return Output22  # original function: Output

    # Function to check if a node is connected to a substation
    def is_connected_to_substation(self, graph, node):
        return any(
            graph.nodes[neighbor]["substation"]
            for neighbor in nx.neighbors(graph, node)
        )

    def initialize_temp_graph(self):
        G_temp = nx.Graph()
        G_temp.add_nodes_from(self.G.nodes(data=True))
        G_temp.add_edges_from(self.G.edges(data=True))

        nx.set_node_attributes(G_temp, False, "power_analyzed")
        nx.set_node_attributes(G_temp, 0, "power_back_time")
        nx.set_node_attributes(G_temp, 0, "failure")
        nx.set_node_attributes(G_temp, 0, "power_out")

        return G_temp

    def update_node_failures(self, G_temp, node_failure_dict, failure_link_dict):
        for node, d in G_temp.nodes(data=True):
            node_id = d["node_id"]
            if node_id in node_failure_dict and node_failure_dict[node_id] == 1:
                G_temp.nodes[node]["failure"] = 1
                G_temp.nodes[node]["power_out"] = 1
                linked_nodes = failure_link_dict[node_id]
                for linked_node_id in linked_nodes:
                    G_temp.nodes[linked_node_id]["power_out"] = 1
            else:
                G_temp.nodes[node]["failure"] = 0

    def prioritize_failed_nodes(self, G_temp, node_importance_list):
        failed_nodes = [
            (n, d) for (n, d) in G_temp.nodes(data=True) if d["failure"] == 1
        ]
        node_importance_array = np.array(
            [
                node[1]["node_id"]
                for node in failed_nodes
                if node[1]["node_id"] in node_importance_list
            ]
        )
        sorted_nodes = sorted(
            failed_nodes,
            key=lambda x: np.where(node_importance_array == x[1]["node_id"])[0][0],
        )
        prioritized_nodes = [(n, d) for (n, d) in sorted_nodes]

        return prioritized_nodes, node_importance_array

    def restore_nodes_and_update_power(
        self,
        G_temp,
        prioritized_nodes,
        node_importance_array,
        time_step,
        failure_link_dict,
    ):
        cumulative_time = self.repair_time_power(node_importance_array, self.num_crew)
        time_day = 0
        visited_nodes = set()
        waiting_list = deque()
        connected_components = {}

        def connected_to_substation(node):
            if node not in G_temp.nodes:
                return False
            if node in connected_components:
                return any(
                    G_temp.nodes[n]["node_type"] == "substation"
                    for n in connected_components[node]
                )
            return False

        def update_node(node):
            G_temp.nodes[node]["power_back_time"] = time_day
            G_temp.nodes[node]["power_analyzed"] = True
            G_temp.nodes[node]["power_out"] = 0

        def process_waiting_nodes():
            nodes_to_remove = set()
            for waiting_node in waiting_list:
                if connected_to_substation(waiting_node):
                    update_node(waiting_node)
                    visited_nodes.add(waiting_node)
                    nodes_to_remove.add(waiting_node)

            for node in nodes_to_remove:
                waiting_list.remove(node)

        while any(d["failure"] == 1 for (n, d) in G_temp.nodes(data=True)):
            restore_nodes = [
                n
                for i, (n, d) in enumerate(prioritized_nodes)
                if d["failure"] == 1 and cumulative_time[i] <= time_day
            ]
            for n in restore_nodes:
                G_temp.nodes[n]["failure"] = 0

            G_temp_clr = nx.subgraph_view(
                G_temp, filter_node=lambda n: G_temp.nodes[n]["failure"] == 0
            )

            for restored_node in restore_nodes:
                if restored_node not in connected_components:
                    connected_component = set(
                        nx.node_connected_component(G_temp_clr, restored_node)
                    )
                    for node in connected_component:
                        connected_components[node] = connected_component

            process_waiting_nodes()

            for restored_node in restore_nodes:
                if restored_node not in visited_nodes:
                    linked_nodes = failure_link_dict[restored_node]
                    connected = connected_to_substation(restored_node)

                    for linked_node in linked_nodes:
                        link_node_connected = connected_to_substation(linked_node)
                        if link_node_connected:
                            update_node(linked_node)
                            visited_nodes.add(linked_node)
                        elif linked_node not in visited_nodes:
                            waiting_list.append(linked_node)

                    if connected:
                        update_node(restored_node)
                        visited_nodes.add(restored_node)

            time_day += time_step

    def update_building_gdf(self, G_temp):
        self.building_gdf["power_back_time"] = None

        for idx, row in self.building_gdf.iterrows():
            nearest_node = row["nearest_node"]
            if nearest_node is not None:
                node_id, node_attrs = nearest_node
                attrs_hashable = frozenset(node_attrs.items())
                power_back_time = G_temp.nodes[node_id]["power_back_time"]
                self.building_gdf.at[idx, "power_back_time"] = power_back_time

        power_result_gdf = self.building_gdf[["guid", "power_back_time", "geometry"]]

        return power_result_gdf

    def get_building_power_back_time(
        self,
        node_failure_dict=None,
        node_importance_list=None,
        failure_link_dict=None,
        time_step=14,
    ):
        # set the initial time
        time_day = 0

        # create a new networkx graph
        G_temp = nx.Graph()
        G_temp.add_nodes_from(self.G.nodes(data=True))
        G_temp.add_edges_from(self.G.edges(data=True))

        # add a new binary attribute to each node of the graph
        nx.set_node_attributes(G_temp, False, "power_analyzed")

        # add a new integer attribute to each node of the graph
        nx.set_node_attributes(G_temp, 0, "power_back_time")

        # add a new binary attribute 'failure' to each node of the graph
        nx.set_node_attributes(G_temp, 0, "failure")

        # Add "power_out" attribute with a default value of 0
        nx.set_node_attributes(G_temp, 0, "power_out")

        # Update the failure attribute of nodes in G_temp
        self.update_node_failures(G_temp, node_failure_dict, failure_link_dict)

        # prioritize the restoration process for failed nodes
        prioritized_nodes, node_importance_array = self.prioritize_failed_nodes(
            G_temp, node_importance_list
        )

        # restore nodes and update power attributes
        self.restore_nodes_and_update_power(
            G_temp,
            prioritized_nodes,
            node_importance_array,
            time_step,
            failure_link_dict,
        )

        # update the main building GeoDataFrame with power_back_time values
        power_result_gdf = self.update_building_gdf(G_temp)

        return power_result_gdf

    def get_failure_dict(self, consider_protective_devices=True):
        G_temp = self.G.copy()

        # Create a directed graph based on source and target attributes
        G_directed = nx.DiGraph()
        G_directed.add_nodes_from(G_temp.nodes(data=True))

        # Create a dictionary to map node_ids to their corresponding nodes
        node_id_to_node = {d["node_id"]: n for n, d in G_temp.nodes(data=True)}

        for edge in G_temp.edges(data=True):
            if "source" in edge[2] and "target" in edge[2]:
                source = node_id_to_node[edge[2]["source"]]
                target = node_id_to_node[edge[2]["target"]]
            else:
                source = edge[0]
                target = edge[1]

            G_directed.add_edge(source, target, **edge[2])

        G_temp = G_directed

        # Precompute all downstream nodes
        all_downstream_nodes = {
            node: set(nx.descendants(G_temp, node)) for node in G_temp.nodes()
        }

        # Function to get upstream nodes considering protective devices
        def get_upstream_nodes(node_key):
            upstream_nodes = set()
            queue = deque([node_key])
            visited = set()

            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                for predecessor in G_temp.predecessors(current):
                    if predecessor not in visited:
                        if not (
                            G_temp.nodes[predecessor].get("protective_device", False)
                            or G_temp.nodes[predecessor].get("node_type")
                            == "substation"
                        ):
                            upstream_nodes.add(predecessor)
                            queue.append(predecessor)
            return upstream_nodes

        # Create failure_dict
        failure_dict = {}

        for node in G_temp.nodes(data=True):
            node_id = node[1]["node_id"]
            node_key = node[0]

            # Get downstream nodes
            downstream_nodes = all_downstream_nodes[node_key]

            if consider_protective_devices:
                # Get upstream nodes
                upstream_nodes = get_upstream_nodes(node_key)

                # Add downstream nodes of each upstream node
                additional_nodes = set()
                for upstream_node in upstream_nodes:
                    additional_nodes.update(all_downstream_nodes[upstream_node])

                linked_nodes = (
                    downstream_nodes | upstream_nodes | additional_nodes | {node_key}
                )
            else:
                linked_nodes = downstream_nodes | {node_key}

            failure_dict[node_id] = linked_nodes

        return failure_dict

    def process_sample(
        self, sample_df, node_importance_list, failure_link_dict, time_step
    ):
        """Process a single sample dataframe.

        Args:
            sample_df (DataFrame): DataFrame with node_id and sample_failure columns.
            node_importance_list (list): List of node IDs in order of importance.
            failure_link_dict (dict): Dictionary mapping node failures to linked nodes.
            time_step (int): Time step for recovery analysis.

        Returns:
            Series: Power back time series for this sample.
        """
        sample_dict = sample_df.set_index("node_id").to_dict()["sample_failure"]
        power_final_result_gdf = self.get_building_power_back_time(
            node_failure_dict=sample_dict,
            node_importance_list=node_importance_list,
            failure_link_dict=failure_link_dict,
            time_step=time_step,
        )
        return power_final_result_gdf["power_back_time"].astype(float)

    def run_monte_carlo_analysis(
        self,
        sample_dfs,
        node_importance_list,
        failure_link_dict,
        time_step,
        building_gdf,
        num_workers,
    ):
        """Run Monte Carlo analysis using concurrent.futures for parallelization.

        Args:
            sample_dfs (list): List of sample dataframes to process.
            node_importance_list (list): List of node IDs in order of importance.
            failure_link_dict (dict): Dictionary mapping node failures to linked nodes.
            time_step (int): Time step for recovery analysis.
            building_gdf (GeoDataFrame): GeoDataFrame with building data.
            num_workers (int): Number of worker processes to use.

        Returns:
            tuple: A tuple containing (power_final_result_gdf, power_final_result_samples).
        """
        # Run the analysis in parallel using concurrent.futures
        power_back_time_results = []

        # Create a list of arguments for each sample
        sample_args = [
            (sample_df, node_importance_list, failure_link_dict, time_step)
            for sample_df in sample_dfs
        ]

        # Process samples in parallel
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers
        ) as executor:
            # Map each sample to the process_sample function with fixed arguments
            futures = [
                executor.submit(self.process_sample, *args) for args in sample_args
            ]
            # Collect results
            power_back_time_results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # Compute the average power back time
        average_power_back_time = np.mean(power_back_time_results, axis=0)

        # Create the final GeoDataFrame with average power back time
        power_final_result_gdf = building_gdf.copy()
        power_final_result_gdf["average_power_back_time"] = average_power_back_time

        # Create a DataFrame with all samples
        power_final_result_samples = building_gdf[["guid", "geometry"]].copy()
        for i, sample in enumerate(power_back_time_results):
            power_final_result_samples[f"sample_{i + 1}"] = sample

        return power_final_result_gdf, power_final_result_samples

    def recovery_analysis(
        self,
        building_shapefile=None,
        pole_damage_mcs_samples_df=None,
        num_thread=1,
        time_step=7,
        consider_protective_devices=True,
    ):
        # Read shapefile into a GeoDataFrame
        self.building_gdf = building_shapefile
        self.building_gdf = self.building_gdf.set_geometry("geometry").to_crs(self.crs)

        # Create a KD-tree from the nodes in the graph
        graph_nodes = list(self.G.nodes(data=True))
        graph_kdtree = cKDTree([data["geometry"].coords[0] for _, data in graph_nodes])

        # Add a new column to building_gdf to store the nearest node for each point
        self.building_gdf["nearest_node"] = self.building_gdf["geometry"].apply(
            lambda p: self.get_nearest_node(p, graph_nodes, graph_kdtree)
        )

        # Merge the dataframes
        power_failure_df = self.node_gdf.merge(
            pole_damage_mcs_samples_df, left_on="guid_pole", right_on="guid", how="left"
        )

        # Define a translation table to swap '0' and '1'
        translation_table = str.maketrans("01", "10")

        # Swap '0' and '1' in the 'failure' column
        power_failure_df["failure"] = power_failure_df["failure"].str.translate(
            translation_table
        )

        # Split the failure string into columns
        failure_columns = power_failure_df["failure"].str.split(",", expand=True)
        failure_columns.columns = [
            str(i + 1) for i in range(len(failure_columns.columns))
        ]  # Column names start from 0

        # Concatenate the node_id and failure_columns dataframes
        power_failure_df = pd.concat(
            [power_failure_df["node_id"], failure_columns], axis=1
        )

        # Convert all columns to int64
        power_failure_df = power_failure_df.astype("int64")

        # Initialize the list to store sample dataframes
        sample_dfs = []

        # get the failure dictionary based on power network rules considering protective devices
        failure_link_dict = self.get_failure_dict(
            consider_protective_devices=consider_protective_devices
        )

        # Loop through power_failure_df and save the resulting dataframes (loop through samples)
        for i in range(self.sample_num):
            sample_df = power_failure_df[["node_id", str(i + 1)]].rename(
                columns={str(i + 1): "sample_failure"}
            )
            sample_dfs.append(sample_df)

        # Calculate the PageRank for all nodes in the graph
        node_pagerank = nx.pagerank(self.G)

        # Sort the nodes in descending order of their PageRank values
        sorted_nodes = sorted(node_pagerank.items(), key=lambda x: x[1], reverse=True)

        # Extract the node_ids in order of their importance
        node_id_importance_order = [node[0] for node in sorted_nodes]

        # Convert node_id_importance_order to actual node_id using attribute
        node_id_importance_order = [
            self.G.nodes[node]["node_id"] for node in node_id_importance_order
        ]

        # Run the Monte Carlo analysis
        (
            power_final_result_gdf,
            power_final_result_samples,
        ) = self.run_monte_carlo_analysis(
            sample_dfs=sample_dfs,
            node_importance_list=node_id_importance_order,
            failure_link_dict=failure_link_dict,
            time_step=time_step,
            building_gdf=self.building_gdf,
            num_workers=num_thread,
        )

        power_final_results = gpd.GeoDataFrame(
            power_final_result_gdf.loc[
                :, ["guid", "geometry", "average_power_back_time"]
            ]
        )

        return power_final_results, power_final_result_samples
