"""Graph class to store graph data."""

import random
from collections import deque


class Graph:
    """Graph class to store graph data.

    Attributes:
        node_edge_list: A dictionary containing nodes and edges information.
        directed: Boolean indicating if the graph is directed.
    """

    def __init__(self, node_edge_list=None, directed=False):
        """Initializes the Graph object."""
        self.node_edge_list = node_edge_list
        self.directed = directed

        self.node_dict = {}
        self.node_in_degrees = {}
        self.node_out_degrees = {}
        self.node_type_list = []
        self.nodes_clustered_by_type = {}

        self.edge_dict = {}
        self.edge_type_list = []
        self.edges_clustered_by_type = {}

        for edge_type in self.edge_type_list:
            if edge_type in self.node_type_list:
                raise ValueError("Edge type and node type cannot be the same.")
            
        self.adjacency_list = {}
        self.sampled_adjacency_list = {}

        self._verify_graph_format()
        self._node_init()
        self._edge_init()

        self.average_in_degrees, self.average_out_degrees = self._average_degree_per_node_type()
        self.node_labels = []
        self.edge_labels = []

    def _verify_graph_format(self):
        """Verifies the format of the input graph."""
        if "node_list" not in self.node_edge_list or "edge_list" not in self.node_edge_list:
            raise ValueError("Graph format is not correct")

        if not self.node_edge_list["node_list"] or not self.node_edge_list["edge_list"]:
            raise ValueError("Graph is empty.")

        required_node_keys = {"node_id", "node_name", "node_type", "node_attributes"}
        required_edge_keys = {"source_node_id", "target_node_id", "edge_type", "edge_weight"}

        if not required_node_keys.issubset(set(self.node_edge_list["node_list"][0])):
            raise ValueError("Graph node format is not correct")

        if not required_edge_keys.issubset(set(self.node_edge_list["edge_list"][0])):
            raise ValueError("Graph edge format is not correct")
        
        if isinstance(self.node_edge_list["node_list"][0]['node_id'], str) == False:
            raise ValueError("Graph node id must be string")

    def _node_init(self):
        """Initializes nodes in the graph."""
        for node in self.node_edge_list["node_list"]:
            self.adjacency_list[node["node_id"]] = []
            self.node_dict[node["node_id"]] = node
            self.node_in_degrees[node["node_id"]] = 0
            self.node_out_degrees[node["node_id"]] = 0
            if node["node_type"] not in self.node_type_list:
                self.nodes_clustered_by_type[node["node_type"]] = []
                self.node_type_list.append(node["node_type"])
            self.nodes_clustered_by_type[node["node_type"]].append(node["node_id"])

    def _edge_init(self):
        """Initializes edges in the graph."""
        for edge in self.node_edge_list["edge_list"]:
            if edge["edge_type"] not in self.edge_type_list:
                self.edges_clustered_by_type[edge["edge_type"]] = []
                self.edge_type_list.append(edge["edge_type"])
            self.edges_clustered_by_type[edge["edge_type"]].append((edge["source_node_id"], edge["target_node_id"]))

            self.edge_dict[(edge["source_node_id"], edge["target_node_id"])] = edge

            if (edge["source_node_id"] not in self.adjacency_list
                or edge["target_node_id"] not in self.adjacency_list):
                raise ValueError("Edge node is not in the node list")

            if self.directed:
                self.adjacency_list[edge["source_node_id"]].append(edge["target_node_id"])
                self.node_in_degrees[edge["target_node_id"]] += 1
                self.node_out_degrees[edge["source_node_id"]] += 1
            else:
                self.adjacency_list[edge["source_node_id"]].append(edge["target_node_id"])
                self.adjacency_list[edge["target_node_id"]].append(edge["source_node_id"])
                self.node_in_degrees[edge["target_node_id"]] += 1
                self.node_out_degrees[edge["target_node_id"]] += 1
                self.node_in_degrees[edge["source_node_id"]] += 1
                self.node_out_degrees[edge["source_node_id"]] += 1

            

    def _average_degree_per_node_type(self):
        """Calculates average degree per node type."""
        total_in_degrees = {node_type: 0 for node_type in self.node_type_list}
        total_out_degrees = {node_type: 0 for node_type in self.node_type_list}
        count_per_type = {node_type: 0 for node_type in self.node_type_list}

        for node_id, node in self.node_dict.items():
            node_type = node["node_type"]
            total_in_degrees[node_type] += self.node_in_degrees[node_id]
            total_out_degrees[node_type] += self.node_out_degrees[node_id]
            count_per_type[node_type] += 1

        average_in_degrees = {nt: total_in_degrees[nt] / count_per_type[nt]
                              for nt in self.node_type_list}
        average_out_degrees = {nt: total_out_degrees[nt] / count_per_type[nt]
                               for nt in self.node_type_list}

        return average_in_degrees, average_out_degrees

    def get_n_hop_neighbor(self, node_id, hop_num=2):
        """Retrieves n-hop neighbors of a given node."""
        if node_id not in self.adjacency_list:
            raise ValueError(f"Node {node_id} does not exist in the graph.")

        queue = deque([(node_id, 0)])
        visited = set()
        n_hop_neighbors = []

        while queue:
            current_node, hop_count = queue.popleft()
            if hop_count > hop_num:
                continue

            if current_node in visited:
                continue

            visited.add(current_node)

            if hop_count > 0:
                n_hop_neighbors.append((current_node, hop_count))

            for neighbor in self.adjacency_list[current_node]:
                if neighbor not in visited:
                    queue.append((neighbor, hop_count + 1))

        return n_hop_neighbors

    def normalized_node_degree_neighbor_sampling(self, node_id, hop=2, k=8):
        """Samples neighbors based on normalized node degree."""
        if node_id not in self.adjacency_list:
            raise ValueError(f"Node {node_id} does not exist in the graph.")
        
        if (node_id, "navgd",hop, k) in self.sampled_adjacency_list:
            return self.sampled_adjacency_list[(node_id, "navgd", hop, k)]

        n_hop_neighbors = self.get_n_hop_neighbor(node_id, hop_num=hop)
        node_importance = [(neighbor,
                            hop_count,
                            self.node_in_degrees[neighbor] / self.average_in_degrees[self.node_dict[neighbor]["node_type"]])
                            for neighbor, hop_count in n_hop_neighbors]

        node_importance.sort(key=lambda x: x[2], reverse=True)

        sampled_neighbor = [{"node_id": ni[0], "hop_count": ni[1]}
                            for ni in node_importance[:min(k, len(node_importance))]]
        
        self.sampled_adjacency_list[(node_id, "navgd", hop, k)] = sampled_neighbor
        return sampled_neighbor

    def random_walk_neighbor_sampling(self, node_id, restart_prob=0.2,
                                      steps=100, walks=5, max_neighbor_num=10):
        """Performs random walk neighbor sampling."""
        if node_id not in self.adjacency_list:
            raise ValueError(f"Node {node_id} does not exist in the graph.")
        if (node_id, "rw", restart_prob, steps, walks, max_neighbor_num) in self.sampled_adjacency_list:
            return self.sampled_adjacency_list[(node_id, "rw", restart_prob, steps, walks, max_neighbor_num)]

        freq_dict = {}

        for _ in range(walks):
            current_node = node_id
            for _ in range(steps):
                if random.random() < restart_prob:
                    current_node = node_id
                elif self.adjacency_list[current_node]:
                    current_node = random.choice(self.adjacency_list[current_node])

                if current_node != node_id:
                    freq_dict[current_node] = freq_dict.get(current_node, 0) + 1

        sorted_nodes = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        sampled_neighbor = [{"node_id": node, "hop_count": 1}
                            for node, _ in sorted_nodes[:min(max_neighbor_num, len(sorted_nodes))]]

        self.sampled_adjacency_list[(node_id, "rw", restart_prob, steps, walks, max_neighbor_num)] = sampled_neighbor

        return sampled_neighbor
