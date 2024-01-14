import unittest
from ..src.graph import Graph
"""         
            node4(B) ++ node5(A)
            +              +
            +              +       
node3(B) ++ node1(A) ++ node2(B)
            +
            +
            node6(B)    

"""


class TestGraph(unittest.TestCase):
    def setUp(self):
        self.node_edge_list = {
            "node_list": [
                {"node_id": "node1", 
                 "node_name": "node_name1", 
                 "node_type": "NodeTypeA", 
                 "node_attributes": {"attribute1": "value1", "attribute2": "value2"}},
                {"node_id": "node2", 
                 "node_name": "node_name2", 
                 "node_type": "NodeTypeB", 
                 "node_attributes": {"attribute1": "value3", "attribute2": "value4"}},
                 {"node_id": "node3", 
                 "node_name": "node_name3", 
                 "node_type": "NodeTypeB", 
                 "node_attributes": {"attribute1": "value1", "attribute2": "value2"}},
                 {"node_id": "node4", 
                 "node_name": "node_name4", 
                 "node_type": "NodeTypeB", 
                 "node_attributes": {"attribute1": "value3", "attribute2": "value4"}},
                 {"node_id": "node5", 
                 "node_name": "node_name5", 
                 "node_type": "NodeTypeA", 
                 "node_attributes": {"attribute1": "value1", "attribute2": "value2"}},
                 {"node_id": "node6", 
                 "node_name": "node_name6",
                 "node_type": "NodeTypeB", 
                 "node_attributes": {"attribute1": "value1", "attribute2": "value2"}}
            ],
            "edge_list": [
                {"source_node_id": "node1", "target_node_id": "node2", "edge_type": "EdgeTypeA", "edge_weight": 1},
                {"source_node_id": "node1", "target_node_id": "node3", "edge_type": "EdgeTypeA", "edge_weight": 1},
                {"source_node_id": "node1", "target_node_id": "node4", "edge_type": "EdgeTypeA", "edge_weight": 1},
                {"source_node_id": "node4", "target_node_id": "node5", "edge_type": "EdgeTypeA", "edge_weight": 1},
                {"source_node_id": "node2", "target_node_id": "node5", "edge_type": "EdgeTypeA", "edge_weight": 1},
                {"source_node_id": "node1", "target_node_id": "node6", "edge_type": "EdgeTypeA", "edge_weight": 1}
            ]
        }
        self.test_graph = Graph(node_edge_list=self.node_edge_list, directed=False)

    def test_initialization(self):
        self.assertEqual(len(self.test_graph.node_dict), 6)
        self.assertEqual(len(self.test_graph.edge_dict), 6)

    def test_average_degree_calculation(self):
        avg_in_deg, avg_out_deg = self.test_graph._average_degree_per_node_type()
        self.assertEqual(avg_in_deg, {"NodeTypeA": 3, "NodeTypeB": 1.5})
        self.assertEqual(avg_out_deg, {"NodeTypeA": 3, "NodeTypeB": 1.5})

    def test_get_n_hop_neighbor(self):
        neighbors = self.test_graph.get_n_hop_neighbor("node1", hop_num=1)
        self.assertEqual(len(neighbors), 4)

        neighbors = self.test_graph.get_n_hop_neighbor("node1", hop_num=2)
        self.assertEqual(len(neighbors), 5)
       
    def test_normalized_node_degree_neighbor_sampling(self):
        sampled_neighbors = self.test_graph.normalized_node_degree_neighbor_sampling("node1", hop=1, k=2)
        self.assertEqual(len(sampled_neighbors), 2)
        node_ids = []
        for node in sampled_neighbors:
            node_ids.append(node["node_id"])
        self.assertEqual(sampled_neighbors[0]["node_id"] in ["node2", "node4"], True)
        self.assertEqual(sampled_neighbors[0]["node_id"] in ["node2", "node4"], True)

    def test_random_walk_neighbor_sampling(self):
        sampled_neighbors = self.test_graph.random_walk_neighbor_sampling("node1", restart_prob=0.2, steps=10, walks=5, max_neighbor_num=4)
        self.assertTrue(len(sampled_neighbors) == 4)

    def test_error_handling(self):
        with self.assertRaises(ValueError):
            wrong_node_edge_list = {"node_list": [], "edge_list": []}
            Graph(node_edge_list=wrong_node_edge_list)

        with self.assertRaises(ValueError):
            self.test_graph.get_n_hop_neighbor("node8", hop_num=1)

        with self.assertRaises(ValueError):
            self.test_graph.random_walk_neighbor_sampling("node9")


if __name__ == '__main__':
    unittest.main()



