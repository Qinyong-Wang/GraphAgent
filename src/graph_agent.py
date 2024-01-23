from .memory import GraphMemory

class GraphAgent:

    def __init__(self,
                 graph,
                 agent_config):
        
        self.agent_config = agent_config
        self.graph_memory = GraphMemory(graph)

    def predict_edge(self, edge_info):
        pass

    def node_classification(self, node_info):
        pass