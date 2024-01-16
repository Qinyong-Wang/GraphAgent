
class Vector_db:
    def __init__(self, db_name):
        self.db_name = db_name
        pass

    def insert(self, vector):
        pass

    def retrieve(self, vector):
        pass

    def update(self, vector):
        pass

    def delete(self, vector):
        pass

class Embedding_model:
    def __init__(self, model_name):
        self.model_name = model_name
        pass

    def embed(self, text):
        pass

class GraphMemory:

    def __init__(self, vector_db = "pinecone", embedding_model = "bert"):
        self.embedding_model = None
        pass

    def node_to_json(self, node):
        pass
    
    def edge_to_json(self, edge):
        pass

    def memorize_nodes(self, node_list):
        pass

    def memorize_edges(self, edge_list):
        pass

    def retrieve_nodes(self, node_list):
        pass

    def retrieve_edges(self, edge_list):
        pass