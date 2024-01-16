
import pinecone
import openai


class Pinecone_db:
    def __init__(self, api_key, environment, index_name):
        self.api_key = api_key
        self.index_name = index_name
        self.environment = environment
        self.pinecone_client = pinecone.Client(api_key=self.api_key, environment=self.environment)
        self.index = self.pinecone_client.index(index_name=self.index_name)

    def insert(self, vectors, namespace):
        self.index.upsert(vectors=vectors, namespace=namespace)

    def retrieve(self, query_vector, namespace, top_k=10):
        results = self.index.query(vector=query_vector, namespace=namespace, top_k=top_k)
        return results

    def update(self, vectors, ids, namespace):
        self.index.upsert(vectors=vectors, ids=ids, namespace=namespace)

    def delete(self, ids, namespace):
        self.index.delete(ids=ids, namespace=namespace)

i

class Embedding_model:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        openai.api_key = api_key

    def embed(self, texts, max_tokens=2048):

        return embeddings


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