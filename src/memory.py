import json
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

    def update(self, vectors, namespace):
        self.index.upsert(vectors=vectors, namespace=namespace)

    def delete(self, ids, namespace):
        self.index.delete(ids=ids, namespace=namespace)

i

class Embedding_model:
    def __init__(self, model_name, api_key):
        self.model_name = model_name
        openai.api_key = api_key

    def embed(self, texts, max_tokens=4096):
        embeddings = openai.Embedding.create(input = texts, model=self.model_name, max_tokens=max_tokens)
        return embeddings


class GraphMemory:

    def __init__(self, graph, vector_db = "pinecone", embedding_model = "text", 
                 api_key = None, environment = None, index_name = None):
        self.embedding_model = embedding_model
        if vector_db == "pinecone":
            self.vector_db = Pinecone_db(api_key=api_key, environment=environment, index_name=index_name)
        else:
            raise ValueError("Vector database is not supported.")
        self.graph = graph
        self.neighbor_sampling = "none"

    def aggregate_node_info(self, node):
        node_neighbors = []
        if self.neighbor_sampling == "none":
            for neighbor in self.graph.adjacency_list[node["node_id"]]:
                node_neighbors.append(self.graph.node_dict[neighbor])
        if self.neighbor_sampling == "normalized_node_degree":
                sampled_neighbors = self.graph.normalized_node_degree_neighbor_sampling(node["node_id"])
                for neighbor in sampled_neighbors:
                    node_neighbors.append(self.graph.node_dict[neighbor])
        if self.neighbor_sampling == "random_walk":
                sampled_neighbors = self.graph.random_walk_neighbor_sampling(node["node_id"])
                for neighbor in sampled_neighbors:
                    node_neighbors.append(self.graph.node_dict[neighbor])

        node_dict = {"node_id": node["node_id"], 
                     "node_name": node["node_name"], 
                     "node_type": node["node_type"], 
                     "node_attributes": node["node_attributes"],
                     "neighbors": node_neighbors}

        return node_dict
    
    def aggregate_edge_info(self, edge):
        edge_dict = {"source_node_id": self.aggregate_node_info(self.graph.node_dict[edge[0]]),
                     "target_node_id": self.aggregate_node_info(self.graph.node_dict[edge[1]]),
                     "edge_type": edge["edge_type"],
                     "edge_weight": edge["edge_weight"]}

        return edge_dict

    def memorize_nodes(self, node_info_list):
        node_list_text = []
        node_ids = []
        for node_info in node_info_list:
            node_list_text.append(json.dumps(node_info))
            node_ids.append(node_info["node_id"])
        node_embeddings = self.embedding_model.embed(node_list_text)
        vectors = []
        for i in range(len(node_embeddings["data"])):
            vectors.append({"id": node_ids[i], "vector": node_embeddings["data"][i]})
        

    def memorize_edges(self, edge_list):
        pass

    def retrieve_nodes(self, node_list):
        pass

    def retrieve_edges(self, edge_list):
        pass