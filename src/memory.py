"""graph_memory.py"""

import json
import pinecone
import openai


class PineconeDB:
    """Pinecone database class."""
    def __init__(self, api_key, environment, index_name):
        self.api_key = api_key
        self.index_name = index_name
        self.environment = environment
        self.pinecone_client = pinecone.Client(api_key=self.api_key, environment=self.environment)
        self.index = self.pinecone_client.index(index_name=self.index_name)

    def insert(self, vectors, namespace):
        """Inserts vectors into the Pinecone database."""
        self.index.upsert(vectors=vectors, namespace=namespace)

    def retrieve(self, query_vector, top_k=10, namespace = None):
        """Retrieves vectors from the Pinecone database."""
        if namespace is not None:
            results = self.index.query(vector=query_vector, namespace=namespace, top_k=top_k)
        else:
            results = self.index.query(vector=query_vector, top_k=top_k)
        return results

    def update(self, vectors, namespace):
        """Updates vectors in the Pinecone database."""
        self.index.upsert(vectors=vectors, namespace=namespace)

    def delete(self, ids, namespace = None):
        """Deletes vectors from the Pinecone database."""
        if namespace is not None:
            self.index.delete(ids=ids, namespace=namespace)
        else:
            self.index.delete(ids=ids)

class EmbeddingModel:
    """Embedding model class."""
    def __init__(self, model_name, api_key):
        if model_name not in ["text-embedding-ada-002"]:
            raise ValueError("Model name is not supported.")
        self.model_name = model_name
        openai.api_key = api_key

    def embed(self, texts, max_tokens=4096):
        """Embeds texts using the OpenAI API."""
        embeddings = openai.Embedding.create(input = texts,
                                             model=self.model_name,
                                             max_tokens=max_tokens)
        return embeddings


class GraphMemory:
    """Graph memory class."""
    def __init__(self,
                 graph,
                 vector_db = "pinecone",
                 embedding_model = "text-embedding-ada-002",
                 neighbor_sampling = "none",
                 embedding_model_api_key = None,
                 vector_db_api_key = None,
                 vector_db_environment = None,
                 index_name = None):

        self.embedding_model = EmbeddingModel(model_name=embedding_model, 
                                              api_key=embedding_model_api_key)
        if vector_db == "pinecone":
            self.vector_db = PineconeDB(api_key=vector_db_api_key, 
                                        environment=vector_db_environment, 
                                        index_name=index_name)
        else:
            raise ValueError("Vector database is not supported.")
        self.graph = graph
        self.neighbor_sampling = neighbor_sampling

    def aggregate_node_info(self, node):
        """Aggregates node information."""
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
        """Aggregates edge information."""
        edge_dict = {"source_node_id": self.aggregate_node_info(self.graph.node_dict[edge[0]]),
                     "target_node_id": self.aggregate_node_info(self.graph.node_dict[edge[1]]),
                     "edge_type": edge["edge_type"],
                     "edge_weight": edge["edge_weight"]}

        return edge_dict

    def memorize_nodes(self, node_list, node_type):
        """Memorizes nodes."""
        node_list_text = []
        node_ids = []
        for node in node_list:
            if node["node_type"] != node_type:
                raise ValueError("Node type does not match.")
            node_info = self.aggregate_node_info(node)
            node_list_text.append(json.dumps(node_info))
            node_ids.append(node["node_id"])
        node_embeddings = self.embedding_model.embed(node_list_text)
        vectors = []
        for i in range(len(node_embeddings["data"])):
            vectors.append({"id": node_ids[i], "vector": node_embeddings["data"][i]})
        self.vector_db.insert(vectors=vectors, namespace=node_type)

    def memorize_edges(self, edge_list, edge_type):
        """Memorizes edges."""
        edge_list_text = []
        edge_ids = []
        for edge in edge_list:
            if edge["edge_type"] != edge_type:
                raise ValueError("Edge type does not match.")   
            edge_info = self.aggregate_edge_info(edge)
            edge_list_text.append(json.dumps(edge_info))
            edge_ids.append(edge["source_node_id"] + "-" + edge["target_node_id"])
        edge_embeddings = self.embedding_model.embed(edge_list_text)
        vectors = []
        for i in range(len(edge_embeddings["data"])):
            vectors.append({"id": edge_ids[i], "vector": edge_embeddings["data"][i]})
        self.vector_db.insert(vectors=vectors, namespace=edge_type)

    def retrieve_nodes(self, node, top_k = 5, same_node_type = False):
        """Retrieves nodes."""  
        node_type = node["node_type"]
        node_info = self.aggregate_node_info(node)
        node_info_text = json.dumps(node_info)
        node_embedding = self.embedding_model.embed(node_info_text)
        if same_node_type:
            results = self.vector_db.retrieve(query_vector=node_embedding["data"][0],
                                              namespace=node_type,
                                              top_k=top_k)
        else:
            results = self.vector_db.retrieve(query_vector=node_embedding["data"][0], top_k=top_k)

        retrieved_node_ids = []
        for result in results:
            retrieved_node_ids.append(self.graph.node_dict[result["id"]])
        retrieved_node_infos = []
        for retrieved_node_id in retrieved_node_ids:
            retrieved_node_infos.append(self.aggregate_node_info(retrieved_node_id))

        return retrieved_node_infos

    def retrieve_edges(self, edge, top_k = 5, same_edge_type = False):
        """Retrieves edges."""
        edge_type = edge["edge_type"]
        edge_info = self.aggregate_edge_info(edge)
        edge_info_text = json.dumps(edge_info)
        edge_embedding = self.embedding_model.embed(edge_info_text)
        if same_edge_type:
            results = self.vector_db.retrieve(query_vector=edge_embedding["data"][0],
                                              namespace=edge_type,
                                              top_k=top_k)
        else:
            results = self.vector_db.retrieve(query_vector=edge_embedding["data"][0], top_k=top_k)

        retrieved_edge_ids = []
        for result in results:
            retrieved_edge_ids.append(self.graph.edge_dict[result["id"]])
        retrieved_edge_infos = []
        for retrieved_edge_id in retrieved_edge_ids:
            edge_id = retrieved_edge_id.split("-")
            retrieved_edge_infos.append(self.aggregate_edge_info(edge_id))

        return  retrieved_edge_infos

    def memorize_all_nodes(self, batch_size=32):
        """Memorizes all nodes."""
        for node_type in self.graph.node_type_list:
            for node_id in self.graph.nodes_clustered_by_type[node_type]:
                node_list = []
                node_list.append(self.graph.node_dict[node_id])
                if (len(node_list) == batch_size or
                    node_id == self.graph.nodes_clustered_by_type[node_type][-1]):
                    self.memorize_nodes(node_list=node_list, node_type=node_type)
                    node_list = []

    def memorize_all_edges(self, batch_size=32):
        """Memorizes all edges."""
        for edge_type in self.graph.edge_type_list:
            for edge_id in self.graph.edges_clustered_by_type[edge_type]:
                edge_info_list = []
                edge_info_list.append(self.aggregate_edge_info(self.graph.edge_dict[edge_id]))
                if (len(edge_list) == batch_size or
                    edge_id == self.graph.edges_clustered_by_type[edge_type][-1]):
                    self.memorize_edges(edge_list==edge_list, edge_type=edge_type)
                    edge_list = []

    def forget_nodes(self, node_ids):
        """Forgets nodes."""
        self.vector_db.delete(ids=node_ids)

    def forget_edges(self, edge_ids):
        """Forgets edges."""
        self.vector_db.delete(ids=edge_ids)