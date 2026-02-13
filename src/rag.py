import logging
import pickle
import json

from typing import List, Dict

from .abstract_tree_builder import AbstractTreeBuilder
from .tree_retriever import TreeRetriever
from .utils import Tree

logging.basicConfig(format="%(asctime)s - %(message)s", 
                    level=logging.INFO,
                    filename="./log/stdout.log",
                    filemode="a"
                    )


class RAG:

    def __init__(self, conf: Dict = None, tree: str | Tree = None) -> None:

        self.conf: Dict = conf
        self.tree = tree

        self.tree_builder = AbstractTreeBuilder(conf=self.conf)
        self.tb_time: float = -1.
        self.tr_time: float = -1.
        self.qa_time: float = -1.

        if self.tree is not None:
            self._init_retrievers()
        else:
            self.retriever = None

        self.retrieve_count = 0
        self.time_dict = {'tree': 0.0, 'sparse': 0.0, 'rerank': 0.0}

    def _init_retrievers(self):
        if isinstance(self.tree, List):
            self.retriever = [TreeRetriever(self.conf.copy(), tree) for tree in self.tree]
        else:
            self.retriever = TreeRetriever(self.conf.copy(), self.tree)

    def add_documents(self, data):
        if self.conf["passage_as_tree"]:
            self.tree, self.tb_time = self.tree_builder.build_index_list(docs=data.all_passages)
        else:
            self.tree, self.tb_time = self.tree_builder.build_index(docs=data.all_passages)
        self._init_retrievers()

    def build_vocab(self, data):
        if isinstance(self.retriever, List):
            for i, retriever in enumerate(self.retriever):
                retriever.hybrid_index(data.all_passages[i] if isinstance(data.all_passages[0], List) else data.all_passages)
        else:
            self.retriever.hybrid_index(data.get_documents())

    def retrieve(self, question, query_id, **kwargs):
        if self.retriever is None:
            raise ValueError(
                "Retriever has not been initialized. Call 'add_documents' first."
            )
        
        self.retrieve_count += 1
        if isinstance(self.retriever, List):
            retrieved_documents, layer_information, self.tr_time, time_dict = self.retriever[query_id].retrieve(question, **kwargs)
        else:
            retrieved_documents, layer_information, self.tr_time, time_dict = self.retriever.retrieve(question, **kwargs)

        for k,v in time_dict.items():
            self.time_dict[k] += v

        return retrieved_documents, layer_information

    def qa(self, question, **kwargs):
        return self.conf["qa_model"].qa(question, **kwargs)

    def save(self, name, path, is_json=False):
        if not hasattr(self, name) or getattr(self, name) is None:
            raise ValueError("There is nothing to save.")
        if is_json:
            with open(path, "w") as file:
                json.dump(getattr(self, name), file)
        else:
            with open(path, "wb") as file:
                pickle.dump(getattr(self, name), file)
        logging.info(f"{name} successfully saved to {path}")

    def load(self, name, path, is_json=False):
        if is_json:
            with open(path, "r") as file:
                setattr(self, name, json.load(file))
        else:
            with open(path, 'rb') as file:
                setattr(self, name, pickle.load(file))
        if name == "tree":
            self._init_retrievers()
        logging.info(f"{name} successfully loaded to {path}")

    def find_ancestors(self, node_index):
        if self.tree is None:
            raise ValueError("There is no tree to search.")
        
        if isinstance(node_index, tuple): # document_index:chunk_index
            for node in self.tree.all_nodes.values():
                if node.document_index == node_index[0] and node.chunk_index == node_index[1]:
                    node_index = node.index
                    break
        
        ancestors = {}
        current_layer_node_index = node_index
        for l, nodes in self.tree.layer_to_node_indices.items():
            for node in nodes:
                if current_layer_node_index in self.tree.all_nodes[node].children:
                    ancestors[l] = self.tree.all_nodes[node]
                    current_layer_node_index = node
                    break
            
        return ancestors
