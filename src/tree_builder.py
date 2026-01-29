import copy
import logging
import time
import tiktoken
import numpy as np

from abc import abstractclassmethod
from itertools import chain
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Set, Tuple
from tqdm import tqdm

from .utils import Node, Tree

logging.basicConfig(format="%(asctime)s - %(message)s", 
                    level=logging.INFO,
                    filename="./log/stdout.log",
                    filemode="a"
                    )


class TreeBuilder:

    def __init__(self, conf) -> None:

        self.conf = conf
        if self.conf["tokenizer"] is not None and isinstance(self.conf["tokenizer"], str):
            self.conf["tokenizer"] = tiktoken.get_encoding(self.conf["tokenizer"])       

    def create_node(
        self, index: int, 
        document_index: int = -1, 
        chunk_index: int = -1,
        text: str = None, 
        children_indices: Optional[Set[int]] = None
    ) -> Tuple[int, Node]:
        
        if children_indices is None:
            children_indices = set()
        
        if text is not None:
            embeddings = self.conf["embed_model"].embed(text)
        else:
            embeddings = None
        return (index, Node(text, index, document_index, chunk_index, children_indices, embeddings))
    
    def create_node_batch(
        self, 
        document_index: int = -1, 
        text: List[str] = None, 
        children_indices: Optional[Set[int]] = None, 
        start_idx: int = 0,
        embedding: np.ndarray = None
    ) -> Tuple[Dict[int, Node], int]:

        if children_indices is None:
            children_indices = set()
        
        nodelist = {}
        for idx, passage in enumerate(text, start_idx):
            if embedding is not None:
                embeddings = embedding[idx]
            else:
                embeddings = self.conf["embed_model"].embed(text)
            nodelist.update({idx: Node(passage, idx, document_index, idx - start_idx, copy.deepcopy(children_indices), embeddings)})

        return nodelist, max(nodelist.keys())

    def embed(self, text) -> List[float]:
        return self.conf["embed_model"].embed(text) 

    def abstract(self, text, max_abs_length=150, leaf=True) -> str:
        return self.conf["abs_model"].abstract(text, 
                                               keyword=self.conf["abstract_type"] == "keyword", 
                                               max_abs_length=max_abs_length, 
                                               leaf=leaf)

    def multithreaded_create_leaf_nodes(self, chunks: List[str], 
                                        document_index: int = -1, 
                                        start_idx: int = 0) -> Tuple[Dict[int, Node], int]:

        with ThreadPoolExecutor() as executor:
            future_nodes = {
                executor.submit(self.create_node, index, document_index, index - start_idx, text): (index, text)
                for index, text in enumerate(chunks, start_idx)
            }

            leaf_nodes = {}
            for future in as_completed(future_nodes):
                index, node = future.result()
                leaf_nodes[index] = node

        return leaf_nodes, max(leaf_nodes.keys())

    def build_index(self, docs: List[str] | List[List[str]], use_multithreading: bool = True) -> Tuple[Tree, float]:
        """Builds a tree for an entire corpus, optionally using multithreading."""

        logging.info("Creating Leaf Nodes")
        
        passage_to_node_indices = {i:[] for i in range(len(docs))}

        batch_embedding_models = ("nvidia/NV-Embed-v2",
                                  "Qwen/Qwen3-Embedding-8B",)
                                #   "qwen3-embedding",) # TODO: turn this off until ollama supports batch embedding

        if self.conf["embed_model"].model_name in batch_embedding_models:
            leaf_nodes = {}
            if isinstance(docs[0], List):
                print(f"Creating batched leaf nodes with {self.conf["embed_model"].model_name}...")
                embeddings = self.conf["embed_model"].embed(list(chain(*docs)))
                max_idx = -1
                for document_index, chunk in enumerate(docs):
                    leaf_batch, max_idx = self.create_node_batch(document_index, chunk, start_idx=max_idx + 1, embedding=embeddings)
                    passage_to_node_indices[document_index].extend([node.index for node in leaf_batch.values()])
                    leaf_nodes.update(leaf_batch)
            else:
                print(f"Creating leaf nodes with {self.conf["embed_model"].model_name}...")
                embeddings = self.conf["embed_model"].embed(docs)
                for index, text in enumerate(docs):
                    _, node = self.create_node(index, -1, index)
                    node.text = text
                    node.embeddings = embeddings[index]
                    leaf_nodes[index] = node
                
        else:
            if use_multithreading:
                if isinstance(docs[0], List):
                    leaf_nodes = {}
                    max_idx = -1
                    bar = tqdm(docs, desc="creating leaf nodes")
                    for document_index, chunk in enumerate(bar):
                        leaf_batch, max_idx = self.multithreaded_create_leaf_nodes(chunk, document_index, max_idx + 1)
                        passage_to_node_indices[document_index].extend([node.index for node in leaf_batch.values()])
                        leaf_nodes.update(leaf_batch)
                    bar.close()
                else:
                    leaf_nodes, _ = self.multithreaded_create_leaf_nodes(docs, document_index=-1)
            else:
                if isinstance(docs[0], List):
                    leaf_nodes = {}
                    max_idx = -1
                    bar = tqdm(docs, desc="creating leaf nodes")
                    for document_index, chunk in enumerate(bar):
                        leaf_batch, max_idx = self.create_node_batch(document_index, chunk, start_idx=max_idx + 1)
                        passage_to_node_indices[document_index].extend([node.index for node in leaf_batch.values()])
                        leaf_nodes.update(leaf_batch)
                    bar.close()
                else:
                    leaf_nodes = {}
                    for index, text in enumerate(docs):
                        _, node = self.create_node(index, -1, index, text)
                        leaf_nodes[index] = node

        layer_to_node_indices = {0: sorted(list(node.index for node in leaf_nodes.values()))}

        logging.info(f"Created {len(leaf_nodes)} Leaf Embeddings")

        logging.info("Building All Nodes")

        all_nodes = copy.deepcopy(leaf_nodes)

        start_time = time.time()
        if self.conf["reorganize_leaf"] or isinstance(docs[0], str): 
            root_nodes = self.__construct_tree(all_nodes, layer_to_node_indices, 
                                               use_multithreading=use_multithreading)
        else: 
            # use preset chunks if there is an explicit split instead of reorganizing leaves
            root_nodes = self.__construct_passage_tree(all_nodes, layer_to_node_indices, passage_to_node_indices, 
                                                     use_multithreading=use_multithreading)
        tree = Tree(all_nodes, root_nodes, leaf_nodes, layer_to_node_indices)
        end_time = time.time()

        return tree, end_time - start_time
    
    def build_index_list(self, docs: List[str] | List[List[str]], use_multithreading: bool = True) -> Tuple[List[Tree], float]:
        """Builds a tree for each input document, optionally using multithreading."""
        
        logging.info("Creating Leaf Nodes")

        batch_embedding_models = ("nvidia/NV-Embed-v2",
                                  "Qwen/Qwen3-Embedding-8B",)
                                #   "qwen3-embedding",) # TODO: turn this off until ollama supports batch embedding

        if self.conf["embed_model"].model_name in batch_embedding_models:
            leaf_nodes_list = []
            if isinstance(docs[0], List): 
                # one tree per passage
                print(f"Creating batched leaf nodes with {self.conf["embed_model"].model_name}...")
                for dataset_index, dataset_docs in enumerate(docs):
                    leaf_nodes_list.append({})
                    embeddings = self.conf["embed_model"].embed(dataset_docs)
                    for index, text in enumerate(dataset_docs):
                        _, node = self.create_node(index, -1, index)
                        node.text = text
                        node.embeddings = embeddings[index]
                        leaf_nodes_list[dataset_index][index] = node
            else: 
                # one passage as a dataset (for demo)
                print(f"Creating leaf nodes with {self.conf["embed_model"].model_name}...")
                leaf_nodes_list.append({})
                embeddings = self.conf["embed_model"].embed(docs)
                for index, text in enumerate(docs):
                    _, node = self.create_node(index, -1, index)
                    node.text = text
                    node.embeddings = embeddings[index]
                    leaf_nodes_list[0][index] = node
                
        else:
            if use_multithreading:
                if isinstance(docs[0], List): 
                    # one tree per passage
                    leaf_nodes_list = []
                    bar = tqdm(docs, desc="creating leaf nodes")
                    for dataset_index, dataset_docs in enumerate(bar):
                        leaf_nodes_list.append({})
                        leaf_batch, _ = self.multithreaded_create_leaf_nodes(dataset_docs, dataset_index, 0)
                        leaf_nodes_list[dataset_index] = leaf_batch
                    bar.close()
                else:
                    # one passage as a dataset (for demo)
                    leaf_nodes, _ = self.multithreaded_create_leaf_nodes(docs, document_index=-1)
                    leaf_nodes_list = [leaf_nodes]
            else:
                if isinstance(docs[0], List): 
                    # one tree per passage
                    leaf_nodes_list = []
                    bar = tqdm(docs, desc="creating leaf nodes")
                    for dataset_index, dataset_docs in enumerate(bar):
                        leaf_batch, _ = self.create_node_batch(dataset_index, dataset_docs, 0)
                        leaf_nodes_list[dataset_index] = leaf_batch
                    bar.close()
                else: 
                    # one passage as a dataset (for demo)
                    leaf_nodes_list = [{}]
                    for index, text in enumerate(docs):
                        _, node = self.create_node(index, -1, index, text)
                        leaf_nodes_list[0][index] = node

        logging.info(f"Created Leaf Embeddings")

        logging.info("Building All Nodes")
        start_time = time.time()

        tree_list = []
        for leaf_nodes in leaf_nodes_list:
            layer_to_node_indices = {0: sorted(list(node.index for node in leaf_nodes.values()))}
            all_nodes = copy.deepcopy(leaf_nodes)

            try:
                root_nodes = self.__construct_tree(all_nodes, layer_to_node_indices, use_multithreading=use_multithreading)
                tree_list.append(Tree(all_nodes, root_nodes, leaf_nodes, layer_to_node_indices))
            except Exception as e:
                print(e)
                print("⚠An exception occured! Aborting tree construction and adding an empty root ...")
                index = max(set(leaf_nodes.keys())) + 1
                _, root = self.create_node(index, text="[EMPTY]", children_indices=set(leaf_nodes.keys()))
                all_nodes = copy.deepcopy(leaf_nodes)
                all_nodes[index] = root
                layer_to_node_indices = {0: layer_to_node_indices[0],
                                         1: [index]}
                tree_list.append(Tree(all_nodes, {index: root}, leaf_nodes, layer_to_node_indices))

        end_time = time.time()

        return tree_list, end_time - start_time

    @abstractclassmethod
    def __construct_passage_tree(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_node_indices: Dict[int, List[int]],
        passage_to_node_indices: Dict[int, List[int]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        pass

    @abstractclassmethod
    def __construct_tree(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_node_indices: Dict[int, List[int]],
        use_multithreading: bool = True,
    ) -> Dict[int, Node]:
        pass

