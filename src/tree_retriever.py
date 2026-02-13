import logging
import os
import shutil
from typing import Dict, Tuple, List, Set
from threading import Lock

import time
import tiktoken
import bm25s
import Stemmer
import numpy as np
from .utils import (Node, Tree, distances_from_embeddings, get_embeddings, get_text_list,
                    reverse_mapping, rrf)

logging.basicConfig(format="%(asctime)s - %(message)s", 
                    level=logging.INFO,
                    filename="./log/stdout.log",
                    filemode="a"
                    )


class TreeRetriever:

    def __init__(self, conf, tree) -> None:
        if not isinstance(tree, Tree):
            raise ValueError("tree must be an instance of Tree")

        self.conf = conf
        if self.conf["tokenizer"] is not None and isinstance(self.conf["tokenizer"], str):
            self.conf["tokenizer"] = tiktoken.get_encoding(self.conf["tokenizer"])
        self.tree = tree
        
        if self.conf["start_layer"] is None:
            if self.conf["abstract_layer_as_context"] > 0: 
                # if context needs higher abstracts, start searching from root
                self.conf["start_layer"] = min(self.conf["abstract_layer_as_context"], self.tree.num_layers)
            else: 
                # otherwise, start searching from the first layer with more than top_k nodes
                for layer, node_idx_list in reversed(self.tree.layer_to_node_indices.items()):
                    if len(node_idx_list) >= self.conf["tree_top_k"]:
                        self.conf["start_layer"] = layer
                        break
                if self.conf["start_layer"] is None:
                    raise ValueError(f"top k value ({self.conf["tree_top_k"]}) is larger than the number of leaf nodes ({len(self.tree.layer_to_node_indices[0])})")
        elif self.conf["start_layer"] > self.tree.num_layers:
            self.conf["start_layer"] = self.tree.num_layers

        self.tree_node_index_to_layer = reverse_mapping(self.tree.layer_to_node_indices)
        
        self.stemmer = Stemmer.Stemmer("english")
        self.hybrid_search_model = bm25s.BM25() if self.conf["hybrid_search"] else None
        if self.hybrid_search_model is not None and self.conf["save_dir"] is not None:
            hybrid_save_dir = os.path.join(self.conf["save_dir"], f"bm25_{self.conf["dataset"]}")
            if self.conf["force_sparse_index_from_scratch"] and os.path.exists(hybrid_save_dir):
                shutil.rmtree(hybrid_save_dir)
            if os.path.exists(hybrid_save_dir): 
                self.hybrid_search_model = self.hybrid_search_model.load(hybrid_save_dir, load_corpus=True)
                logging.info(f"Loaded vocab from \"{hybrid_save_dir}\".")
            os.makedirs(hybrid_save_dir, exist_ok=True)
    
    def embed(self, text: str) -> List[float]:
        return self.conf["embed_model"].embed(text)


    def _tree_retrieve(
        self, current_nodes: List[Node], query: str, start_layer: int
    ) -> Tuple[List[Node], List[str], List[float]]:

        query_embedding = self.embed(query)
        
        selected_nodes = []
        node_list = current_nodes

        for layer in range(start_layer, -1, -1):
            
            # 1) Calculate embeddings between query and node
            embeddings = get_embeddings(node_list)
            distances = distances_from_embeddings(query_embedding, embeddings, self.conf["distance"])
            indices = np.argsort(distances)

            # 2) Remove duplicate nodes
            embeddings = np.asarray(embeddings)
            mask = np.any(embeddings[indices[1:]] != embeddings[indices[:-1]], axis=-1)
            indices = indices[np.concatenate(([True], mask))]

            if self.conf["selection_mode"] == "threshold":
                best_indices = [
                    index for index in indices if distances[index] > self.conf["threshold"]
                ]
            elif self.conf["selection_mode"] == "top_k":
                # 3) Extract top-k nodes
                best_indices = indices[: self.conf["tree_top_k"]]

            nodes_to_add = [node_list[idx] for idx in best_indices]

            if layer <= self.conf["abstract_layer_as_context"]:
                selected_nodes.extend(nodes_to_add)
                
                if layer == 0:
                    if self.conf["distance"] == "cosine":
                        # normalized distance as document scores
                        scores = (2 - np.asarray(distances)) / 2
                    else:
                        raise NotImplementedError
                    scores = scores[best_indices]
            
            # 4) Add all children to the candidate set
            if layer > 0:
                child_nodes = []
                for index in best_indices:
                    child_nodes.extend(node_list[index].children)
                child_nodes = list(dict.fromkeys(child_nodes))
                node_list = [self.tree.all_nodes[i] for i in child_nodes]

        context = get_text_list(selected_nodes)
        return selected_nodes, context, scores.tolist()

    def retrieve(
        self,
        query: str,
        max_tokens: int = 3500,
        tokenizer_lock: Lock = None,
    ) -> Tuple[List[str], List[Dict], float, Dict[str, float]]:
        
        if not isinstance(query, str):
            raise ValueError("query must be a string")

        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be an integer and at least 1")

        start_time = time.time()

        layer_nodes = [self.tree.all_nodes[idx] for idx in self.tree.layer_to_node_indices[self.conf["start_layer"]]]
        retrieved_nodes, _, scores = self._tree_retrieve(
            layer_nodes, query, self.conf["start_layer"]
        )
        retrieved_node_indices = [node.index for node in retrieved_nodes]
        single_retrieval_time = time.time() - start_time

        sparse_start_time = time.time()
        if self.conf["hybrid_search"] and self.hybrid_search_model is not None:
            hybrid_node_indices = self._hybrid_retrieve(query, self.conf["sparse_top_k"])
        else:
            hybrid_node_indices = []
        sparse_time = time.time() - sparse_start_time
        
        rerank_start_time = time.time()
        if self.conf["rerank"] and self.conf["rerank_model"] is not None:
            # Reranking
            all_retrieved_docs = {self.tree.all_nodes[idx].text: self.tree.all_nodes[idx].index  
                                  for idx in retrieved_node_indices + hybrid_node_indices}
            if self.conf["rerank_batch_size"] >= self.conf["rerank_top_k"]:
                self.conf["rerank_batch_size"] = -1
            if self.conf["multithreading_qa_batch_size"] > 1:
                with tokenizer_lock:
                    rerank_scores = self.conf["rerank_model"].rerank(query=query, 
                                                                     documents=list(all_retrieved_docs.keys()),
                                                                     batch_size=self.conf["rerank_batch_size"])
            else:
                rerank_scores = self.conf["rerank_model"].rerank(query=query, 
                                                                 documents=list(all_retrieved_docs.keys()),
                                                                 batch_size=self.conf["rerank_batch_size"])
            final_node_indices = sorted(zip(all_retrieved_docs.values(), rerank_scores), 
                                        key=lambda x: x[1], 
                                        reverse=True)[:self.conf["rerank_top_k"]]
            final_node_indices, scores = zip(*final_node_indices)
            final_nodes = [self.tree.all_nodes[idx] for idx in final_node_indices]
            context = get_text_list(final_nodes)
                
        else:
            # Combine to result sets with RRF
            retrieved_docs = {self.tree.all_nodes[idx].text: self.tree.all_nodes[idx].index  
                              for idx in retrieved_node_indices}
            hybrid_docs = {self.tree.all_nodes[idx].text: self.tree.all_nodes[idx].index  
                           for idx in hybrid_node_indices}
            context, scores = rrf([list(retrieved_docs.keys()), list(hybrid_docs.keys())], 
                                  top_k=self.conf["rerank_top_k"])
            retrieved_docs.update(hybrid_docs)
            final_node_indices = [retrieved_docs[passage] for passage in context]
            final_nodes = [self.tree.all_nodes[idx] for idx in final_node_indices]
        rerank_time = time.time() - rerank_start_time
        
        def _add_info(context, final_nodes):
            '''Prepend extra info like chunk ID to mark relative positions. Used for summarization.'''
            for i in range(len(context)):
                node = final_nodes[i]
                ctx = context[i]
                if len(node.children):
                    context[i] = f"[LAYER: {self.tree_node_index_to_layer[node.index]}] " + ctx
                else:
                    context[i] = f"[ID: {node.index}] " + ctx
            return context
        
        if self.conf["abstract_layer_as_context"] or self.conf["answer_type"] == "long":
            context = _add_info(context, final_nodes)

        end_time = time.time()

        layer_information = []
        for i, node in enumerate(final_nodes):
            layer_information.append(
                {
                    "node_index": node.index,
                    "document_index": node.document_index,
                    "chunk_index": node.chunk_index,
                    "layer_number": self.tree_node_index_to_layer[node.index],
                    "score": scores[i],
                }
            )

        return (context, layer_information, end_time - start_time, 
               {'tree': single_retrieval_time, 'sparse': sparse_time, 'rerank': rerank_time})

    def hybrid_index(self, docs: List[str]) -> None:
        '''Build a sparse keyword index with BM25. '''
        if hasattr(self.hybrid_search_model, "vocab_dict"):
            return
        
        if self.conf["save_dir"] is not None:
            hybrid_save_dir = os.path.join(self.conf["save_dir"], f"bm25_{self.conf["dataset"]}")
            if os.path.exists(hybrid_save_dir): 
                corpus_tokens = bm25s.tokenize(docs, stopwords="en", stemmer=self.stemmer, show_progress=False)
                self.hybrid_search_model.index(corpus_tokens)
                self.hybrid_search_model.save(hybrid_save_dir)
        else:
            corpus_tokens = bm25s.tokenize(docs, stopwords="en", stemmer=self.stemmer, show_progress=False)
            self.hybrid_search_model.index(corpus_tokens)

    def _hybrid_retrieve(self, query: str, top_k: int | str = 5) -> List[int]:
        '''Retrieve chunks from the sparse keyword index. '''
        if self.hybrid_search_model is None:
            raise ValueError("There is no model for hybrid search.")
        elif not hasattr(self.hybrid_search_model, "vocab_dict"):
            raise ValueError("There is no index for hybrid search. Call ``hybrid_index()`` first. ")

        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer, show_progress=False)
        retrieved_node_indices, scores = list(map(lambda x: x[0], 
            self.hybrid_search_model.retrieve(query_tokens, 
                k=min(5 * top_k, self.hybrid_search_model.scores["num_docs"]), 
                sorted=True, 
                show_progress=False
            )
        ))
        
        retrieved_docs = {}
        # Deduplication
        for node_idx in retrieved_node_indices: 
            text = self.tree.all_nodes[node_idx].text
            duplicate_fn = lambda t, db: t in db
            if duplicate_fn(text, retrieved_docs):
                retrieved_docs[text] = min(int(node_idx), retrieved_docs[text])
            else:
                retrieved_docs[text] = int(node_idx)

        return list(retrieved_docs.values())[:top_k]
