import logging
import copy
import time
import numpy as np

from typing import Dict, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from bisect import bisect_right
from ordered_set import OrderedSet

from .tree_builder import TreeBuilder
from .utils import (Node, get_text, reverse_mapping, prototype_embeddings)


class UnionFind:

    def __init__(self, n: int):
        self.n = n
        self._parent = []
        self._rank = []

        for i in range(n):
            self._parent.append(OrderedSet([]))
            self._rank.append(0)

        self._next_id = n
        self._tree = []
        for i in range(2 * n - 1):
            self._tree.append(-1)
        self._id = []
        for i in range(n):
            self._id.append([i])

    def _find(self, i: int):
        if len(self._parent[i]) == 0 or (len(self._parent[i]) == 1 and self._parent[i][0] == i):
            return OrderedSet([i])
        else:
            if self._parent[i][0] == i:
                self._parent[i] |= self._find(self._parent[i][1])
            else:
                self._parent[i] |= self._find(self._parent[i][0])
            return self._parent[i]

    def find(self, i: int):
        if (i < 0) or (i > self.n):
            raise ValueError("Out of bounds index.")
        return self._find(i)
    
    def union(self, i: int, j: int):

        root_i = self._find(i)[-1]
        root_j = self._find(j)[-1]
        if root_i == root_j:
            return False
        else:
            if self._rank[root_i] < self._rank[root_j]:
                if len(self._parent[root_j]) == 0:
                    self._parent[root_j].add(root_j)
                higher_rank_idx = bisect_right(self._parent[j], self._rank[root_i], key=lambda x: self._rank[x])
                self._parent[root_i] |= self.parent[j][higher_rank_idx:]
                self._build(root_i, root_j, insert_point=self.parent[j][higher_rank_idx])
            elif self._rank[root_i] > self._rank[root_j]:
                if len(self._parent[root_i]) == 0:
                    self._parent[root_i].add(root_i)
                higher_rank_idx = bisect_right(self._parent[i], self._rank[root_j], key=lambda x: self._rank[x])
                self._parent[root_j] |= self.parent[i][higher_rank_idx:]
                self._build(root_j, root_i, insert_point=self.parent[i][higher_rank_idx])
            else:
                if len(self._parent[root_i]) == 0:
                    self._parent[root_i].add(root_i)
                self._parent[root_j] |= self._parent[i][-1:]
                self._rank[root_i] += 1
                self._build(root_i, root_j)
            return True

    def merge(self, ij: np.ndarray):
        """ Merge a sequence of pairs. """
        merge_time = 0
        bar = tqdm(range(self.n - 1), desc="merging")
        for coord in ij:
            is_merged = self.union(int(coord[0]), int(coord[1]))
            merge_time += is_merged
            bar.update(is_merged)
            if merge_time == self.n - 1:
                break
        bar.close()

    def _build(self, i: int, j: int, insert_point: int = None):
        """ Track the tree changes when node j gets merged into node i. """
        if insert_point is not None:
            self._tree[self._id[i][-1]] = self._id[insert_point][self._rank[i] + 1]
        else:
            self._tree[self._id[i][-1]] = self._next_id
            self._tree[self._id[j][-1]] = self._next_id
            self._id[i].append(self._next_id)
            self._next_id += 1

    @property
    def sets(self):
        return 2 * self.n - self._next_id

    @property
    def parent(self):
        return [list(self._parent[i]) for i in range(self.n)]

    @property
    def tree(self):
        return [self._tree[i] for i in range(2 * self.n - 1)][:self._next_id]

    @property
    def rank(self):
        return [self._rank[i] for i in range(self.n)]
    

def get_unionfind_tree(node_embeddings: np.ndarray, partition_ratio: float = None) -> List[int]:
    # similarity ranking
    n = node_embeddings.shape[0]
    dist_mat = -node_embeddings @ node_embeddings.mT

    # iterative merging and collapse
    i, j = np.meshgrid(np.arange(n, dtype=int), np.arange(n, dtype=int))
    idx = np.tril_indices(n, -1)
    ij = np.stack([i[idx], j[idx]], axis=-1)
    dist_mat_upper = dist_mat[idx]
    if partition_ratio is None or partition_ratio <= 1.:
        idx2 = np.argsort(dist_mat_upper, axis=0)
    else:
        k, ks = ij.shape[0], []
        while k > 0:
            k = int(k // partition_ratio)
            ks.append(k)
        ks = np.array(ks)[::-1]
        idx2 = np.argpartition(dist_mat_upper, ks, axis=0)
    ij = ij[idx2]

    uf = UnionFind(n)
    uf.merge(ij)

    return uf.tree

def get_unionfind_children(tree: List[int]) -> List[int]:
    children = {}
    for i, j in enumerate(tree):
        if j == -1:
            continue
        if j in children.keys():
            children[j].append(i)
        else:
            children[j] = [i]

    return children


class AbstractTreeBuilder(TreeBuilder):
    def __init__(self, conf) -> None:
        super().__init__(conf)

        if "partition_ratio" not in self.conf:
            self.conf["partition_ratio"] = 1

    def __rebalance(tree: List, children: List, keep_passage: bool = False) -> List:
        """Tree rebalancing: reorganize nodes with too many children. """
        if self.conf["max_num_children"] is not None and self.conf["max_num_children"] > 1:
            layers = max(list(layer_to_node_indices.keys()))
            current_node_index = len(tree)
            for l in range(layers):
                if l == 0:
                    continue
                if l == 1 and keep_passage:
                    continue
                node_list = layer_to_node_indices[l]
                for node in node_list:
                    batches = len(children[node]) // self.conf["max_num_children"] + 1
                    split = np.linspace(0, len(children[node]), num=batches + 1, dtype=int)
                    for i in range(1, batches):
                        # 1) create a new node in this layer
                        new_node_index = current_node_index
                        layer_to_node_indices[l].append(new_node_index)
                        node_indices_to_layer[new_node_index] = l
                        current_node_index += 1
                        # 2) change l-1 level node to point to the new node
                        children[new_node_index] = children[node][split[i] : split[i+1]]
                        # 3) if node has no parent (root), create an upper layer and a new root
                        if l == max(list(layer_to_node_indices.keys())):
                            new_root = current_node_index
                            layer_to_node_indices[l + 1] = [new_root]
                            node_indices_to_layer[new_root] = l + 1
                            children[new_root] = [node]
                            current_node_index += 1
                        # 4) set the parent's children
                        for parent in layer_to_node_indices[l + 1]:
                            if node in children[parent]:
                                children[parent].append(new_node_index)
                    children[node] = children[node][:split[1]]
        return children

    def __construct_tree(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_node_indices: Dict[int, List[int]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Building hierarchical abstract tree...")

        tree_start_time = time.time()

        # 1) Construct tree
        current_level_nodes = copy.deepcopy(all_tree_nodes)
        node_indices_to_layer = reverse_mapping(layer_to_node_indices)

        all_node_emb = np.asarray([current_level_nodes[node_id].embeddings
                                   for node_id in range(len(current_level_nodes))])
        
        tree = get_unionfind_tree(all_node_emb, self.conf["partition_ratio"])
        children = get_unionfind_children(tree)

        for parent_index, children_indices in children.items():
            child_layer = node_indices_to_layer[children_indices[0]]
            if child_layer + 1 in layer_to_node_indices.keys():
                layer_to_node_indices[child_layer + 1].append(parent_index)
            else:
                layer_to_node_indices[child_layer + 1] = [parent_index]
            node_indices_to_layer[parent_index] = child_layer + 1
        layer_to_node_indices = dict(sorted(layer_to_node_indices.items()))

        children = self.__rebalance(tree, children)

        tree_build_time = time.time() - tree_start_time
        sum_start_time = time.time()

        # 2) Generate abstracts & create higher nodes
        bar = tqdm(range(len(children)), desc="generating abstracts")
        for node_list in list(layer_to_node_indices.values()):
            if node_indices_to_layer[node_list[0]] == 0:
                continue
            for node in node_list:
                if not self.conf["exclude_abs"]:
                    # abstraction
                    node_texts = get_text([current_level_nodes[i] for i in children[node]])
                    abstracts = self.abstract(
                        text=node_texts,
                        max_abs_length=self.conf["max_abs_length"],
                        leaf=node_indices_to_layer[node] == 1
                    )
                    current_level_nodes[node] = self.create_node(node, text=abstracts, 
                                                                 children_indices=set(children[node]))[1]
                else: 
                    current_level_nodes[node] = self.create_node(node, children_indices=set(children[node]))[1]
                    
                    current_level_nodes[node].embeddings = prototype_embeddings(
                        np.asarray([current_level_nodes[i].embeddings for i in children[node]]))
                    
                bar.update(1)
        bar.close()

        all_tree_nodes.update(current_level_nodes)
        
        root_layer = max(layer_to_node_indices.keys())
        sum_end_time = time.time() - sum_start_time

        logging.info(f"Tree building time: {tree_build_time:.2f}s")
        logging.info(f"Summarization time: {sum_end_time:.2f}s")
        print(f"Tree building time: {tree_build_time:.2f}s")
        print(f"Summarization time: {sum_end_time:.2f}s")

        return {node_idx:current_level_nodes[node_idx] for node_idx in layer_to_node_indices[root_layer]}

    def __construct_passage_tree(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_node_indices: Dict[int, List[int]],
        passage_to_node_indices: Dict[int, List[int]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Building hierarchical abstract tree...")

        node_indices_to_layer = reverse_mapping(layer_to_node_indices)
        
        # 1) Construct passage nodes
        def construct_passage_node(passage):
            if not self.conf["exclude_abs"]:
                node_texts = get_text([all_tree_nodes[node_index] for node_index in passage_to_node_indices[passage]])
                summarized_text = self.abstract(
                    text=node_texts,
                    max_abs_length=self.conf["max_abs_length"],
                    leaf=True
                )
                passage_node = self.create_node(passage, text=summarized_text, children_indices=set(passage_to_node_indices[passage]))[1]
            else: 
                passage_node = self.create_node(passage, children_indices=set(passage_to_node_indices[passage]))[1]
                
                passage_node.embeddings = prototype_embeddings(
                    np.asarray([all_tree_nodes[i].embeddings for i in passage_to_node_indices[passage]])
                )
            return passage, passage_node
        
        passage_level_nodes = {}
        if use_multithreading:
            passage_batch_size = 10
            bar = tqdm(range(0, max(list(passage_to_node_indices.keys())), passage_batch_size), 
                       desc="summarizing passage nodes")
            for i in bar:
                with ThreadPoolExecutor() as executor:
                    future_passage_nodes = [
                        executor.submit(construct_passage_node, passage)
                        for passage in list(passage_to_node_indices.keys())[i : i + passage_batch_size]
                    ]

                    for future in as_completed(future_passage_nodes):
                        passage_level_nodes[future.result()[0]] = future.result()[1]

        else:
            bar = tqdm(passage_to_node_indices.keys(), desc="summarizing passage nodes")
            for passage in bar:
                passage_level_nodes[passage] = construct_passage_node(passage)[1]
        
        passage_level_nodes = dict(sorted(passage_level_nodes.items()))

        # 2) Construct passage tree
        passage_node_emb = np.asarray([passage_node.embeddings for passage_node in passage_level_nodes.values()])
        tree = get_unionfind_tree(passage_node_emb, self.conf["partition_ratio"])
        
        # 3) Synchronize node ids
        passage_level_nodes = {k + len(all_tree_nodes):v for k, v in passage_level_nodes.items()}
        tree = [nid + len(all_tree_nodes) if nid > 0 else nid for nid in tree]
        children = get_unionfind_children(tree)
        for father, children_list in children.items():
            children[father] = [child + len(all_tree_nodes) for child in children_list]
        passage_tree = [node.document_index + len(all_tree_nodes) for node in all_tree_nodes.values()]
        children.update(get_unionfind_children(passage_tree))
        children = dict(sorted(children.items()))

        # 4) Tree balancing: reorganize nodes with too many children
        # cunstruct_passage_tree deems the given passage layer (1) already balanced
        children = self.__rebalance(tree, children, keep_passage=True)

        # 5) Generate abstracts & create higher nodes
        for parent_index, children_indices in children.items():
            # layer indexing
            child_layer = node_indices_to_layer[children_indices[0]]
            if child_layer + 1 in layer_to_node_indices.keys():
                layer_to_node_indices[child_layer + 1].append(parent_index)
            else:
                layer_to_node_indices[child_layer + 1] = [parent_index]
            node_indices_to_layer[parent_index] = child_layer + 1
        layer_to_node_indices = dict(sorted(layer_to_node_indices.items()))

        all_tree_nodes.update(passage_level_nodes)

        bar = tqdm(range(len(node_indices_to_layer) - len(all_tree_nodes) + 1), 
                   desc="generateing higher abstracts")
        for layer, node_list in layer_to_node_indices.items():
            if layer <= 1:
                continue
            for node in node_list:
                if not self.conf["exclude_abs"]:
                    # summarizing
                    node_texts = get_text([all_tree_nodes[i] for i in children[node]])
                    summarized_text = self.abstract(
                        text=node_texts,
                        max_abs_length=self.conf["max_abs_length"],
                        leaf=False
                    )
                    all_tree_nodes[node] = self.create_node(node, text=summarized_text, 
                                                            children_indices=set(children[node]))[1]
                else: 
                    all_tree_nodes[node] = self.create_node(node, children_indices=set(children[node]))[1]
                    
                    all_tree_nodes[node].embeddings = prototype_embeddings(
                        np.asarray([all_tree_nodes[i].embeddings for i in children[node]])
                    )
                bar.update(1)
        bar.close()
        
        root_layer = max(layer_to_node_indices.keys())
        return {node_idx:all_tree_nodes[node_idx] for node_idx in layer_to_node_indices[root_layer]}
