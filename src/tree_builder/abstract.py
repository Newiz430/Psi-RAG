import logging
import time
import copy
import numpy as np

from typing import Dict, List
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from bisect import bisect_right
from ordered_set import OrderedSet

from .base import TreeBuilder
from ..utils import Node, get_text, prototype_embeddings, reverse_mapping


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
        if insert_point is not None:
            self._tree[self._id[i][-1]] = self._id[insert_point][self._rank[i] + 1]
        else:
            self._tree[self._id[i][-1]] = self._next_id
            self._tree[self._id[j][-1]] = self._next_id
            self._id[i].append(self._next_id)
            self._next_id += 1

    @property
    def parent(self):
        return [list(self._parent[i]) for i in range(self.n)]

    @property
    def tree(self):
        return [self._tree[i] for i in range(2 * self.n - 1)][:self._next_id]


def get_unionfind_tree(
    node_embeddings: np.ndarray,
    partition_ratio: float = None,
    diagnostics: bool = False,
) -> List[int]:
    n = node_embeddings.shape[0]
    if n <= 1:
        return UnionFind(n).tree

    gb = 1024 ** 3
    start_time = time.time()
    dist_mat = -node_embeddings @ node_embeddings.mT
    matrix_mul_mem = dist_mat.nbytes / gb

    i, j = np.meshgrid(np.arange(n, dtype=int), np.arange(n, dtype=int))
    idx = np.tril_indices(n, -1)
    ij = np.stack([i[idx], j[idx]], axis=-1)
    dist_mat_upper = dist_mat[idx]
    if partition_ratio is None or partition_ratio <= 1.0:
        idx2 = np.argsort(dist_mat_upper, axis=0)
    else:
        k, ks = ij.shape[0], []
        while k > 0:
            k = int(k // partition_ratio)
            ks.append(k)
        ks = np.array(ks)[::-1]
        idx2 = np.argpartition(dist_mat_upper, ks, axis=0)
    ij = ij[idx2]
    ranking_mem = (
        i.nbytes +
        j.nbytes +
        idx[0].nbytes +
        idx[1].nbytes +
        ij.nbytes +
        dist_mat_upper.nbytes +
        idx2.nbytes
    ) / gb

    if diagnostics:
        end_time = time.time()
        tqdm.write(f"matrix mul & ranking time: {end_time - start_time:.2f}s")
        tqdm.write(f"matrix mul memory: {matrix_mul_mem:.4f} GB")
        tqdm.write(f"ranking memory: {ranking_mem:.4f} GB")

    uf = UnionFind(n)
    uf.merge(ij)
    return uf.tree


def get_unionfind_children(tree: List[int]) -> Dict[int, List[int]]:
    children = {}
    for i, j in enumerate(tree):
        if j == -1:
            continue
        children.setdefault(j, []).append(i)
    return children


class AbstractTreeBuilder(TreeBuilder):
    def __init__(self, conf) -> None:
        super().__init__(conf)
        self.conf.setdefault("partition_ratio", 1)
        self.conf.setdefault("tree_build_diagnostics", False)

    def _get_unionfind_tree(self, node_embeddings: np.ndarray) -> List[int]:
        return get_unionfind_tree(
            node_embeddings,
            self.conf["partition_ratio"],
            diagnostics=bool(self.conf.get("tree_build_diagnostics")),
        )

    def _rebalance(
        self,
        tree: List,
        children: Dict[int, List[int]],
        layer_to_node_indices: Dict[int, List[int]],
        node_indices_to_layer: Dict[int, int],
        keep_passage: bool = False,
    ) -> List:
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
                        new_node_index = current_node_index
                        layer_to_node_indices[l].append(new_node_index)
                        node_indices_to_layer[new_node_index] = l
                        current_node_index += 1
                        children[new_node_index] = children[node][split[i] : split[i + 1]]
                        if l == max(list(layer_to_node_indices.keys())):
                            new_root = current_node_index
                            layer_to_node_indices[l + 1] = [new_root]
                            node_indices_to_layer[new_root] = l + 1
                            children[new_root] = [node]
                            current_node_index += 1
                        for parent in layer_to_node_indices[l + 1]:
                            if node in children[parent]:
                                children[parent].append(new_node_index)
                    children[node] = children[node][:split[1]]
        return children, node_indices_to_layer

    def _construct_tree(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_node_indices: Dict[int, List[int]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Building hierarchical abstract tree...")

        tree_start_time = time.time()
        current_level_nodes = copy.deepcopy(all_tree_nodes)
        node_indices_to_layer = reverse_mapping(layer_to_node_indices)
        all_node_emb = np.asarray(
            [current_level_nodes[node_id].embeddings for node_id in range(len(current_level_nodes))]
        )

        tree = self._get_unionfind_tree(all_node_emb)
        children = get_unionfind_children(tree)

        for parent_index, children_indices in children.items():
            child_layer = node_indices_to_layer[children_indices[0]]
            layer_to_node_indices.setdefault(child_layer + 1, []).append(parent_index)
            node_indices_to_layer[parent_index] = child_layer + 1
        layer_to_node_indices = dict(sorted(layer_to_node_indices.items()))

        children, node_indices_to_layer = self._rebalance(
            tree, children, layer_to_node_indices, node_indices_to_layer
        )

        tree_build_time = time.time() - tree_start_time
        abs_start_time = time.time()

        bar = tqdm(range(len(children)), desc="generating abstracts")
        for node_list in list(layer_to_node_indices.values()):
            if node_indices_to_layer[node_list[0]] == 0:
                continue
            for node in node_list:
                if not self.conf["exclude_abs"]:
                    node_texts = get_text([current_level_nodes[i] for i in children[node]])
                    abstracts = self.abstract(
                        text=node_texts,
                        max_abs_length=self.conf["max_abs_length"],
                        leaf=node_indices_to_layer[node] == 1,
                    )
                    current_level_nodes[node] = self.create_node(
                        node,
                        text=abstracts,
                        children_indices=set(children[node]),
                    )[1]
                else:
                    current_level_nodes[node] = self.create_node(
                        node,
                        children_indices=set(children[node]),
                    )[1]
                    current_level_nodes[node].embeddings = prototype_embeddings(
                        np.asarray([current_level_nodes[i].embeddings for i in children[node]])
                    )
                bar.update(1)
        bar.close()

        all_tree_nodes.update(current_level_nodes)
        root_layer = max(layer_to_node_indices.keys())
        abs_end_time = time.time() - abs_start_time

        logging.info(f"Tree building time: {tree_build_time:.2f}s")
        logging.info(f"Abstraction time: {abs_end_time:.2f}s")
        print(f"Tree building time: {tree_build_time:.2f}s")
        print(f"Abstraction time: {abs_end_time:.2f}s")

        return {
            node_idx: current_level_nodes[node_idx]
            for node_idx in layer_to_node_indices[root_layer]
        }

    def _construct_tree_with_preset_chunks(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_node_indices: Dict[int, List[int]],
        passage_to_node_indices: Dict[int, List[int]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Building hierarchical abstract tree...")
        node_indices_to_layer = reverse_mapping(layer_to_node_indices)

        def construct_abstract_node(passage):
            if not self.conf["exclude_abs"]:
                node_texts = get_text([all_tree_nodes[node_index] for node_index in passage_to_node_indices[passage]])
                summarized_text = self.abstract(
                    text=node_texts,
                    max_abs_length=self.conf["max_abs_length"],
                    leaf=True,
                )
                passage_node = self.create_node(
                    passage,
                    text=summarized_text,
                    children_indices=set(passage_to_node_indices[passage]),
                )[1]
            else:
                passage_node = self.create_node(
                    passage,
                    children_indices=set(passage_to_node_indices[passage]),
                )[1]
                passage_node.embeddings = prototype_embeddings(
                    np.asarray([all_tree_nodes[i].embeddings for i in passage_to_node_indices[passage]])
                )
            return passage, passage_node

        passage_level_nodes = {}
        if use_multithreading:
            passage_batch_size = 10
            bar = tqdm(
                range(0, max(list(passage_to_node_indices.keys())), passage_batch_size),
                desc="generating 1st-layer abstracts",
            )
            for i in bar:
                with ThreadPoolExecutor() as executor:
                    future_passage_nodes = [
                        executor.submit(construct_abstract_node, passage)
                        for passage in list(passage_to_node_indices.keys())[i : i + passage_batch_size]
                    ]
                    for future in as_completed(future_passage_nodes):
                        passage_level_nodes[future.result()[0]] = future.result()[1]
        else:
            bar = tqdm(passage_to_node_indices.keys(), desc="generating 1st-layer abstracts")
            for passage in bar:
                passage_level_nodes[passage] = construct_abstract_node(passage)[1]

        passage_level_nodes = dict(sorted(passage_level_nodes.items()))
        passage_node_emb = np.asarray([passage_node.embeddings for passage_node in passage_level_nodes.values()])
        tree = self._get_unionfind_tree(passage_node_emb)

        passage_level_nodes = {k + len(all_tree_nodes): v for k, v in passage_level_nodes.items()}
        tree = [nid + len(all_tree_nodes) if nid > 0 else nid for nid in tree]
        children = get_unionfind_children(tree)
        for father, children_list in children.items():
            children[father] = [child + len(all_tree_nodes) for child in children_list]
        passage_tree = [node.document_index + len(all_tree_nodes) for node in all_tree_nodes.values()]
        children.update(get_unionfind_children(passage_tree))
        children = dict(sorted(children.items()))

        start_time = time.time()
        children, node_indices_to_layer = self._rebalance(
            tree,
            children,
            layer_to_node_indices,
            node_indices_to_layer,
            keep_passage=True,
        )
        rebalance_time = time.time() - start_time
        if self.conf.get("tree_build_diagnostics"):
            tqdm.write(f"Tree rebalancing time: {rebalance_time:.2f}s")

        for parent_index, children_indices in children.items():
            child_layer = node_indices_to_layer[children_indices[0]]
            layer_to_node_indices.setdefault(child_layer + 1, []).append(parent_index)
            node_indices_to_layer[parent_index] = child_layer + 1
        layer_to_node_indices = dict(sorted(layer_to_node_indices.items()))

        all_tree_nodes.update(passage_level_nodes)
        bar = tqdm(
            range(len(node_indices_to_layer) - len(all_tree_nodes) + 1),
            desc="generating higher abstracts",
        )
        for layer, node_list in layer_to_node_indices.items():
            if layer <= 1:
                continue
            for node in node_list:
                if not self.conf["exclude_abs"]:
                    node_texts = get_text([all_tree_nodes[i] for i in children[node]])
                    summarized_text = self.abstract(
                        text=node_texts,
                        max_abs_length=self.conf["max_abs_length"],
                        leaf=False,
                    )
                    all_tree_nodes[node] = self.create_node(
                        node,
                        text=summarized_text,
                        children_indices=set(children[node]),
                    )[1]
                else:
                    all_tree_nodes[node] = self.create_node(
                        node,
                        children_indices=set(children[node]),
                    )[1]
                    all_tree_nodes[node].embeddings = prototype_embeddings(
                        np.asarray([all_tree_nodes[i].embeddings for i in children[node]])
                    )
                bar.update(1)
        bar.close()

        root_layer = max(layer_to_node_indices.keys())
        return {node_idx: all_tree_nodes[node_idx] for node_idx in layer_to_node_indices[root_layer]}
