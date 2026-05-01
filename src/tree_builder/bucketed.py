import copy
import logging
import math
import time
import numpy as np

from abc import abstractmethod
from typing import Dict, List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from .abstract import AbstractTreeBuilder, get_unionfind_children
from ..utils import Node, get_text, prototype_embeddings, reverse_mapping


def _normalize_embeddings(node_embeddings: np.ndarray) -> np.ndarray:
    node_embeddings = np.asarray(node_embeddings, dtype=np.float32)
    norms = np.linalg.norm(node_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return node_embeddings / norms


class BucketedAbstractTreeBuilder(AbstractTreeBuilder):

    def __init__(self, conf) -> None:
        super().__init__(conf)
        self.conf.setdefault("bucket_max_fanout", 32)
        self.conf.setdefault("bucket_sample_size", 8192)
        self.conf.setdefault("bucket_kmeans_iters", 5)

    @abstractmethod
    def _get_unionfind_tree_fragment(self, node_embeddings: np.ndarray) -> List[int]:
        pass

    def _split_group(
        self,
        normalized_embeddings: np.ndarray,
        group: np.ndarray,
    ) -> List[List[int]]:
        group = np.asarray(group, dtype=int)
        group_size = group.shape[0]
        bucket_size = max(2, int(self.conf["bucket_size"]))
        if group_size <= bucket_size:
            return [group.tolist()]

        fanout = min(
            max(2, int(math.ceil(group_size / bucket_size))),
            int(self.conf.get("bucket_max_fanout", 32)),
            group_size,
        )
        if fanout <= 1:
            return [group.tolist()]

        group_embeddings = normalized_embeddings[group]
        # Target sampling budget for estimating coarse bucket centers.
        # The actual sample count may exceed bucket_sample_size when fanout * 8 is larger.
        sample_size = min(
            group_size,
            max(fanout * 8, int(self.conf.get("bucket_sample_size", 8192))),
        )
        sample_idx = np.linspace(0, group_size - 1, num=sample_size, dtype=int)
        sample_embeddings = group_embeddings[sample_idx]
        center_idx = np.linspace(0, sample_embeddings.shape[0] - 1, num=fanout, dtype=int)
        centers = sample_embeddings[center_idx].copy()

        for step in range(int(self.conf.get("bucket_kmeans_iters", 5))):
            sims = sample_embeddings @ centers.T
            labels = np.argmax(sims, axis=1)
            for center_id in range(fanout):
                mask = labels == center_id
                if not np.any(mask):
                    centers[center_id] = sample_embeddings[
                        (step + center_id) % sample_embeddings.shape[0]
                    ]
                    continue
                center = sample_embeddings[mask].mean(axis=0)
                norm = np.linalg.norm(center)
                if norm > 0:
                    center = center / norm
                centers[center_id] = center

        sims = group_embeddings @ centers.T
        labels = np.argmax(sims, axis=1)
        buckets = []
        for center_id in range(fanout):
            bucket = group[labels == center_id].tolist()
            if bucket:
                buckets.append(bucket)

        if len(buckets) <= 1:
            buckets = [
                group[start: start + bucket_size].tolist()
                for start in range(0, group_size, bucket_size)
            ]

        return buckets

    def _coarse_partition(self, node_embeddings: np.ndarray) -> List[List[int]]:
        n = node_embeddings.shape[0]
        if n <= 1:
            return [list(range(n))]

        normalized_embeddings = _normalize_embeddings(node_embeddings)
        groups = [np.arange(n, dtype=int)]
        buckets = []

        while groups:
            group = np.asarray(groups.pop(), dtype=int)
            if group.shape[0] <= int(self.conf["bucket_size"]):
                buckets.append(group.tolist())
                continue
            groups.extend(self._split_group(normalized_embeddings, group))

        buckets = [sorted(bucket) for bucket in buckets if bucket]
        buckets.sort(key=lambda x: (len(x), x[0]))
        return buckets

    def _build_fragment(
        self,
        leaf_nodes: Dict[int, Node],
    ) -> Tuple[Dict[int, Node], Dict[int, List[int]], List[int]]:
        local_all_nodes = copy.deepcopy(leaf_nodes)
        layer_to_node_indices = {0: sorted(leaf_nodes.keys())}
        if len(leaf_nodes) <= 1:
            return local_all_nodes, layer_to_node_indices, layer_to_node_indices[0]

        node_indices_to_layer = reverse_mapping(layer_to_node_indices)
        all_node_emb = np.asarray(
            [local_all_nodes[node_id].embeddings for node_id in range(len(local_all_nodes))]
        )
        tree = self._get_unionfind_tree_fragment(all_node_emb)
        children = get_unionfind_children(tree)

        for parent_index, children_indices in children.items():
            child_layer = node_indices_to_layer[children_indices[0]]
            layer_to_node_indices.setdefault(child_layer + 1, []).append(parent_index)
            node_indices_to_layer[parent_index] = child_layer + 1
        layer_to_node_indices = dict(sorted(layer_to_node_indices.items()))

        children, node_indices_to_layer = self._rebalance(
            tree, children, layer_to_node_indices, node_indices_to_layer
        )

        for node_list in list(layer_to_node_indices.values()):
            if node_indices_to_layer[node_list[0]] == 0:
                continue
            for node in node_list:
                local_all_nodes[node] = Node(
                    text=None,
                    index=node,
                    document_index=-1,
                    chunk_index=-1,
                    children=set(children[node]),
                    embeddings=prototype_embeddings(
                        np.asarray([local_all_nodes[i].embeddings for i in children[node]])
                    ),
                )

        root_layer = max(layer_to_node_indices.keys())
        return local_all_nodes, layer_to_node_indices, layer_to_node_indices[root_layer]

    def _pad_fragment_roots(
        self,
        fragment_nodes: Dict[int, Node],
        fragment_layers: Dict[int, List[int]],
        root_ids: List[int],
        target_layer: int,
    ) -> Tuple[Dict[int, Node], Dict[int, List[int]], List[int]]:
        if not root_ids:
            return fragment_nodes, fragment_layers, root_ids

        root_id = root_ids[0]
        current_layer = max(fragment_layers.keys())
        next_index = max(fragment_nodes.keys()) + 1 if fragment_nodes else 0

        while current_layer < target_layer:
            fragment_nodes[next_index] = Node(
                text=None,
                index=next_index,
                document_index=-1,
                chunk_index=-1,
                children={root_id},
                embeddings=np.asarray(fragment_nodes[root_id].embeddings),
            )
            fragment_layers.setdefault(current_layer + 1, []).append(next_index)
            root_id = next_index
            next_index += 1
            current_layer += 1

        return fragment_nodes, dict(sorted(fragment_layers.items())), [root_id]

    def _generate_abstracts(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_node_indices: Dict[int, List[int]],
        start_layer: int = 1,
    ) -> None:
        node_indices_to_layer = reverse_mapping(layer_to_node_indices)
        target_nodes = [
            node
            for layer, node_list in sorted(layer_to_node_indices.items())
            if layer >= start_layer
            for node in node_list
            if len(all_tree_nodes[node].children) > 0
        ]

        tqdm.write(f"Pending abstract nodes: {len(target_nodes)}")
        bar = tqdm(target_nodes, desc="generating abstracts")
        for node in bar:
            node_layer = node_indices_to_layer[node]
            children = set(all_tree_nodes[node].children)
            if not self.conf["exclude_abs"]:
                node_texts = get_text([all_tree_nodes[i] for i in children])
                abstracts = self.abstract(
                    text=node_texts,
                    max_abs_length=self.conf["max_abs_length"],
                    leaf=node_layer == 1,
                )
                all_tree_nodes[node] = self.create_node(
                    node,
                    text=abstracts,
                    children_indices=children,
                )[1]
            else:
                all_tree_nodes[node].embeddings = prototype_embeddings(
                    np.asarray([all_tree_nodes[i].embeddings for i in children])
                )
        bar.close()

    def _commit_fragment(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_node_indices: Dict[int, List[int]],
        fragment_nodes: Dict[int, Node],
        fragment_layers: Dict[int, List[int]],
        leaf_local_to_global: Dict[int, int],
        layer_offset: int = 0,
    ) -> List[int]:
        local_to_global = dict(leaf_local_to_global)
        next_index = max(all_tree_nodes.keys()) + 1 if all_tree_nodes else 0

        for local_idx in sorted(fragment_nodes.keys()):
            if local_idx in local_to_global:
                continue
            local_to_global[local_idx] = next_index
            next_index += 1

        for local_idx, node in fragment_nodes.items():
            if local_idx in leaf_local_to_global:
                continue
            global_idx = local_to_global[local_idx]
            child_global_indices = {local_to_global[child] for child in node.children}
            all_tree_nodes[global_idx] = Node(
                text=node.text,
                index=global_idx,
                document_index=node.document_index,
                chunk_index=node.chunk_index,
                children=child_global_indices,
                embeddings=node.embeddings,
            )

        for local_layer, node_list in fragment_layers.items():
            global_layer = local_layer + layer_offset
            layer_to_node_indices.setdefault(global_layer, [])
            for local_idx in node_list:
                if local_layer == 0 and local_idx in leaf_local_to_global:
                    continue
                layer_to_node_indices[global_layer].append(local_to_global[local_idx])

        root_layer = max(fragment_layers.keys())
        return [local_to_global[idx] for idx in fragment_layers[root_layer]]

    def _build_bucketed_hierarchy(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_node_indices: Dict[int, List[int]],
        base_node_indices: List[int],
        base_layer: int = 0,
    ) -> List[int]:
        if len(base_node_indices) <= 1:
            return list(base_node_indices)

        base_embeddings = np.asarray([all_tree_nodes[idx].embeddings for idx in base_node_indices])
        buckets = self._coarse_partition(base_embeddings)

        fragments = []
        max_root_layer = 0
        for bucket in tqdm(buckets, desc="building buckets"):
            bucket_global_indices = [base_node_indices[idx] for idx in bucket]
            leaf_nodes = {}
            for local_idx, global_idx in enumerate(bucket_global_indices):
                node = all_tree_nodes[global_idx]
                leaf_nodes[local_idx] = Node(
                    text=node.text,
                    index=local_idx,
                    document_index=node.document_index,
                    chunk_index=node.chunk_index,
                    children=set(),
                    embeddings=np.asarray(node.embeddings),
                )
            fragment_nodes, fragment_layers, root_ids = self._build_fragment(leaf_nodes)
            fragments.append((bucket_global_indices, fragment_nodes, fragment_layers, root_ids))
            max_root_layer = max(max_root_layer, max(fragment_layers.keys()))

        bucket_root_indices = []
        for bucket_global_indices, fragment_nodes, fragment_layers, root_ids in fragments:
            fragment_nodes, fragment_layers, root_ids = self._pad_fragment_roots(
                fragment_nodes,
                fragment_layers,
                root_ids,
                max_root_layer,
            )
            leaf_local_to_global = {
                local_idx: global_idx
                for local_idx, global_idx in enumerate(bucket_global_indices)
            }
            bucket_root_indices.extend(
                self._commit_fragment(
                    all_tree_nodes,
                    layer_to_node_indices,
                    fragment_nodes,
                    fragment_layers,
                    leaf_local_to_global,
                    layer_offset=base_layer,
                )
            )

        if len(bucket_root_indices) <= 1:
            return bucket_root_indices

        meta_leaf_nodes = {}
        for local_idx, global_idx in enumerate(bucket_root_indices):
            node = all_tree_nodes[global_idx]
            meta_leaf_nodes[local_idx] = Node(
                text=node.text,
                index=local_idx,
                document_index=node.document_index,
                chunk_index=node.chunk_index,
                children=set(),
                embeddings=np.asarray(node.embeddings),
            )

        meta_nodes, meta_layers, _ = self._build_fragment(meta_leaf_nodes)
        meta_leaf_local_to_global = {
            local_idx: global_idx
            for local_idx, global_idx in enumerate(bucket_root_indices)
        }
        return self._commit_fragment(
            all_tree_nodes,
            layer_to_node_indices,
            meta_nodes,
            meta_layers,
            meta_leaf_local_to_global,
            layer_offset=base_layer + max_root_layer,
        )

    def _construct_tree(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_node_indices: Dict[int, List[int]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Building bucketed hierarchical abstract tree...")
        merge_start_time = time.time()

        root_indices = self._build_bucketed_hierarchy(
            all_tree_nodes,
            layer_to_node_indices,
            list(layer_to_node_indices[0]),
            base_layer=0,
        )

        merge_time = time.time() - merge_start_time
        abs_start_time = time.time()
        self._generate_abstracts(all_tree_nodes, layer_to_node_indices, start_layer=1)
        abs_time = time.time() - abs_start_time

        logging.info(f"Bucketed merge time: {merge_time:.2f}s")
        logging.info(f"Bucketed abstraction time: {abs_time:.2f}s")
        print(f"Bucketed merge time: {merge_time:.2f}s")
        print(f"Bucketed abstraction time: {abs_time:.2f}s")

        return {node_idx: all_tree_nodes[node_idx] for node_idx in root_indices}

    def _construct_tree_with_preset_chunks(
        self,
        all_tree_nodes: Dict[int, Node],
        layer_to_node_indices: Dict[int, List[int]],
        passage_to_node_indices: Dict[int, List[int]],
        use_multithreading: bool = False,
    ) -> Dict[int, Node]:
        logging.info("Building bucketed hierarchical abstract tree...")
        merge_start_time = time.time()

        def construct_passage_node(passage):
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
                range(0, max(list(passage_to_node_indices.keys())) + 1, passage_batch_size),
                desc="summarizing passage nodes",
            )
            for i in bar:
                with ThreadPoolExecutor() as executor:
                    future_passage_nodes = [
                        executor.submit(construct_passage_node, passage)
                        for passage in list(passage_to_node_indices.keys())[i: i + passage_batch_size]
                    ]
                    for future in as_completed(future_passage_nodes):
                        passage_level_nodes[future.result()[0]] = future.result()[1]
        else:
            bar = tqdm(passage_to_node_indices.keys(), desc="summarizing passage nodes")
            for passage in bar:
                passage_level_nodes[passage] = construct_passage_node(passage)[1]

        passage_level_nodes = dict(sorted(passage_level_nodes.items()))
        next_index = max(all_tree_nodes.keys()) + 1 if all_tree_nodes else 0
        passage_global_indices = []
        for passage_idx in sorted(passage_level_nodes.keys()):
            node = passage_level_nodes[passage_idx]
            node.index = next_index
            all_tree_nodes[next_index] = node
            passage_global_indices.append(next_index)
            next_index += 1

        layer_to_node_indices[1] = list(passage_global_indices)

        root_indices = self._build_bucketed_hierarchy(
            all_tree_nodes,
            layer_to_node_indices,
            passage_global_indices,
            base_layer=1,
        )

        merge_time = time.time() - merge_start_time
        abs_start_time = time.time()
        self._generate_abstracts(all_tree_nodes, layer_to_node_indices, start_layer=1)
        abs_time = time.time() - abs_start_time

        logging.info(f"Bucketed merge time: {merge_time:.2f}s")
        logging.info(f"Bucketed abstraction time: {abs_time:.2f}s")
        print(f"Bucketed merge time: {merge_time:.2f}s")
        print(f"Bucketed abstraction time: {abs_time:.2f}s")

        return {node_idx: all_tree_nodes[node_idx] for node_idx in root_indices}
