import time
import numpy as np

from typing import Dict, List
from tqdm import tqdm

from .abstract import AbstractTreeBuilder, UnionFind


def _normalize_embeddings(node_embeddings: np.ndarray) -> np.ndarray:
    node_embeddings = np.asarray(node_embeddings, dtype=np.float32)
    norms = np.linalg.norm(node_embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return node_embeddings / norms


def _edge_scores_from_neighbors(
    node_embeddings: np.ndarray,
    neighbors: np.ndarray,
) -> Dict[tuple, float]:
    edge_to_score = {}
    for i, row in enumerate(neighbors):
        for j in row:
            j = int(j)
            if j < 0 or i == j:
                continue
            edge = (max(i, j), min(i, j))
            if edge not in edge_to_score:
                edge_to_score[edge] = float(node_embeddings[edge[0]] @ node_embeddings[edge[1]])
    return edge_to_score


def _component_groups(n: int, edges: np.ndarray) -> List[List[int]]:
    parent = np.arange(n, dtype=int)

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for u, v in edges:
        union(int(u), int(v))

    groups = {}
    for i in range(n):
        root = find(i)
        groups.setdefault(root, []).append(i)
    return sorted(groups.values(), key=len, reverse=True)


def _repair_disconnected_graph(
    node_embeddings: np.ndarray,
    edge_to_score: Dict[tuple, float],
) -> Dict[tuple, float]:
    n = node_embeddings.shape[0]
    if n <= 1:
        return edge_to_score

    edges = (
        np.asarray(list(edge_to_score.keys()), dtype=int)
        if edge_to_score
        else np.empty((0, 2), dtype=int)
    )
    components = _component_groups(n, edges)
    if len(components) <= 1:
        return edge_to_score

    anchor = np.asarray(components[0], dtype=int)
    for comp in components[1:]:
        comp = np.asarray(comp, dtype=int)
        scores = node_embeddings[anchor] @ node_embeddings[comp].T
        idx = np.unravel_index(np.argmax(scores), scores.shape)
        u = int(anchor[idx[0]])
        v = int(comp[idx[1]])
        edge = (max(u, v), min(u, v))
        edge_to_score[edge] = float(node_embeddings[edge[0]] @ node_embeddings[edge[1]])

    return edge_to_score


def _rank_edges(edge_to_score: Dict[tuple, float]) -> np.ndarray:
    if not edge_to_score:
        return np.empty((0, 2), dtype=int)
    ranked_edges = sorted(edge_to_score.items(), key=lambda x: x[1], reverse=True)
    return np.asarray([edge for edge, _ in ranked_edges], dtype=int)


class ANNAbstractTreeBuilder(AbstractTreeBuilder):

    def _get_neighbors(self, node_embeddings: np.ndarray, top_k: int) -> np.ndarray:
        raise NotImplementedError

    def _get_unionfind_tree(self, node_embeddings: np.ndarray) -> List[int]:
        n = node_embeddings.shape[0]
        if n <= 1:
            return UnionFind(n).tree

        top_k = max(1, min(int(self.conf.get("hnsw_top_k", 32)), n - 1))
        gb = 1024 ** 3
        start_time = time.time()

        neighbors = self._get_neighbors(node_embeddings, top_k)
        matrix_mul_mem = neighbors.nbytes / gb

        edge_to_score = _edge_scores_from_neighbors(node_embeddings, neighbors)
        edge_to_score = _repair_disconnected_graph(node_embeddings, edge_to_score)
        ij = _rank_edges(edge_to_score)
        ranking_mem = (
            ij.nbytes +
            len(edge_to_score) * (
                2 * np.dtype(np.int64).itemsize + np.dtype(np.float64).itemsize
            )
        ) / gb

        if self.conf.get("tree_build_diagnostics"):
            end_time = time.time()
            tqdm.write(f"matrix mul & ranking time: {end_time - start_time:.2f}s")
            tqdm.write(f"matrix mul memory: {matrix_mul_mem:.4f} GB")
            tqdm.write(f"ranking memory: {ranking_mem:.4f} GB")

        uf = UnionFind(n)
        uf.merge(ij)
        return uf.tree


class AbstractTreeBuilderHNSW(ANNAbstractTreeBuilder):

    def _get_neighbors(self, node_embeddings: np.ndarray, top_k: int) -> np.ndarray:
        try:
            import hnswlib
        except ModuleNotFoundError as e:
            raise ImportError(
                'AbstractTreeBuilderHNSW requires the "hnswlib" package. Install it first.'
            ) from e

        n = node_embeddings.shape[0]
        query_k = min(top_k + 1, n)
        gb = 1024 ** 3
        start_time = time.time()

        normalized_embeddings = _normalize_embeddings(node_embeddings)
        index = hnswlib.Index(space="cosine", dim=normalized_embeddings.shape[1])
        index.init_index(
            max_elements=n,
            ef_construction=int(self.conf.get("hnsw_ef_construction", 200)),
            M=int(self.conf.get("hnsw_m", 16)),
        )
        index.add_items(normalized_embeddings, np.arange(n))
        index.set_ef(max(query_k, int(self.conf.get("hnsw_ef_search", 64))))
        labels, _ = index.knn_query(normalized_embeddings, k=query_k)

        if self.conf.get("tree_build_diagnostics"):
            end_time = time.time()
            tqdm.write(f"hnsw matrix mul & ranking time: {end_time - start_time:.2f}s")
            tqdm.write(f"hnsw matrix mul memory: {labels.nbytes / gb:.4f} GB")
            tqdm.write(
                f"hnsw ranking memory: {normalized_embeddings.nbytes / gb:.4f} GB"
            )

        return labels
