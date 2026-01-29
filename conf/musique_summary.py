conf = {
    "dataset": "musique",
    # "force_index_from_scratch": True,
    "max_retrieval_time": 3, 
    "tree_top_k": 10, 
    # "force_qa_from_scratch": True,
    "hybrid_search": True, 
    "force_sparse_index_from_scratch": True,
    "sparse_top_k": 10, 
    "rerank": True,
    "rerank_top_k": 5,
    "rerank_name": "transformers:Qwen/Qwen3-Reranker-8B",
}