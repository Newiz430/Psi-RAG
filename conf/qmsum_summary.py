conf = {
    "dataset": "qmsum",
    # "force_index_from_scratch": True,
    "force_split": True,
    "passage_as_tree": True,
    "answer_type": "long",
    "max_retrieval_time": 0,
    "max_response_length": 200,
    "tree_top_k": 20, 
    # "force_qa_from_scratch": True,
    "rerank": True,
    "rerank_top_k": 10,
    "rerank_name": "transformers:Qwen/Qwen3-Reranker-8B",
}