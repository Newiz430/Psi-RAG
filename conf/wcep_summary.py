conf = {
    "dataset": "wcep",
    # "force_index_from_scratch": True,
    "force_split": True,
    "passage_as_tree": True,
    "answer_type": "long",
    "max_retrieval_time": 0, 
    "max_response_length": 150,
    "tree_top_k": 15, 
    # "force_qa_from_scratch": True,
    "rerank": True,
    "rerank_top_k": 7,
    "rerank_name": "transformers:Qwen/Qwen3-Reranker-8B",
}