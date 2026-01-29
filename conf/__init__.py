import os
import importlib
from typing import Sequence, TypedDict
from pathlib import Path


class Config(TypedDict, total=False):
    # ===================================== Data config =====================================
    # Dataset name: "nq", "popqa", "hotpotqa", "2wikimultihopqa", "musique", 
    #   "multihoprag", "narrativeqa", "infinitybench_longbook", "qmsum", "wcep"
    dataset: str
    # Dataset directory, default to "./data"
    data_dir: Path
    # Number of samples for testing with part of the dataset. Data becomes data[:test_samples].
    test_samples: int
    # Always split the data even there are preset chunks. Used when preset chunks are 
    #   long passages: multihoprag, infinitybench-longbook, qmsum, wcep.
    force_split: bool
    # Tokenizer name (from tiktoken) for spliting datasets with no preset chunks.
    tokenizer: str
    # Maximum token count of one chunk when tokenizer splits the data
    max_tokens_per_chunk: int

    # ================================== Embedding config ==================================
    # Model name for embedding documents and queries, "[PLATFORM]:[MODEL_NAME_OR_PATH]"
    #   e.g., "transformers:nvidia/NV-Embed-v2", "transformers:facebook/contriever", 
    #   "sentence-transformers:multi-qa-mpnet-base-cos-v1", "ollama:qwen3-embedding"
    embed_name: str
    # Cache directory of the embedding model, default to os.environ["HF_HOME"]
    embed_cache_dir: Path

    # ================================= Tree indexing config =================================
    # Partition ratio for similarity ranking. Useful when corpus is fairly large. 
    #   Default to 1. Set >1 for faster-but-not-as-accurate similarity ranking.
    partition_ratio: float
    # The single-document retrieval setup, i.e., an independent tree index for each document.
    #   Used for passage-level and document-level datasets whose queries are passage-specific 
    #   (narrativeqa, infinitybench_longbook, qmsum, wcep).
    passage_as_tree: bool
    # Reorganize leaves such that chunks from different documents could share a parent node. 
    #   Default to False and chunks from the same document will be automatically merged. 
    #   Set this on when there are preset chunks to be manually split (multihoprag)
    #   Forced True when no preset chunks and force_split=False. 
    reorganize_leaf: bool
    # Maximum number of children per abstract node. Should be >3. 
    #   If a node has more than max_num_children children, tree rebalancing will be activated 
    #   to split the node and its children into multiple subtrees.
    #   You would like to set this value not smaller than 2 * tree_top_k 
    #   to ensure the amount of retrieved documents.
    #   Note: if the data has preset chunks and force_split=False, 
    #   the last abstract layer (penaltimate layer) will NOT be checked nor reorganized.
    max_num_children: int | None

    # ================================= Abstraction config ================================
    # Do not generate abstracts, use prototype embeddings instead. 
    #   Only for testing; setting this on drops the performance.
    exclude_abs: bool
    # Model name for node abstraction, "[PLATFORM]:[MODEL_NAME_OR_PATH]"
    #   e.g., "ollama:llama3.3:latest", "transformers:Voicelab/vlt5-base-keywords", 
    #   "api:google/gemini-2.5-flash"
    abs_name: str | None
    # Cache directory of the abstraction agent, default to os.environ["HF_HOME"].
    abs_cache_dir: Path
    # Type of abstract. "summary" for summative text and "keyword" for keywords.
    abstract_type: str | None
    # Maximum word count for node abstraction (for LLM prompting so not always work 
    #   depending on the LLM). Maximum keyword count for keyword abstract. 
    max_abs_length: int
    # Abstract nodes not higher than this layer will also be retrieved. 
    #   Default to 0 (to only retrieve leaf nodes). Generated abstracts occupy tree_top_k slots, 
    #   so retrieval evaluation results would be inaccurate.
    abstract_layer_as_context: int
    # Recreate embeddings and trees even if saved files exist in save_dir.
    #   Warning: your previous saved file will be covered! 
    force_index_from_scratch: bool

    # ===================================== R\&A Agent config ====================================
    # Model name for agentic retrieval and QA, "[PLATFORM]:[MODEL_NAME_OR_PATH]"
    #   e.g., "transformers:meta-llama/Llama-3.3-70B-Instruct", "ollama:llama3.3:latest", 
    #   "api:openai/gpt-5-mini"
    qa_name: str
    # Cache directory of the R&A agent, default to os.environ["HF_HOME"]
    qa_cache_dir: Path
    # The expected answer type. This shapes the system instruction to match the task.
    #   "short" for token-level tasks (single-hop and multi-hop qa).
    #   "medium" for passage-level tasks (narrative qa).
    #   "long" for document-level tasks (summarization).
    answer_type: str
    # Maximum time of retrieval attempts. The first retrieval is fixed and does not count. 
    max_retrieval_time: int
    # Maximum word count for LLM's thoughts or summary text (for LLM prompting 
    #   so not always work depending on the LLM). 
    #   You may want to lower this value if you are calling the OpenAI API, 
    #   but lowering this value may cause erroneous response parsing.
    max_response_length: int
    # Multithread qa for efficient reproduction on large corpus. If Ollama models are used, 
    #   it is recommended to set this equal to the environment variable "OLLAMA_NUM_PARALLEL"
    multithreading_qa_batch_size: int
    # Reanswering questions even if the answer result file exists in save_dir
    #   Warning: your previous saved file will be covered! 
    force_qa_from_scratch: bool

    # =============================== General retrieval config ==============================
    # Document selection mode during retrieval, ("top_k", "threshold"). Unused.
    selection_mode: str
    # Number of retrieved documents if selection_mode="top_k". Note that this is only 
    #   for tree retrieval; the final top-k always depend on rerank_top_k if set.
    #   This is the final top_k only if rerank_top_k is None or rerank_top_k == tree_top_k.
    tree_top_k: int
    # Threshold for retrieving documents if selection_mode="threshold". Unused.
    threshold: float
    # From what layer should tree search start. Set to None to automatically identify. 
    #   Set to 0 to search on the entire leaf set.
    start_layer: int | None
    # Metric to calculate vector distance for retrieval, ("cosine", "L1", "L2", "Linf"). 
    distance: str

    # =============================== Hybrid retrieval config ==============================
    # If sparse keyword search search is enabled (only BM25S is supported for now)
    hybrid_search: bool
    # Rebuild the sparse token vocab even if saved files exist in save_dir/<sparse_method>_<dataset>.
    #   Warning: your previous saved files will be covered! 
    force_sparse_index_from_scratch: bool
    # Number of retrieved documents by sparse keyword search. 
    sparse_top_k: int
    # Threshold for BM25. Unused.
    sparse_threshold: float
    # k in reciprocal rank fusion, only if rerank=False. Default to 60.
    rrf_k: int
    
    # =================================== Reranking config ==================================
    # If reranking is enabled
    rerank: bool
    # Model name for reranking, "[PLATFORM]:[MODEL_NAME_OR_PATH]"
    #   e.g., "transformers:Qwen/Qwen3-Reranker-8B", "transformers:BAAI/bge-reranker-large"
    rerank_name: str | None
    # Cache directory of the reranking model, default to os.environ["HF_HOME"]
    rerank_cache_dir: Path | None
    # Number of final returned documents by the reranker if selection_mode="top_k"
    rerank_top_k: int | None
    # Threshold for reranking score if selection_mode="threshold"
    rerank_threshold: float | None

    # ================================== Evaluation config ==================================
    # Set of evaluation metrics, ("em", "f1", "rouge", "recall", "answerrate")
    #   Default to "all" to use all supported metrics
    evaluation_metrics: str | Sequence[str]

    # ==================================== Other config =====================================
    # Save directory of everything intermediate: tree, query embedding, BM25 vocab, etc. 
    #   Set to None to skip saving for a single run.
    save_dir: Path
    # Path of your log file. 
    log_path: Path
    # Whether to output the detail information during QA.
    verbose: bool
    # Python file name under the conf directory. Can only be set by argparse["config"].
    config: str


# Default configuration. Each conf/<setting>.py applies custom setting on top of this.   
conf = Config(
    dataset="musique",
    data_dir="./data",
    test_samples=-1,
    force_split=False,
    tokenizer="cl100k_base",
    max_tokens_per_chunk=100,
    
    embed_name="ollama:qwen3-embedding",
    embed_cache_dir=None,
    
    partition_ratio=1.,
    passage_as_tree=False,
    reorganize_leaf=False,
    max_num_children=40,

    exclude_abs=False,
    abs_name="ollama:llama3.3:latest",
    abs_cache_dir=None,
    abstract_type="summary",
    max_abs_length=100,
    abstract_layer_as_context=0,
    force_index_from_scratch=False,
    
    qa_name="ollama:llama3.3:latest",
    qa_cache_dir=None,
    answer_type="short",
    max_retrieval_time=3,
    max_response_length=500,
    multithreading_qa_batch_size=int(os.environ["OLLAMA_NUM_PARALLEL"]) if "OLLAMA_NUM_PARALLEL" in os.environ else -1,
    force_qa_from_scratch=False,

    selection_mode="top_k",
    tree_top_k=5,
    threshold=0.5,
    start_layer=None,
    distance="cosine",

    hybrid_search=False,
    force_sparse_index_from_scratch=False,
    sparse_top_k=5,
    sparse_threshold=10,
    rrf_k=60,

    rerank=False,
    rerank_name=None,
    rerank_cache_dir=None,
    rerank_top_k=5,
    rerank_threshold=None,

    evaluation_metrics="all",

    save_dir="./output",
    log_path="./log",
    verbose=True,
)


def read_config(conf_name: str = None) -> Config:
    config_path = "./conf"
    if os.path.exists(os.path.join(config_path, f"{conf_name}.py")):
        module = importlib.import_module(f"conf.{conf_name}")
        
        if hasattr(module, 'conf'):
            conf.update(module.conf)
        else:
            raise AttributeError(f"There is no importable config in the config file \"{conf_name}\". ")
        
        conf["config"] = conf_name
    else:
        raise FileNotFoundError(f"Config file \"{conf_name}\" not found. ")

    return conf