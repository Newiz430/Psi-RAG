import os
import argparse

from typing import List
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from conf import apply_config_overrides, read_config
from src import (
    RAG,
    DataManager,
    OpenAIAbstractModel,
    OllamaAbstractModel,
    OllamaEmbeddingModel,
    VLLMEmbeddingModel,
    VLLMAbstractModel,
    SentenceTransformersEmbeddingModel,
    TransformersAbstractModel,
    TransformersEmbeddingModel,
)
from src.pdf import prepare_local_pdf_dataset
from src.utils import (
    get_token_length,
    is_bucketed_tree,
    get_tree_save_name,
    remove_tree_target,
    print_tree_check,
)


def main():
    if conf.get("read_local_pdf") is not None:
        print("Parsing local PDF file(s)...")
        conf["dataset"] = prepare_local_pdf_dataset(
            conf["data_dir"],
            conf["read_local_pdf"],
        )
        if conf["save_dir"] is not None:
            save_tree_name = get_tree_save_name(conf)
            save_tree_path = os.path.join(conf["save_dir"], save_tree_name)
            sparse_index_path = os.path.join(conf["save_dir"], f"bm25_{conf['dataset']}")
            if (os.path.exists(save_tree_path) or os.path.exists(sparse_index_path)) \
                and not conf["force_index_from_scratch"]:
                raise ValueError(
                    f'Dataset name "{conf["dataset"]}" already corresponds to an existing index. '
                    'Set "force_index_from_scratch=True" to overwrite it. \n'
                    'Note: if you do so, both the tree and the sparse index will be overwritten! '
                )
            if conf["force_index_from_scratch"]:
                conf["force_sparse_index_from_scratch"] = True
        conf["force_split"] = True

    data = DataManager(
        dataset_name=conf["dataset"],
        data_dir=conf["data_dir"],
        test_samples=conf["test_samples"],
        local_pdf=conf.get("read_local_pdf"),
        pdf_batch_size=conf.get("pdf_batch_size", 1),
    )

    if conf.get("tree_build_diagnostics"):
        if isinstance(data.all_passages, str):
            passages = [data.all_passages]
        else:
            passages = data.get_documents()
        tqdm.write(f"Dataset token stats: {get_token_length(passages)}")

    if isinstance(data.all_passages, str):
        conf["passage_as_tree"] = True
        conf["force_split"] = True
        data.split_text(
            tokenizer=conf["tokenizer"],
            max_tokens=conf["max_tokens_per_chunk"],
        )
    elif isinstance(data.all_passages, List) and isinstance(data.all_passages[0], str):
        if conf["passage_as_tree"] or conf["force_split"]:
            conf["force_split"] = True
            data.split_text(
                tokenizer=conf["tokenizer"],
                max_tokens=conf["max_tokens_per_chunk"],
            )
    elif isinstance(data.all_passages[0], List):
        if conf["force_split"]:
            data.split_text(
                tokenizer=conf["tokenizer"],
                max_tokens=conf["max_tokens_per_chunk"],
            )

    def set_model(model_name, task_type):
        framework, model_name = model_name.split(sep=":", maxsplit=1)
        model_class = {
            "ollama": {
                "embed": OllamaEmbeddingModel,
                "abs": OllamaAbstractModel,
            },
            "transformers": {
                "embed": TransformersEmbeddingModel,
                "abs": TransformersAbstractModel,
            },
            "sentence-transformers": {
                "embed": SentenceTransformersEmbeddingModel,
            },
            "vllm": {
                "embed": VLLMEmbeddingModel,
                "abs": VLLMAbstractModel,
            },
            "api": {
                "abs": OpenAIAbstractModel,
            },
        }[framework][task_type]
        model_kwargs = {
            "embed": conf["embed_model_kwargs"],
            "abs": conf["abs_model_kwargs"],
        }[task_type]

        return model_class(
            model_name,
            cache_dir=conf.get(f"{task_type}_cache_dir", None),
            **model_kwargs,
        )

    models_to_prepare = [model_type for model_type in conf.keys()
        if model_type.endswith("_name") and conf[model_type] is not None
        and model_type.rsplit("_name", maxsplit=1)[0] in ("embed", "abs")]
    for model_type in models_to_prepare:
        task_type = model_type.rsplit("_name", maxsplit=1)[0]
        conf[f"{task_type}_model"] = set_model(conf[model_type], task_type)

    tree_rag = RAG(conf)

    if conf["save_dir"] is not None:
        os.makedirs(conf["save_dir"], exist_ok=True)
        save_tree_name = get_tree_save_name(conf)
        save_tree_path = os.path.join(conf["save_dir"], save_tree_name)
        tree_target_name = "directory" if is_bucketed_tree(conf) else "file"

        if conf["force_index_from_scratch"]:
            tqdm.write("Constructing tree index...")
            if os.path.exists(save_tree_path):
                remove_tree_target(save_tree_path)
            tree_rag.add_documents(data)
            if conf.get("tree_build_diagnostics"):
                print_tree_check(tree_rag.tree)
            tree_rag.save("tree", save_tree_path)
            tqdm.write(
                f'Tree construction completed! Tree {tree_target_name} saved to "{save_tree_path}".'
            )
        elif not os.path.exists(save_tree_path):
            tqdm.write("Downloading tree index file from Hugging Face...")
            try:
                REPO_ID = "Newiz430/Psi-RAG"
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=save_tree_name,
                    local_dir=conf["save_dir"],
                )
                tree_rag.load("tree", save_tree_path)
                tqdm.write(
                    f'Downloading index completed! Tree {tree_target_name} saved to "{save_tree_path}".'
                )
            except Exception as e:
                tqdm.write(
                    f"Downloading index failed as an exception occured: \n{e}\nConstructing tree index from scratch..."
                )
            tree_rag.add_documents(data)
            if conf.get("tree_build_diagnostics"):
                print_tree_check(tree_rag.tree)
            tree_rag.save("tree", save_tree_path)
            tqdm.write(
                f'Tree construction completed! Tree {tree_target_name} saved to "{save_tree_path}".'
            )
        else:
            tqdm.write(
                f'Loading tree from existing index {tree_target_name} "{save_tree_path}"...'
            )
            tree_rag.load("tree", save_tree_path)
            tqdm.write("Loading tree completed!")
    else:
        tqdm.write("Constructing tree index...")
        tree_rag.add_documents(data)
        if conf.get("tree_build_diagnostics"):
            print_tree_check(tree_rag.tree)
        tqdm.write(f"Tree construction completed! Run `python qa.py --config {conf['config']}` to chat with your tree.")

    if conf["hybrid_search"]:
        tqdm.write("Constructing sparse index (BM25)...")
        tree_rag.build_vocab(data)
        tqdm.write("Sparse index construction completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        metavar="CONFIG FILE NAME",
        help='Config file name (without ".py") under the "conf" directory.',
    )
    args, unknown_args = parser.parse_known_args()

    try:
        conf = read_config(conf_name=args.config)
        conf = apply_config_overrides(conf, unknown_args)
    except (FileNotFoundError, AttributeError, ValueError) as e:
        parser.error(str(e))

    main()
