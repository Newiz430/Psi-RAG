import os
import argparse
import tiktoken

from typing import List
from tqdm import tqdm

from conf import apply_config_overrides, read_config
from src import RAG
from src.pdf import prepare_local_pdf_dataset
from src.utils import (
    get_token_length,
    reverse_mapping,
    is_bucketed_tree,
    get_tree_save_name,
    check_single_tree,
)


def get_path_size(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)

    total_size = 0
    for root, _, files in os.walk(path):
        for file_name in files:
            total_size += os.path.getsize(os.path.join(root, file_name))
    return total_size


def get_tree_builder(path: str) -> str:
    path_name = os.path.basename(path.rstrip("/\\"))
    return "approximate (hnsw)" if "_hnsw_" in path_name else "exact"


def is_bucketed_path(path: str) -> bool:
    if os.path.isdir(path):
        manifest_path = os.path.join(path, "manifest.json")
        return os.path.exists(manifest_path)
    return False

def get_tree_list(tree):
    return tree if isinstance(tree, List) else [tree]


def get_total_layers(tree) -> int:
    return max(tree.layer_to_node_indices.keys()) + 1


def to_internal_layer(tree, layer):
    return get_total_layers(tree) - 1 - layer


def format_id_list(ids):
    if len(ids) <= 100:
        return ids
    return ids[:95] + ["..."] + ids[-5:]


def truncate_text(text: str, max_tokens: int, tail_tokens: int = 50) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text

    tail_tokens = min(tail_tokens, max_tokens - 1)
    head_tokens = max_tokens - tail_tokens
    return (
        encoding.decode(tokens[:head_tokens]).strip()
        + "\n...\n"
        + encoding.decode(tokens[-tail_tokens:]).strip()
    )


def short_text(text: str, max_tokens: int = 20) -> str:
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return encoding.decode(tokens[:max_tokens]).strip() + " ..."


def flatten_nodes(tree):
    nodes = []
    for t in get_tree_list(tree):
        nodes.extend(t.all_nodes.values())
    return nodes


def get_selected_tree(tree, require_tree_id: bool = False):
    if isinstance(tree, List):
        if args.tree_id is None:
            if require_tree_id:
                raise ValueError("--tree_id is required for list tree indices.")
            return None, None
        if args.tree_id < 0 or args.tree_id >= len(tree):
            raise ValueError(f"tree_id must be in [0, {len(tree) - 1}].")
        return tree[args.tree_id], args.tree_id

    if args.tree_id is None or args.tree_id == 0:
        return tree, 0
    raise ValueError("tree_id must be 0 for a single tree index.")


def get_embedding_dim(nodes):
    return next(
        (
            int(node.embeddings.shape[-1]) if hasattr(node.embeddings, "shape")
            else len(node.embeddings)
            for node in nodes
            if node.embeddings is not None
        ),
        0,
    )


def get_layer_counts(tree):
    total_layers = get_total_layers(tree)
    return [
        len(tree.layer_to_node_indices[to_internal_layer(tree, layer)])
        for layer in range(total_layers)
    ]


def get_node_token_stats(nodes):
    token_lengths = [
        (node.index, get_token_length(node.text)["total_token"])
        for node in nodes
    ]
    avg_token = sum(token for _, token in token_lengths) / len(token_lengths)
    min_node_id, min_token = min(token_lengths, key=lambda x: (x[1], x[0]))
    max_node_id, max_token = max(token_lengths, key=lambda x: (x[1], -x[0]))
    return avg_token, min_token, min_node_id, max_token, max_node_id


def get_child_stats(nodes):
    child_counts = [
        (node.index, len(node.children))
        for node in nodes
        if len(node.children) > 0
    ]
    if not child_counts:
        return 0.0, 0, None, 0, None

    avg_children = sum(count for _, count in child_counts) / len(child_counts)
    min_node_id, min_children = min(child_counts, key=lambda x: (x[1], x[0]))
    max_node_id, max_children = max(child_counts, key=lambda x: (x[1], -x[0]))
    return avg_children, min_children, min_node_id, max_children, max_node_id


def get_tree_layer_stats(tree_list):
    tree_layers = [(tree_id, get_total_layers(tree)) for tree_id, tree in enumerate(tree_list)]
    avg_layers = sum(layer for _, layer in tree_layers) / len(tree_layers)
    min_tree_id, min_layers = min(tree_layers, key=lambda x: (x[1], x[0]))
    max_tree_id, max_layers = max(tree_layers, key=lambda x: (x[1], -x[0]))
    return avg_layers, min_layers, min_tree_id, max_layers, max_tree_id


# def preprocess_data():
#     data = DataManager(
#         dataset_name=conf["dataset"],
#         data_dir=conf["data_dir"],
#         test_samples=conf["test_samples"],
#         local_pdf=conf.get("read_local_pdf"),
#     )

#     if isinstance(data.all_passages, str):
#         conf["passage_as_tree"] = True
#         conf["force_split"] = True
#         data.split_text(
#             tokenizer=conf["tokenizer"],
#             max_tokens=conf["max_tokens_per_chunk"],
#         )
#     elif isinstance(data.all_passages, List) and isinstance(data.all_passages[0], str):
#         if conf["passage_as_tree"] or conf["force_split"]:
#             conf["force_split"] = True
#             data.split_text(
#                 tokenizer=conf["tokenizer"],
#                 max_tokens=conf["max_tokens_per_chunk"],
#             )
#     elif isinstance(data.all_passages[0], List):
#         if conf["force_split"]:
#             data.split_text(
#                 tokenizer=conf["tokenizer"],
#                 max_tokens=conf["max_tokens_per_chunk"],
#             )

#     return data


def load_tree():
    if conf["save_dir"] is None:
        raise ValueError('Tree inspection requires "save_dir".')

    save_tree_name = get_tree_save_name(conf)
    save_tree_path = os.path.join(conf["save_dir"], save_tree_name)
    tree_target_name = "directory" if is_bucketed_tree(conf) else "file"

    if not os.path.exists(save_tree_path):
        raise FileNotFoundError(
            f'Tree index {tree_target_name} "{save_tree_path}" not found. Please run "python index.py --config {conf["config"]}" first.'
        )

    tree_rag = RAG(conf)
    tqdm.write(
        f'Loading tree from existing index {tree_target_name} "{save_tree_path}"...'
    )
    tree_rag.load("tree", save_tree_path)
    tqdm.write("Loading tree completed!")
    return tree_rag, save_tree_path


def print_layer(tree):
    is_tree_list = isinstance(tree, List)
    tree, tree_id = get_selected_tree(tree, require_tree_id=True)

    total_layers = get_total_layers(tree)
    if args.layer == "root":
        layer = 0
    elif args.layer == "leaf":
        layer = total_layers - 1
    else:
        layer = int(args.layer)

    if layer < 0 or layer >= total_layers:
        raise ValueError(f"Layer must be in [0, {total_layers - 1}].")

    print("\n=========================== LAYER STATISTICS ============================\n")

    internal_layer = to_internal_layer(tree, layer)
    node_ids = list(tree.layer_to_node_indices[internal_layer])
    if is_tree_list:
        print(f"tree_id: {tree_id}")
    print(f"layer: {layer}")
    print(f"#nodes: {len(node_ids)}")
    print(f"node ids: {format_id_list(node_ids)}")

    print("\n=========================================================================\n")


def print_node(tree_rag):
    is_tree_list = isinstance(tree_rag.tree, List)
    tree, tree_id = get_selected_tree(tree_rag.tree, require_tree_id=True)

    if args.id not in tree.all_nodes:
        raise ValueError(f"Node id {args.id} not found.")

    node = tree.all_nodes[args.id]
    layer_map = reverse_mapping(tree.layer_to_node_indices)
    layer = to_internal_layer(tree, layer_map[args.id])
    text = truncate_text(node.text, args.max_print_size)

    print("\n============================ NODE STATISTICS ============================\n")

    if is_tree_list:
        print(f"tree_id: {tree_id}")
    print(f"id: {node.index}")
    if args.text_only:
        print(f"text:\n{text}")
        return

    print(f"layer: {layer}")
    print(f"text:\n{text}")

    tree_rag.tree = tree
    ancestors = tree_rag.find_ancestors(args.id)
    if ancestors:
        print("ancestors:")
        for ancestor_layer in sorted(ancestors.keys()):
            ancestor = ancestors[ancestor_layer]
            print(
                f"  layer {to_internal_layer(tree, ancestor_layer)}: "
                f"id={ancestor.index}, text={short_text(ancestor.text)}"
            )
    else:
        print("ancestors: None")

    child_ids = sorted(node.children)
    if child_ids:
        print("children:")
        for child_id in child_ids:
            child = tree.all_nodes[child_id]
            print(f"  id={child_id}, text={short_text(child.text)}")
    else:
        print("children: None")

    print("\n=========================================================================\n")


def print_tree_stats(tree, save_tree_path):
    is_tree_list = isinstance(tree, List)
    tree_list = get_tree_list(tree)
    # dataset_token_stats = get_token_length(data.get_documents())
    builder = get_tree_builder(save_tree_path)
    # is_bucketed = _is_bucketed_path(save_tree_path)

    print("\n============================ TREE STATISTICS ============================\n")
    print(f"dataset: {conf['dataset']}")
    if args.tree_id is not None:
        selected_tree, tree_id = get_selected_tree(tree)
        if is_tree_list:
            print(f"tree_id: {tree_id}")
    else:
        selected_tree = None
        embedding_dim = get_embedding_dim(flatten_nodes(tree))
    # print(f"dataset #tokens: {dataset_token_stats['total_token']}")
    print(f"#trees: {len(tree_list)}")
    print(f"tree path: {os.path.abspath(save_tree_path)}")
    print(f"tree size (GB): {get_path_size(save_tree_path) / (1024 ** 3):.4f}")
    print(f"builder: {builder}")
    if selected_tree is None and len(tree_list) > 1:
        print("connectivity: NA as --tree_id is not specified")
    else:
        tree_to_check = selected_tree if selected_tree is not None else tree_list[0]
        print(f"connected: {'yes' if check_single_tree(tree_to_check) else 'no'}")
    if selected_tree is None and len(tree_list) > 1:
        avg_layers, min_layers, min_tree_id, max_layers, max_tree_id = get_tree_layer_stats(tree_list)
        print(
            "#layers: "
            f"avg={avg_layers:.2f}, "
            f"min={min_layers} (tree id={min_tree_id}), "
            f"max={max_layers} (tree id={max_tree_id})"
        )
    else:
        tree_to_report = selected_tree if selected_tree is not None else tree_list[0]
        nodes = list(tree_to_report.all_nodes.values())
        layer_counts = get_layer_counts(tree_to_report)
        embedding_dim = get_embedding_dim(nodes)
        avg_token, min_token, min_node_id, max_token, max_node_id = get_node_token_stats(nodes)
        avg_children, min_children, min_child_node_id, max_children, max_child_node_id = get_child_stats(nodes)
        print(f"#layers: {get_total_layers(tree_to_report)}")
        print(f"#nodes per layer: {layer_counts}")
        # TODO: add bucket stats in saved tree index files
        # if is_bucketed:
        #     print("#buckets: unavailable from saved tree index")
        #     print("bucket size (#chunks): unavailable from saved tree index")
        # else:
        #     print("#buckets: not bucketed")
        #     print("bucket size (#chunks): not bucketed")
        print(
            "#tokens per node: "
            f"avg={avg_token:.2f}, "
            f"min={min_token} (node id={min_node_id}), "
            f"max={max_token} (node id={max_node_id})"
        )
        print(f"embedding dim: {embedding_dim}")
        if min_child_node_id is None:
            print("#children: avg=0.00, min=0, max=0")
        else:
            print(
                "#children: "
                f"avg={avg_children:.2f}, "
                f"min={min_children} (node id={min_child_node_id}), "
                f"max={max_children} (node id={max_child_node_id})"
            )
    if selected_tree is None and len(tree_list) > 1:
        print(f"embedding dim: {embedding_dim}")
    print("\n=========================================================================\n")


def main():
    if conf.get("read_local_pdf") is not None:
        conf["dataset"] = prepare_local_pdf_dataset(
            conf["data_dir"],
            conf["read_local_pdf"],
        )

    # data = preprocess_data()
    tree_rag, save_tree_path = load_tree()

    if args.layer is not None:
        print_layer(tree_rag.tree)
        return

    if args.id is not None:
        print_node(tree_rag)
        return

    print_tree_stats(tree_rag.tree, save_tree_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, metavar="CONFIG FILE NAME",
        help='Config file name (without ".py") under the "conf" directory.')
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--layer", type=str,
        help='Print node count and node ids for a layer. Use an integer, "root", or "leaf".')
    mode_group.add_argument("--id", type=int,
        help="Print detailed information for a node id.")
    parser.add_argument( "--max_print_size", type=int, default=500,
        help="Maximum token count when printing node text.")
    parser.add_argument("--text_only", action="store_true",
        help="Only print node id and text when used with --id.")
    parser.add_argument("--tree_id", type=int, default=None,
        help="Tree id for list tree indices.",)
    args, unknown_args = parser.parse_known_args()

    try:
        conf = read_config(conf_name=args.config)
        conf = apply_config_overrides(conf, unknown_args)
        if args.text_only and args.id is None:
            parser.error("--text_only must be used with --id.")
        print("Rat-a-tat-tat! The woodpecker is pecking at your tree...")
        main()
    except (FileNotFoundError, AttributeError, ValueError) as e:
        parser.error(str(e))
