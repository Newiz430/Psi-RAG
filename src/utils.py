import logging
import re
import os
import json
import string
import numpy as np
import tiktoken

from typing import Dict, Tuple, List, Set
from pathlib import Path
from scipy import spatial
from tqdm import tqdm


logging.basicConfig(format="%(asctime)s - %(message)s", 
                    level=logging.INFO,
                    filename="./log/stdout.log",
                    filemode="a"
                    )


class Node:
    """
    Represents a node in the hierarchical tree structure.
    """

    def __init__(self, text: str, index: int, document_index: int, chunk_index: int, children: Set[int], embeddings: np.ndarray) -> None:
        self.text: str = text
        self.index: int = index
        self.document_index: int = document_index
        self.chunk_index: int = chunk_index
        self.children: Set[int] = children
        self.embeddings: np.ndarray = embeddings


class Tree:
    """
    Represents the entire hierarchical tree structure.
    """

    def __init__(
        self, all_nodes, root_nodes, leaf_nodes, layer_to_node_indices
    ) -> None:
        self.all_nodes: List[Node] = all_nodes
        self.root_nodes: List[Node] = root_nodes
        self.leaf_nodes: List[Node] = leaf_nodes
        self.num_layers: int = max(layer_to_node_indices.keys())
        # self.num_layers = num_layers
        self.layer_to_node_indices: Dict[int, List[int]] = layer_to_node_indices


def reverse_mapping(layer_to_node_indices: Dict[int, List[int]]) -> Dict[int, int]:
    node_to_layer = {}
    for layer, nodes in layer_to_node_indices.items():
        for node in nodes:
            node_to_layer[node] = layer
    return node_to_layer


def split_text(
    text: str, tokenizer: str, max_tokens: int, overlap: int = 0
) -> List[str]:
    """
    Splits the input text into smaller chunks based on the tokenizer and maximum allowed tokens.
    
    Args:
        text (str): The text to be split.
        tokenizer (CustomTokenizer): The tokenizer to be used for splitting the text.
        max_tokens (int): The maximum allowed tokens.
        overlap (int, optional): The number of overlapping tokens between chunks. Defaults to 0.
    
    Returns:
        List[str]: A list of text chunks.
    """
    if tokenizer is None:
        raise ValueError("There is no tokenizer. ")
    tokenizer = tiktoken.get_encoding(tokenizer)

    # Split the text into sentences using multiple delimiters
    delimiters = [".", "!", "?", "\n"]
    regex_pattern = "|".join(map(re.escape, delimiters))
    sentences = re.split(regex_pattern, text)
    
    # Calculate the number of tokens for each sentence
    n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence, token_count in zip(sentences, n_tokens):
        # If the sentence is empty or consists only of whitespace, skip it
        if not sentence.strip():
            continue
        
        # If the sentence is too long, split it into smaller parts
        if token_count > max_tokens:
            sub_sentences = re.split(r"[,;:]", sentence)
            
            # there is no need to keep empty os only-spaced strings
            # since spaces will be inserted in the beginning of the full string
            # and in between the string in the sub_chunk list
            filtered_sub_sentences = [sub.strip() for sub in sub_sentences if sub.strip() != ""]
            sub_token_counts = [len(tokenizer.encode(" " + sub_sentence)) 
                                for sub_sentence in filtered_sub_sentences]
            
            sub_chunk = []
            sub_length = 0
            
            for sub_sentence, sub_token_count in zip(filtered_sub_sentences, sub_token_counts):
                if sub_length + sub_token_count > max_tokens:
                    
                    # if the phrase does not have sub_sentences, it would create an empty chunk
                    # this big phrase would be added anyways in the next chunk append
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                        sub_chunk = sub_chunk[-overlap:] if overlap > 0 else []
                        sub_length = sum(sub_token_counts[max(0, len(sub_chunk) - overlap):len(sub_chunk)])
                
                sub_chunk.append(sub_sentence)
                sub_length += sub_token_count
            
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
        
        # If adding the sentence to the current chunk exceeds the max tokens, start a new chunk
        elif current_length + token_count > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = current_chunk[-overlap:] if overlap > 0 else []
            current_length = sum(n_tokens[max(0, len(current_chunk) - overlap):len(current_chunk)])
            current_chunk.append(sentence)
            current_length += token_count
        
        # Otherwise, add the sentence to the current chunk
        else:
            current_chunk.append(sentence)
            current_length += token_count
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def distances_from_embeddings(
    query_embedding: List[float],
    embeddings: List[List[float]],
    distance_metric: str = "cosine",
) -> List[float]:
    """
    Calculates the distances between a query embedding and a list of embeddings.

    Args:
        query_embedding (List[float]): The query embedding.
        embeddings (List[List[float]]): A list of embeddings to compare against the query embedding.
        distance_metric (str, optional): The distance metric to use for calculation. Defaults to 'cosine'.

    Returns:
        List[float]: The calculated distances between the query embedding and the list of embeddings.
    """
    distance_metrics = {
        "cosine": spatial.distance.cosine,
        "L1": spatial.distance.cityblock,
        "L2": spatial.distance.euclidean,
        "Linf": spatial.distance.chebyshev,
    }

    if distance_metric not in distance_metrics:
        raise ValueError(
            f"Unsupported distance metric '{distance_metric}'. Supported metrics are: {list(distance_metrics.keys())}"
        )

    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances


def get_node_list(node_dict: Dict[int, Node]) -> List[Node]:
    """
    Converts a dictionary of node indices to a sorted list of nodes.

    Args:
        node_dict (Dict[int, Node]): Dictionary of node indices to nodes.

    Returns:
        List[Node]: Sorted list of nodes.
    """
    indices = sorted(node_dict.keys())
    node_list = [node_dict[index] for index in indices]
    return node_list


def get_embeddings(node_list: List[Node]) -> List:
    """
    Extracts the embeddings of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.
        embedding_model (str): The name of the embedding model to be used.

    Returns:
        List: List of node embeddings.
    """
    return [node.embeddings for node in node_list]


def get_children(node_list: List[Node]) -> List[Set[int]]:
    """
    Extracts the children of nodes from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        List[Set[int]]: List of sets of node children indices.
    """
    return [node.children for node in node_list]


def get_text(node_list: List[Node]) -> str:
    """
    Generates a single text string by concatenating the text from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        str: Concatenated text.
    """
    text = ""
    for node in node_list:
        text += f"{' '.join(node.text.splitlines())}"
        text += "\n\n"
    return text


def get_text_list(node_list: List[Node]) -> List[str]:
    """
    Generates a text string list from a list of nodes.

    Args:
        node_list (List[Node]): List of nodes.

    Returns:
        List: List of text.
    """
    text_list = []
    for node in node_list:
        text_list.append(node.text)
    return text_list


def get_token_length(text: str | List[str]) -> Dict:

    if isinstance(text, str):
        text = [text]
    
    tokens = []

    for t in text:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        # Split the text into sentences using multiple delimiters
        delimiters = [".", "!", "?", "\n"]
        regex_pattern = "|".join(map(re.escape, delimiters))
        sentences = re.split(regex_pattern, t)
        
        # Calculate the number of tokens for each sentence
        tokens.append(sum([len(tokenizer.encode(" " + sentence)) for sentence in sentences]))
    
    return {
        "total_token": sum(tokens), 
        "avg_token": float(sum(tokens)) / len(text),
        "max_token": max(tokens),
        "min_token": min(tokens),
    }


def prototype_embeddings(embeddings, method='average'):
    if method == 'average':
        return np.mean(embeddings, axis=0)
    elif method == 'tf-idf':
        return NotImplementedError
    else:
        return NotImplementedError


def normalize_answer(answer: str) -> str:
    """
    Normalize a given string by applying the following transformations:
    1. Convert the string to lowercase.
    2. Remove punctuation characters.
    3. Remove the articles "a", "an", and "the".
    4. Normalize whitespace by collapsing multiple spaces into one.

    Args:
        answer (str): The input string to be normalized.

    Returns:
        str: The normalized string.
    """

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(answer))))


def parse_response(response, verbose=False):
    if isinstance(response, str):
        thought, action, info = response.rpartition('Retrieve: ')
        if action != 'Retrieve: ':
            thought, action, info = response.rpartition('Answer: ')
            if action != 'Answer: ':
                if verbose:
                    tqdm.write(f"Warning: LLM output not as intended: \n{response}")
                return "", "answer", "Error"
        
        return thought.rstrip('\n'), action.rstrip(' :').lower(), info.rstrip(' .')
    else: # dict
        if response.retrieve is not None and response.answer is None:
            return response.thought, "retrieve", response.retrieve
        elif response.retrieve is None and response.answer is not None:
            return response.thought, "answer", response.answer
        else:
            raise ValueError(f"response.answer: {response.answer}\nresponse.retrieve: {response.retrieve}")
            

def rrf(docs: List[List[str]], top_k: int = 5, k: int = 60) -> Tuple[List[str], List[float]]:
    node_scoring_sheet = {}
    for passages in docs:
        for rank, passage in enumerate(passages, 1):
            node_scoring_sheet.setdefault(passage, 0)
            node_scoring_sheet[passage] += 1 / (rank + k)
    
    node_scoring_sheet = dict(sorted(node_scoring_sheet.items(), key=lambda x: x[1], reverse=True))
    return list(node_scoring_sheet.keys())[:top_k], list(node_scoring_sheet.values())[:top_k]


def save_answers(conf: Dict, results: Dict, ans_path: Path) -> None:
    results["conf"] = repr(conf)
    if not os.path.exists(ans_path):
        os.makedirs(ans_path)
    ans_file = os.path.join(ans_path, f"{conf["config"]}.json")
    with open(ans_file, "w") as f:
        json.dump(results, f)
    logging.info(f"QA results successfully saved to {ans_file}!")
    tqdm.write(f"Question answering completed! Answer file saved to \"{ans_file}\".")


def load_answers(conf: Dict) -> None | Tuple[List[str], List[List[str]]]:
    ans_file = os.path.join(conf["save_dir"], "results", f"{conf["config"]}.json")
    if not os.path.exists(ans_file):
        logging.info("No answer file detected. Running from scratch.")
        return None
    if conf["force_qa_from_scratch"]:
        logging.info("\"force_qa_from_scratch\" is on. Running from scratch.")
        os.remove(ans_file)
        return None
    with open(ans_file, "r") as f:
        results = json.load(f)
    if not {"conf", "answers"}.issubset(results.keys()):
        logging.info("Invalid result file.")
        return None
    logging.info(f"QA results successfully loaded from {ans_file}!")
    return results