import os
import logging

from abc import ABC, abstractmethod
from typing import List
from tqdm import tqdm

import torch
import ollama
import numpy as np
import transformers
from torch.nn.functional import normalize
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer
from tenacity import retry, stop_after_attempt, wait_random_exponential

logging.basicConfig(format="%(asctime)s - %(message)s", 
                    level=logging.INFO,
                    filename="./log/stdout.log",
                    filemode="a"
                    )


class BaseEmbeddingModel(ABC):
    model_name: str

    @abstractmethod
    def embed(self, text) -> np.ndarray:
        pass

    def __repr__(self):
        return self.model_name


class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="text-embedding-ada-002", **kwargs):
        self.client = OpenAI()
        self.model_name = model_name

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def embed(self, text):
        text = text.replace("\n", " ")
        return (
            self.client.embeddings.create(input=[text], model=self.model_name)
            .data[0]
            .embedding
        )


class SentenceTransformersEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1", cache_dir=None, **kwargs):
        """
        Args:
            model_name (str): sentence-transformers/multi-qa-mpnet-base-cos-v1, 
                              sentence-transformers/all-mpnet-base-v2,
                              nvidia/NV-Embed-v2,
                              Qwen/Qwen3-Embedding-0.6B,
                              Qwen/Qwen3-Embedding-8B,
        """
        self.model_name = model_name
        self.model = None
        self.cache_dir = cache_dir

    def load_model(self):
        if self.model is None:
            if self.model_name in ["nvidia/NV-Embed-v2"]:
                assert transformers.__version__ <= "4.46.0", "Use old transformers package (<= 4.46.0)."
                self.model = SentenceTransformer(self.model_name, 
                                                 trust_remote_code=True, 
                                                 cache_folder=self.cache_dir,
                                                 config_kwargs={'use_cache': False},
                                                 )
                self.model.max_seq_length = 32768
                self.model.tokenizer.padding_side="right"
            else:
                self.model = SentenceTransformer(self.model_name)

    def add_eos(self, input_examples):
        input_examples = [input_example + self.model.tokenizer.eos_token for input_example in input_examples]
        return input_examples

    def embed(self, text, **kwargs):
        self.load_model()
            
        if self.model_name in ("nvidia/NV-Embed-v2",):
            text = self.add_eos(text)

            kwargs.setdefault("batch_size", 8)
            kwargs.setdefault("prompt", "")
            kwargs.setdefault("normalize_embeddings", True)
        elif self.model_name in ("Qwen/Qwen3-Embedding-8B",):
            kwargs.setdefault("batch_size", 32)
            
        return self.model.encode(text, show_progress_bar=False, **kwargs)


class OllamaEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="qwen3", **kwargs):
        """
        Args:
            model_name (str): qwen3-embedding:latest, 
        """
        self.model_name = model_name
    
    def embed(self, text):
        embs = ollama.embed(model=self.model_name, 
                            input=text,
                            keep_alive='10m',
                            ).embeddings[0]
        return embs


class TransformersEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_name="facebook/contriever", cache_dir=None, **kwargs):
        """
        Args:
            model_name (str): facebook/contriever, 
                              nvidia/NV-Embed-v2 (one GPU only), 
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.cache_dir = cache_dir
        if self.model_name in ("nvidia/NV-Embed-v2"):
            assert transformers.__version__ <= "4.46.0", "Use old transformers package (<= 4.46.0)."

    def load_model(self):
        if self.model is None:
            model_init_params = {
                "trust_remote_code": True,
                'device_map': "auto",  # added this line to use multiple GPUs
                "torch_dtype": "auto",
            }
            self.model = AutoModel.from_pretrained(self.model_name, 
                                                   mirror=os.environ["HF_ENDPOINT"], 
                                                   cache_dir=self.cache_dir,
                                                   **model_init_params)
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                                           mirror=os.environ["HF_ENDPOINT"],
                                                           cache_dir=self.cache_dir)

    def embed(self, text, **kwargs):
        self.load_model()
        if self.model_name in ("facebook/contriever"):
            return self._embed_contriever(text, **kwargs)
        elif self.model_name in ("nvidia/NV-Embed-v2"):
            return self._embed_nvidia(text, **kwargs)

    def _embed_contriever(self, text, **kwargs):
        kwargs.setdefault("normalize", True)
        kwargs.setdefault("output_hidden_states", False)

        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)

        return outputs
    
    def _embed_nvidia(self, text: List[str], **kwargs):
        
        if isinstance(text, str):
            text = [text]
        kwargs.setdefault("instruction", "")
        kwargs.setdefault("batch_size", 4)
        kwargs.setdefault("num_workers", 32)
        kwargs.setdefault("norm", True)

        if len(text) <= kwargs["batch_size"]:
            kwargs["prompts"] = text 
            embs = self.model.encode(**kwargs)
        else:
            embs = []
            bar = tqdm(range(0, len(text), kwargs["batch_size"]), desc="creating leaf nodes")
            for i in bar:
                kwargs["prompts"] = text[i:i + kwargs["batch_size"]]
                embs.append(self.model.encode(**kwargs))
            bar.close()
            embs = torch.cat(embs, dim=0)
        
        if kwargs["norm"]:
            embs = normalize(embs, p=2, dim=1)

        return embs.squeeze().cpu().numpy()

