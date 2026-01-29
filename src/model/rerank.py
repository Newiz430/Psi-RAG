import os

from typing import List

import transformers
from abc import ABC, abstractmethod

import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification)


class BaseRerankModel(ABC):
    model_name: str
    
    @abstractmethod
    def rerank(self, query, documents):
        pass

    def __repr__(self):
        return self.model_name


class TransformersRerankModel(BaseRerankModel):
    def __init__(self, model_name="Qwen/Qwen3-Reranker-8B", cache_dir=None, **kwargs):
        """
        Args:
            model_name (str): BAAI/bge-reranker-large,
                              Qwen/Qwen3-Reranker-8B, 
        """
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.cache_dir = cache_dir

        if model_name in ("Qwen/Qwen3-Reranker-8B"):
            assert transformers.__version__ >= "4.51.0", "Use new transformers package (>= 4.51.0)."
    
    def load_model(self):
        if self.model_name in ("BAAI/bge-reranker-large"):
            self._load_model_bge()
        elif self.model_name in ("Qwen/Qwen3-Reranker-8B"):
            self._load_model_qwen()
        
    def _load_model_bge(self):
        if self.model is None:
            model_init_params = {
                "trust_remote_code": True,
                "device_map": "auto", 
                "torch_dtype": "auto",
            }
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, 
                mirror=os.environ["HF_ENDPOINT"],
                cache_dir=self.cache_dir,
                **model_init_params
            ).cuda().eval()
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                mirror=os.environ["HF_ENDPOINT"],
                cache_dir=self.cache_dir
            )
    
    def _load_model_qwen(self):
        if self.model is None:
            model_init_params = {
                "trust_remote_code": True,
                # "device_map": "auto",  # This triggers a BUG when using Qwen3-Reranker-8B! https://github.com/QwenLM/Qwen/issues/920
                "torch_dtype": "auto",
            }
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                mirror=os.environ["HF_ENDPOINT"],
                cache_dir=self.cache_dir,
                **model_init_params
            ).cuda().eval()
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                mirror=os.environ["HF_ENDPOINT"],
                cache_dir=self.cache_dir,
                padding_side='left'
            )
            
    def rerank(self, **kwargs) -> List[float]:
        self.load_model()
        if self.model_name in ("BAAI/bge-reranker-large"):
            return self._rerank_bge(**kwargs)
        elif self.model_name in ("Qwen/Qwen3-Reranker-8B"):
            return self._rerank_qwen(**kwargs)
        
    def _rerank_bge(self, query: str, documents: List[str], max_length: int = 512, **kwargs) -> List[float]:
        pairs = [[query, doc] for doc in documents]
        with torch.no_grad():
            inputs = self.tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=max_length)
            for key in inputs:
                inputs[key] = inputs[key].cuda()
            scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            return scores.cpu().numpy().tolist()

    def _rerank_qwen(self, query: str, documents: List[str], max_length: int = 8192, **kwargs) -> List[float]:
        token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        token_true_id = self.tokenizer.convert_tokens_to_ids("yes")

        prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        prefix_tokens = self.tokenizer.encode(prefix, add_special_tokens=False)
        suffix_tokens = self.tokenizer.encode(suffix, add_special_tokens=False)

        task = "Given a user question, retrieve relevant passages that answer the query"
        pairs = [f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {doc}" for doc in documents]
        
        def process_inputs(pairs):
            input_tokens = self.tokenizer(
                pairs, padding=False, truncation='longest_first',
                return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
            )
            for i, ele in enumerate(input_tokens['input_ids']):
                input_tokens['input_ids'][i] = prefix_tokens + ele + suffix_tokens
            input_tokens = self.tokenizer.pad(input_tokens, padding=True, return_tensors="pt", max_length=max_length)
            for key in input_tokens:
                input_tokens[key] = input_tokens[key].cuda()
            return input_tokens

        @torch.no_grad()
        def compute_logits(inputs):
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            return scores

        inputs = process_inputs(pairs)
        scores = compute_logits(inputs)

        return scores

    