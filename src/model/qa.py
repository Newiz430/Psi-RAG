import os

from openai import OpenAI
from datetime import datetime

import ollama
from abc import ABC, abstractmethod

from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import (AutoModelForCausalLM, AutoTokenizer)

class BaseQAModel(ABC):
    model_name: str
    
    @abstractmethod
    def qa(self, question):
        pass

    def __repr__(self):
        return self.model_name


class OpenAIQAModel(BaseQAModel):
    def __init__(self, model_name="openai/gpt-4o-mini", **kwargs):
        """
        Args:
            model_name (str): openai/gpt-4o-mini, openai/gpt-5-mini, 
                google/gemini-2.5-flash, 
                deepseek/deepseek-v3.2
        """
        self.model_name = model_name
        self.client = OpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"]
        )
        self.kwargs = kwargs

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def qa(self, question, max_tokens=1000, **kwargs):
        params = {
            "model": self.model_name,
            "messages": question,
            "n": self.kwargs.get("number_of_response", 1),
            "temperature": self.kwargs.get("temperature", 0),
            "max_tokens": max_tokens,
            "stream": False,
            "frequency_penalty": self.kwargs.get("frequency_penalty", 0),
            "presence_penalty": self.kwargs.get("presence_penalty", 0),
        }
            
        try:
            with open(f"./output/qa/{self.model_name.rsplit('/', maxsplit=1)[1]}.txt", "a") as f:
                retry_time = 10
                for i in range(retry_time):
                    response = self.client.chat.completions.create(**params)
                    answer = response.choices[0].message.content.strip()
                    if not answer == "":
                        break

                f.write(str(datetime.now()) + "\n" + answer + "\n\n")
        
            return answer
        
        except Exception as e:
            print(e)
            return e
        

class OllamaQAModel(BaseQAModel):
    def __init__(self, model_name="qwen3", cache_dir=None, **kwargs):
        """
        Args:
            model_name (str): qwen3:8b, 
                              qwen3:32b,
                              llama3.1:8b,
                              llama3.3,

        """
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.model_kwargs = kwargs
        self.think = model_name in ("qwen3:8b", "qwen3:30b", "qwen3:32b",)

    def qa(self, question, max_tokens=1000, **kwargs):
        stream = kwargs.get("stream", False)

        try:
            params = {
                "messages": question,
                "options": {
                    "num_ctx": self.model_kwargs.get("num_ctx", 4096),
                    "num_predict": self.model_kwargs.get("num_predict", max_tokens),
                    "temperature": self.model_kwargs.get("temperature", 0),
                    "repeat_penalty": self.model_kwargs.get("repeat_penalty", 1.3),
                    "top_k": self.model_kwargs.get("top_k", 40),
                    "top_p": self.model_kwargs.get("top_p", 0.9),
                },
                "model": self.model_name,
                "think": self.think,
                "stream": stream,
                "keep_alive": '10m',
            }
            
            response = ollama.chat(**params)
            if stream:
                return response
            if self.think:
                return response.message.thinking.strip() + "\n" + response.message.content.strip()
            else:
                return response.message.content.strip()
        except Exception as e:
            print(e)
            if stream:
                raise
            return e


class VLLMQAModel(BaseQAModel):
    def __init__(self, model_name="meta-llama/Llama-3.3-70B-Instruct", cache_dir=None, **kwargs):
        self.model_name = model_name
        self.model = None
        self.model_kwargs = kwargs
        self.cache_dir = cache_dir

    def load_model(self):
        if self.model is None:
            try:
                from vllm import LLM
            except ImportError as e:
                raise ImportError("vllm is not installed.") from e

            init_kwargs = self.model_kwargs.copy()
            if self.cache_dir is not None:
                init_kwargs["download_dir"] = self.cache_dir
            self.model = LLM(model=self.model_name, **init_kwargs)

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def qa(self, question, max_tokens=1000, **kwargs):
        self.load_model()
        from vllm import SamplingParams

        sampling_kwargs = {
            "n": self.model_kwargs.get("number_of_response", 1),
            "temperature": self.model_kwargs.get("temperature", 0),
            "max_tokens": max_tokens,
            "frequency_penalty": self.model_kwargs.get("frequency_penalty", 0),
            "presence_penalty": self.model_kwargs.get("presence_penalty", 0),
            "repetition_penalty": self.model_kwargs.get("repetition_penalty", 1.0),
            "top_k": self.model_kwargs.get("top_k", -1),
            "top_p": self.model_kwargs.get("top_p", 1.0),
        }
        if "stop" in self.model_kwargs:
            sampling_kwargs["stop"] = self.model_kwargs["stop"]
        sampling_params = SamplingParams(**sampling_kwargs)

        try:
            retry_time = 10
            for i in range(retry_time):
                response = self.model.chat(
                    question,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                answer = response[0].outputs[0].text.strip()
                if not answer == "":
                    break

            return answer

        except Exception as e:
            print(e)
            return e
        

class TransformersQAModel(BaseQAModel):
    def __init__(self, model_name: str = "meta-llama/Llama-3.3-70B-Instruct", cache_dir: str = None, **kwargs):

        self.model_name = model_name
        self.model = None
        self.model_kwargs = kwargs
        self.tokenizer = None
        self.cache_dir = cache_dir
    
    def load_model(self):
        if self.model is None:
            model_init_params = {
                "trust_remote_code": True,
                'device_map': "auto",  # added this line to use multiple GPUs
                "torch_dtype": "auto",
            }
            model_kwargs = self.model_kwargs.copy()
            model_kwargs.update(model_init_params)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, 
                                                              mirror=os.environ["HF_ENDPOINT"],
                                                              cache_dir=self.cache_dir,
                                                              **model_kwargs)
        if self.tokenizer is None:    
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, 
                                                           mirror=os.environ["HF_ENDPOINT"],
                                                           cache_dir=self.cache_dir,
                                                           )
            
    def qa(self, question, max_tokens=1000, **kwargs):
        self.load_model()

        input_ids = self.tokenizer.encode(question, return_tensors="pt")

        generate_params = {
            "temperature": 0,
            "max_new_tokens": max_tokens,
        }
        res = self.model.generate(input_ids, **generate_params)
        return self.tokenizer.batch_decode(res, skip_special_tokens=True)

