import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime

import ollama
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import T5Tokenizer, T5ForConditionalGeneration
from .prompt import get_abs_template

logging.basicConfig(format="%(asctime)s - %(message)s", 
                    level=logging.INFO,
                    filename="./log/stdout.log",
                    filemode="a"
                    )


class BaseAbstractModel(ABC):
    model_name: str

    @abstractmethod
    def abstract(self, context, max_abs_length):
        pass

    def __repr__(self):
        return self.model_name


class OpenAIAbstractModel(BaseAbstractModel):
    def __init__(self, model_name="openai/gpt-5-mini", **kwargs):

        self.model_name = model_name
        self.client = OpenAI(
            base_url=os.environ["OPENAI_BASE_URL"],
            api_key=os.environ["OPENAI_API_KEY"]
        )
        self.kwargs = kwargs

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def abstract(self, context, keyword, max_abs_length, leaf=True):
        messages = get_abs_template(context, keyword=keyword, leaf=leaf, abs_max_length=max_abs_length)
        params = {
            "model": self.model_name,
            "messages": messages,
            "n": self.kwargs.get("number_of_response", 1),
            "temperature": self.kwargs.get("temperature", 0),
            "max_tokens": 5 * max_abs_length,
            "stream": False,
            "frequency_penalty": self.kwargs.get("frequency_penalty", 0),
            "presence_penalty": self.kwargs.get("presence_penalty", 0),
        }
        try:
            with open(f"./output/abs/{self.model_name.rsplit("/", maxsplit=1)[1]}.txt", "a") as f:
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


class OllamaAbstractModel(BaseAbstractModel):
    def __init__(self, model_name="qwen3", **kwargs):
        self.model_name = model_name

    def abstract(self, context, keyword, max_abs_length, leaf=True):

        try:
            params = {
                "model": self.model_name,
                "messages": get_abs_template(context, keyword=keyword, leaf=leaf, abs_max_length=max_abs_length),
                "options": {  
                    "num_predict": 5 * max_abs_length,
                },
                "stream": False,
                "keep_alive": '10m',
            }
            response = ollama.chat(**params)
            return response.message.content

        except Exception as e:
            print(e)
            return e


class TransformersAbstractModel(BaseAbstractModel):
    def __init__(self, model_name="Voicelab/vlt5-base-keywords", cache_dir=None, **kwargs):

        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.cache_dir = cache_dir

    def load_model(self):
        if self.model is None:
            model_init_params = {
                "trust_remote_code": True,
                'device_map': "auto",  # added this line to use multiple GPUs
                "torch_dtype": "auto",
            }
            if self.model_name in ("Voicelab/vlt5-base-keywords",):
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name, 
                    mirror=os.environ["HF_ENDPOINT"], 
                    cache_dir=self.cache_dir, 
                    **model_init_params
                )
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name, 
                    mirror=os.environ["HF_ENDPOINT"], 
                    cache_dir=self.cache_dir,
                )
            else:
                raise NotImplementedError

    def abstract(self, text, **kwargs):
        self.load_model()
        if self.model_name in ("Voicelab/vlt5-base-keywords",):
            return self.__abstract_vlt5(text, **kwargs)
        else:
            raise NotImplementedError

    def __abstract_vlt5(self, text, **kwargs):
        if isinstance(text, str):
            text = [text]

        input_ids = self.tokenizer(text, truncation=True, return_tensors="pt").input_ids
        output = self.model.generate(input_ids, no_repeat_ngram_size=3, num_beams=4)
        predicted = self.tokenizer.decode(output[0], skip_special_tokens=True)

        return predicted