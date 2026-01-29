import os
import json

from typing import Dict, List, Set
from tqdm import tqdm

from .utils import split_text

dataset_pool = (
    "nq",
    "popqa",
    "narrativeqa",
    "hotpotqa",
    "2wikimultihopqa",
    "musique",
    "multihoprag",
    "medical",
    "novel",
    "infinitybench_longbook",
    "hypertension",
    "agriculture",
    "cs",
    "legal",
    "mix",
    "qmsum",
    "wcep",
    "booksum",
    "govreport",
    "squality",
)

class DataManager:
    
    def __init__(self,
                 dataset_name: str, 
                 data_dir: str = "./data",
                 test_samples: int = -1) -> None:
        
        self.dataset_name: str = dataset_name.lower()
        if self.dataset_name not in dataset_pool:
            raise NotImplementedError(f"Dataset {self.dataset_name} is currently not supported.")

        self.data: List[Dict] | None = None
        self.data_path: str | None = None 
        self.corpus: str | List[str] | None = None
        self.load_data(data_dir)

        # ------------------- FOR QA TESTING ONLY -------------------
        if test_samples > 0:
            if self.data is not None:
                self.data = self.data[:test_samples]
            if self.corpus is not None:
                self.corpus = self.corpus[:test_samples]
        # ------------------- FOR QA TESTING ONLY -------------------

        self.gold_docs: List[List[str]] | None = []
        self.gold_nodes_id: List[List[int]] | None = []
        self.get_gold_docs()

        self.gold_answers: List[Set[str]] | None = []
        self.get_gold_answers()
        
        self.all_queries: List[str] = []
        self.query_to_doc_ids: List[int] = []
        self.all_text_ids: List[str] = []
        self.all_passages: List[str] | List[List[str]] = []
        self.preprocess()

    def load_data(self, data_dir: str = "./data") -> None:
        try:
            if self.dataset_name in  ("hotpotqa", "2wikimultihopqa", "musique", 
                                      "nq", "popqa", "multihoprag", "narrativeqa"):
                self.data_path = os.path.join(data_dir, self.dataset_name) + ".json"
                corpus_path = os.path.join(data_dir, f"{self.dataset_name}_corpus") + ".json"
                self.corpus = json.load(open(corpus_path, "r"))
                self.data = json.load(open(self.data_path, "r"))
            elif self.dataset_name in ("infinitybench_longbook"):
                self.data_path = os.path.join(data_dir, self.dataset_name) + ".jsonl"
                self.data = []
                with open(self.data_path, 'rb') as file:
                    for line in file:
                        self.data.append(json.loads(line))
            elif self.dataset_name in ("qmsum"):
                self.data_path = os.path.join(data_dir, self.dataset_name) + ".jsonl"
                self.data = []
                with open(self.data_path, 'r') as file:
                    for line in file:
                        self.data.append(json.loads(line))
            elif self.dataset_name in ("wcep"):
                self.data_path = os.path.join(data_dir, self.dataset_name) + ".txt"
                self.data = []
                with open(self.data_path, 'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        self.data.append(json.loads(line))
            else:
                raise NotImplementedError
        except FileNotFoundError:
            raise FileNotFoundError(f"{self.data_path} not found.")

    def get_gold_docs(self) -> None:
        for sample in self.data:
            gold_node_id = []
            if 'supporting_facts' in sample:  # hotpotqa, 2wikimultihopqa
                gold_title = set([item[0] for item in sample['supporting_facts']])
                gold_title_and_content_list = []
                for i, item in enumerate(sample['context']):
                    if item[0] in gold_title:
                        gold_title_and_content_list.append(item)
                        gold_node_id.append(i)
                if self.dataset_name.startswith('hotpotqa'):
                    gold_doc = [item[0] + '\n' + ''.join(item[1]) for item in gold_title_and_content_list]
                else:
                    gold_doc = [item[0] + '\n' + ' '.join(item[1]) for item in gold_title_and_content_list]
            elif 'contexts' in sample:
                gold_doc = []
                for i, item in enumerate(sample['contexts']):
                    if item['is_supporting']:
                        gold_doc.append(item['title'] + '\n' + item['text'])
                        gold_node_id.append(i)
            elif 'paragraphs' in sample:
                gold_paragraphs = []
                for i, item in enumerate(sample['paragraphs']):
                    if 'is_supporting' in item and item['is_supporting'] is False:
                        continue
                    gold_paragraphs.append(item)
                    if 'idx' in item:
                        gold_node_id.append(item['idx'])
                    else:
                        gold_node_id.append(i)
                gold_doc = [item['title'] + '\n' + (item['text'] if 'text' in item else item['paragraph_text']) for item in gold_paragraphs]
            elif 'evidence_list' in sample: # multihoprag
                gold_doc = [evidence['fact'] for evidence in sample['evidence_list']]
                gold_node_id = -1
            else:
                print(f'Warning: dataset "{self.dataset_name}" seems to be lack of labels for retrieval evaluation.') 
                self.gold_docs = None
                self.gold_nodes_id = None
                return

            gold_doc = list(set(gold_doc))
            self.gold_docs.append(gold_doc)
            self.gold_nodes_id.append(gold_node_id)

    def get_gold_answers(self) -> None:
        for sample in self.data:
            gold_ans = None

            if 'answer' in sample:
                gold_ans = sample['answer']
            elif 'gold_ans' in sample:
                gold_ans = sample['gold_ans']
            elif 'reference' in sample:
                gold_ans = sample['reference']
            elif 'obj' in sample:
                gold_ans = set(
                    [sample['obj']] + [sample['possible_answers']] + [sample['o_wiki_title']] + [sample['o_aliases']])
                gold_ans = list(gold_ans)
            elif 'golden_answers' in sample:
                gold_ans = sample['golden_answers']
            elif self.dataset_name in ("qmsum"):
                gold_ans = sample['general_query_list'][0]['answer']
            elif self.dataset_name in ("wcep"):
                word_num = len(" ".join(sample["document"]).split())
                if word_num <= 6000:
                    continue
                gold_ans = sample["summary"]
            else:
                print(f'Warning: dataset "{self.dataset_name}" seems to be lack of labels for QA evaluation.') 
                self.gold_answers = None
                return

            if isinstance(gold_ans, str):
                gold_ans = [gold_ans]
            assert isinstance(gold_ans, list)
            gold_ans = set(gold_ans)
            if 'answer_aliases' in sample:
                gold_ans.update(sample['answer_aliases'])

            self.gold_answers.append(gold_ans)

    def preprocess(self):
        if self.corpus is not None: # musique, hotpotqa, 2wikimultihopqa, nq, popqa, narrativeqa, multihoprag
            bar = tqdm(self.corpus, desc="preprocessing text") 
        else: # qmsum, wcep, infinitybench_longbook
            bar = tqdm(self.data, desc="preprocessing text") 

        if self.dataset_name in ("narrativeqa", "infinitybench_longbook"):
            doc_dict: Dict[str, str|List[str]] = {}

        doc_idx = 0
        for sample_text in bar:
            if self.dataset_name in ('musique', 'hotpotqa', '2wikimultihopqa', 'nq', 'popqa'):
                self.all_text_ids.append(str(doc_idx))
                self.all_passages.append(sample_text['title'] + "\n" + sample_text['text'])   # List[List[str]]
            if self.dataset_name in ("multihoprag"):
                self.all_text_ids.append(str(doc_idx))
                self.all_passages.append(sample_text['title'] + "\n" + sample_text['body'])   # List[str]
            elif self.dataset_name in ('qmsum'):
                def clean_qmsum_data(text: str) -> str:
                    text = text.replace('{vocalsound}', '')
                    text = text.replace('{disfmarker}', '')
                    text = text.replace('A_M_I_', 'ami')
                    text = text.replace('L_C_D_', 'lcd')
                    text = text.replace('P_M_S', 'pms')
                    text = text.replace('T_V_', 'tv')
                    text = text.replace('{pause}', '')
                    text = text.replace('{nonvocalsound}', '')
                    text = text.replace('{gap}', '')
                    text = text.replace('  ', ' ')
                    return text
                self.all_text_ids.append(str(doc_idx))
                passage = "\n".join(["Speaker: " + i["speaker"] + "\n" + "Content: " + i["content"]
                                    for i in sample_text['meeting_transcripts']])
                self.all_passages.append(clean_qmsum_data(passage))   # List[str]
            elif self.dataset_name in ('wcep'):
                word_num = len(" ".join(sample_text["document"]).split())
                if word_num <= 6000:
                    # self.all_queries = self.all_queries[:-1]
                    continue
                self.all_passages.append(' '.join(sample_text["document"]))   # List[str]
                self.all_queries.append("Summarize the contents of this news event.")
                self.query_to_doc_ids.append(doc_idx)
            elif self.dataset_name in ("narrativeqa"):
                doc_id = sample_text['idx'].split('_', maxsplit=1)[0]
                if doc_id in doc_dict:
                    doc_dict[doc_id].append(sample_text['title'] + "\n" + sample_text['text'])
                else:
                    doc_dict[doc_id] = [sample_text['title'] + "\n" + sample_text['text']]   # List[List[str]]
            elif self.dataset_name in ("infinitybench_longbook"):
                doc_id = sample_text['context'][:100]
                doc_dict[doc_id] = sample_text['context']   # List[str]

            doc_idx += 1

        if self.dataset_name in ("narrativeqa", "infinitybench_longbook"):
            self.all_text_ids = list(doc_dict.keys())
            self.all_passages = list(doc_dict.values())
        bar.close()
                
        bar = tqdm(self.data, desc="preprocessing query")
        for i, sample_text in enumerate(bar):
            
            if self.dataset_name in ('narrativeqa'):
                self.all_queries.append(sample_text['question'])
                idx = self.all_text_ids.index(sample_text['document']['id'])
                self.query_to_doc_ids.append(idx)
            elif self.dataset_name in ('infinitybench_longbook'):
                self.all_queries.append(sample_text['input'])
                idx = self.all_text_ids.index(sample_text['context'][:100])
                self.query_to_doc_ids.append(idx)
            elif self.dataset_name in ('qmsum'):
                self.all_queries.append(sample_text['general_query_list'][0]['query'])
                self.query_to_doc_ids.append(i)
            elif 'query' in sample_text:
                self.all_queries.append(sample_text['query'])
                self.query_to_doc_ids.append(i)
            elif 'question' in sample_text:
                self.all_queries.append(sample_text['question'])
                self.query_to_doc_ids.append(i)
        bar.close()
        
    def split_text(self, **kwargs):
        """Split text into chunks according to dataset format."""
        if isinstance(self.all_passages, str):
            self.all_passages = split_text(self.all_passages, **kwargs) # List[str]
        elif isinstance(self.all_passages, List) and isinstance(self.all_passages[0], str): 
            self.all_passages = [split_text(t, **kwargs) for t in self.all_passages] # List[List[str]]
        elif isinstance(self.all_passages[0], List):
            self.all_passages = [split_text("\n".join(psg), **kwargs) for psg in self.all_passages] # List[List[str]]

    def get_documents(self) -> List[str]:
        """Get a flattened document list (for sparse retrieval)."""
        if isinstance(self.all_passages[0], List):
            docs = []
            for passage in self.all_passages:
                docs.extend(passage)
        else:
            docs = self.all_passages
        return docs
    
    def __len__(self) -> int:
        return len(self.data) if self.data is not None else 0
