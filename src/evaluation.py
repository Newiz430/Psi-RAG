import numpy as np

from typing import Dict, List, Tuple, Callable
from bisect import insort_right
from collections import Counter
from tqdm import tqdm

from rouge_score import rouge_scorer

from src.dataset import DataManager
from src.utils import normalize_answer


class Evaluator:
    
    def __init__(self,
                 data: DataManager,
                 top_k_nodes_per_layer: int = 10) -> None:
        
        self.data: DataManager = data

        rc_k_list = [1, 2, 5, 10, 20, 50, 100, 200]
        if top_k_nodes_per_layer not in rc_k_list:
            insort_right(rc_k_list, top_k_nodes_per_layer)
        self.rc_k_list: List[int] = rc_k_list[:rc_k_list.index(top_k_nodes_per_layer) + 1]

    def qa_exactmatch(self, 
                      predicted_answers: List[str],
                      aggregation_fn: Callable = np.max
                      ) -> Dict[str, float]:
        """
        Calculates the Exact Match (EM) score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]:
                - A dictionary with the averaged EM score.
                - A list of dictionaries with EM scores for each example.
        """
        assert len(self.data.gold_answers) == len(
            predicted_answers), "Length of gold answers and predicted answers should be the same."

        example_eval_results = []
        total_em = 0

        for gold_list, predicted in zip(self.data.gold_answers, predicted_answers):
            em_scores = [1.0 if normalize_answer(gold) == normalize_answer(predicted) else 0.0 for gold in gold_list]
            aggregated_em = aggregation_fn(em_scores)
            example_eval_results.append({"ExactMatch": aggregated_em})
            total_em += aggregated_em

        avg_em = total_em / len(self.data.gold_answers) if self.data.gold_answers else 0.0
        pooled_eval_results = {"ExactMatch": avg_em}

        return pooled_eval_results

    def qa_f1(self,
              predicted_answers: List[str],
              aggregation_fn: Callable = np.max
              ) -> Dict[str, float]:
        """
        Calculates the F1 score.

        Args:
            gold_answers (List[List[str]]): List of lists containing ground truth answers.
            predicted_answers (List[str]): List of predicted answers.
            aggregation_fn (Callable): Function to aggregate scores across multiple gold answers (default: np.max).

        Returns:
            Tuple[Dict[str, float], List[Dict[str, float]]]:
                - A dictionary with the averaged F1 score.
                - A list of dictionaries with F1 scores for each example.
        """
        assert len(self.data.gold_answers) == len(
            predicted_answers), "Length of gold answers and predicted answers should be the same."

        def compute_f1(gold: str, predicted: str) -> float:
            gold_tokens = normalize_answer(gold).split()
            predicted_tokens = normalize_answer(predicted).split()
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                return 0.0

            precision = 1.0 * num_same / len(predicted_tokens)
            recall = 1.0 * num_same / len(gold_tokens)
            return 2 * (precision * recall) / (precision + recall)

        example_eval_results = []
        total_f1 = 0.0

        for gold_list, predicted in zip(self.data.gold_answers, predicted_answers):
            f1_scores = [compute_f1(gold, predicted) for gold in gold_list]
            aggregated_f1 = aggregation_fn(f1_scores)
            example_eval_results.append({"F1": aggregated_f1})
            total_f1 += aggregated_f1

        avg_f1 = total_f1 / len(self.data.gold_answers) if self.data.gold_answers else 0.0
        pooled_eval_results = {"F1": avg_f1}

        return pooled_eval_results

    def qa_rouge(self, predicted_answers: List[str], aggregation_fn: Callable = np.max):

        assert len(self.data.gold_answers) == len(
            predicted_answers), "Length of gold answers and predicted answers should be the same."
        
        def get_rouge_type(name: str) -> Tuple[str, str]:
            name = name.split("-", maxsplit=2)
            
            return {
                "L": "rougeL",
                "1": "rouge1",
                "2": "rouge2",
            }[name[1]], {
                "P": "precision",
                "R": "recall",
                "F": "fmeasure",
            }[name[2]]

        example_eval_results = []
        total_rouge = {
            "ROUGE-L-P": 0.0,
            "ROUGE-L-R": 0.0,
            "ROUGE-L-F": 0.0,
            "ROUGE-1-P": 0.0,
            "ROUGE-1-R": 0.0,
            "ROUGE-1-F": 0.0,
            "ROUGE-2-P": 0.0,
            "ROUGE-2-R": 0.0,
            "ROUGE-2-F": 0.0,
        }
        
        for gold_list, predicted in zip(self.data.gold_answers, predicted_answers):
            scorer = rouge_scorer.RougeScorer(['rougeL', 'rouge1', 'rouge2'], use_stemmer=True)
            rouge_scores = [scorer.score(prediction=predicted, target=gold) for gold in gold_list]

            for metric in total_rouge.keys():
                rouge_type = get_rouge_type(metric)
                aggregated_rouge = aggregation_fn([getattr(s[rouge_type[0]], rouge_type[1]) for s in rouge_scores])
                example_eval_results.append({metric: aggregated_rouge})
                total_rouge[metric] += aggregated_rouge

        pooled_eval_results = {}
        for metric, value in total_rouge.items():
            pooled_eval_results[metric] = value / len(self.data.gold_answers) if self.data.gold_answers else 0.0

        return pooled_eval_results
        
    def qa_answer_rate(self, predicted_answers: List[str]):
        
        total_answer_rate = {
            "false_rate": 0.0,
            "not_mentioned_rate": 0.0,
            "error_rate": 0.0,
        }

        for gold_list, predicted in zip(self.data.gold_answers, predicted_answers):
            if predicted == "Error":
                total_answer_rate["error_rate"] += 1
                continue

            predicted_tokens = normalize_answer(predicted).split()
            if any([{"not", "mentioned"}.issubset(predicted_tokens), 
                    {"not", "specified"}.issubset(predicted_tokens),
                    {"not", "stated"}.issubset(predicted_tokens)]):
                total_answer_rate["not_mentioned_rate"] += 1
                continue

            gold_tokens = set()
            for gold in gold_list:    
                gold_tokens |= set(normalize_answer(gold).split())
            
            common = Counter(predicted_tokens) & Counter(gold_tokens)
            num_same = sum(common.values())

            if num_same == 0:
                total_answer_rate["false_rate"] += 1

        pooled_eval_results = {}
        for metric, value in total_answer_rate.items():
            pooled_eval_results[metric] = value / len(self.data.gold_answers) if self.data.gold_answers else 0.0

        return pooled_eval_results

    def rt_recall(self, retrieved_docs: List[List[str]]) -> Tuple[Dict[str, float], List[Dict[str, float]]]:
        
        self.rc_k_list = sorted(set(self.rc_k_list))
            
        example_eval_results = []
        pooled_eval_results = {f"Recall@{k}": 0.0 for k in self.rc_k_list}
        for example_gold_docs, example_retrieved_docs in zip(self.data.gold_docs, retrieved_docs):
            if len(example_retrieved_docs) < self.rc_k_list[-1]:
                tqdm.write(f"Warning: Length of retrieved docs ({len(example_retrieved_docs)}) is smaller than largest topk for recall score ({self.rc_k_list[-1]})")
            
            example_eval_result = {f"Recall@{k}": 0.0 for k in self.rc_k_list}

            # Compute Recall@k for each k
            for k in self.rc_k_list:
                # Get top-k retrieved documents
                top_k_docs = example_retrieved_docs[:k]
                # Calculate intersection with gold documents
                relevant_retrieved = set(top_k_docs) & set(example_gold_docs)
                # Compute recall
                if example_gold_docs:  # Avoid division by zero
                    example_eval_result[f"Recall@{k}"] = len(relevant_retrieved) / len(set(example_gold_docs))
                else:
                    example_eval_result[f"Recall@{k}"] = 0.0
            
            # Append example results
            example_eval_results.append(example_eval_result)
            
            # Accumulate pooled results
            for k in self.rc_k_list:
                pooled_eval_results[f"Recall@{k}"] += example_eval_result[f"Recall@{k}"]

        # Average pooled results over all examples
        for k in self.rc_k_list:
            pooled_eval_results[f"Recall@{k}"] /= len(self.data.gold_docs)

        # round off to 4 decimal places for pooled results
        pooled_eval_results = {k: round(v, 4) for k, v in pooled_eval_results.items()}
        return pooled_eval_results
        
    def evaluate(self, answers: List[str] = None, retrieved_docs: List[List[str]] = None, 
                 metrics: str | Tuple[str] = "all") -> Dict:

        implemented_metrics = set(["em", "f1", "rouge", "rsim", "recall", "answerrate", "llmjudge"])
        if metrics == "all":
            metrics = implemented_metrics
        elif isinstance(metrics, List):
            metrics = set(metrics)
            assert metrics.issubset(implemented_metrics), "Some evaluation metrics are not supported."
        else:
            raise ValueError(f"Invalid evaluation metric(s) \"{metrics}\".")

        overall_results = {}
        
        if answers is not None and self.data.gold_answers is not None:
            if "em" in metrics:
                try:
                    overall_results.update(self.qa_exactmatch(
                        predicted_answers=answers,
                        aggregation_fn=np.max
                    ))
                except Exception as e:
                    print(e)
            if "f1" in metrics:
                try:
                    overall_results.update(self.qa_f1(
                        predicted_answers=answers,
                        aggregation_fn=np.max
                    ))
                except Exception as e:
                    print(e)
            if "rouge" in metrics:
                try:
                    overall_results.update(self.qa_rouge(
                        predicted_answers=answers,
                        aggregation_fn=np.max
                    ))
                except Exception as e:
                    print(e)
            if "answerrate" in metrics:
                try:
                    overall_results.update(self.qa_answer_rate(
                        predicted_answers=answers
                    ))
                except Exception as e:
                    print(e)
            # round off to 4 decimal places for QA results
            overall_results = {k: round(float(v), 4) for k, v in overall_results.items()}
        if retrieved_docs is not None and self.data.gold_docs is not None:
            if "recall" in metrics:
                try:
                    overall_results.update(self.rt_recall(
                        retrieved_docs=retrieved_docs
                    ))
                except Exception as e:
                    print(e)
            
        return overall_results
