import os
import time
import argparse
from typing import List
from tqdm import tqdm
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

from conf import read_config
from src import (RAG,
                 TransformersEmbeddingModel,
                 SentenceTransformersEmbeddingModel,
                 OpenAIQAModel,
                 OpenAIAbstractModel,
                 TransformersAbstractModel,
                 OllamaEmbeddingModel,
                 OllamaAbstractModel,
                 OllamaQAModel,
                 TransformersQAModel,
                 TransformersRerankModel,
                 DataManager,
                 Evaluator)
from src.prompt import get_qa_template, AgentPrompt
from src.utils import parse_response, save_answers, load_answers


def main():
    # ============ 1) preparation (datasets, model, config, evaluator) ======
    data = DataManager(dataset_name=conf["dataset"],
                       data_dir=conf["data_dir"],
                       test_samples=conf["test_samples"]
                      )
    
    # split text
    if isinstance(data.all_passages, str):
        conf["passage_as_tree"] = True
        conf["force_split"] = True
        data.split_text(tokenizer=conf["tokenizer"], max_tokens=conf["max_tokens_per_chunk"])
    elif isinstance(data.all_passages, List) and isinstance(data.all_passages[0], str): 
        if conf["passage_as_tree"] or conf["force_split"]:
            conf["force_split"] = True
            data.split_text(tokenizer=conf["tokenizer"], max_tokens=conf["max_tokens_per_chunk"])
    elif isinstance(data.all_passages[0], List):
        if conf["force_split"]:
            data.split_text(tokenizer=conf["tokenizer"], max_tokens=conf["max_tokens_per_chunk"])
    
    top_k = min(conf["tree_top_k"], conf["rerank_top_k"]) if conf["rerank_top_k"] is not None else conf["tree_top_k"]
    evaluator = Evaluator(data=data, top_k_nodes_per_layer=top_k)

    if not os.path.exists(conf["log_path"]):
        os.makedirs(conf["log_path"])
    logger = open(os.path.join(conf["log_path"], f'{conf["config"]}.log'), "a")

    # load answer file to skip running and evaluate
    results = None
    if conf["save_dir"] is not None:
        results = load_answers(conf)
    
    if not results:
        # run RAG
        def set_model(model_name, task_type):
            framework, model_name = model_name.split(sep=":", maxsplit=1)
            model_class = {
                "ollama": {
                    "embed": OllamaEmbeddingModel,
                    "abs": OllamaAbstractModel,
                    "qa": OllamaQAModel,
                },
                "transformers": {
                    "embed": TransformersEmbeddingModel,
                    "abs": TransformersAbstractModel,
                    "qa": TransformersQAModel,
                    "rerank": TransformersRerankModel,
                },
                "sentence-transformers": {
                    "embed": SentenceTransformersEmbeddingModel,
                },
                "api": {
                    "abs": OpenAIAbstractModel,
                    "qa": OpenAIQAModel,
                },
            }[framework][task_type]

            return model_class(model_name, cache_dir=conf.get(f"{task_type}_cache_dir", None))

        models_to_prepare = [model_type for model_type in conf.keys() 
                             if model_type.endswith("_name") and conf[model_type] is not None]
        for model_type in models_to_prepare:
            task_type = model_type.rsplit("_name", maxsplit=1)[0]
            conf[f"{task_type}_model"] = set_model(conf[model_type], task_type)

        TreeRAG = RAG(conf)

        # ===================== 2) construct tree index =====================
        if conf["save_dir"] is not None:
            os.makedirs(conf["save_dir"], exist_ok=True)
            save_tree_file = os.path.join(conf["save_dir"], 
                                        f'{conf["dataset"]}_{conf["embed_name"].replace("/", "_")}'
                                        f'_{conf["abs_name"].replace("/", "_")}_{conf["abstract_type"]}_tree.pkl')
            if conf["force_index_from_scratch"] and os.path.exists(save_tree_file):
                os.remove(save_tree_file)
            if os.path.exists(save_tree_file):
                TreeRAG.load("tree", save_tree_file)
                tqdm.write(f"Loaded tree from pickle file \"{save_tree_file}\".")
            else:
                TreeRAG.add_documents(data)
                TreeRAG.save("tree", save_tree_file)
                tqdm.write(f"Saved tree to pickle file \"{save_tree_file}\".") 
        else:
            TreeRAG.add_documents(data)
        
        if conf["hybrid_search"]:
            TreeRAG.build_vocab(data)

        # =================== 3) retrieve & question answering ===================

        all_qa_time = []
        all_answers = {}
        all_contexts = {}

        def qa(query, query_id, tokenizer_lock=None):
            
            start_time = time.time()
            documents, layer_information = TreeRAG.retrieve(query, data.query_to_doc_ids[query_id], tokenizer_lock=tokenizer_lock)
            
            def get_agentic_answer(query, documents, state_log, response=None):
                documents = '\n'.join(documents)
                message = agent_prompt.get_template(query, documents, answer=response)
                response = TreeRAG.qa(question=message, max_tokens=2 * conf["max_response_length"])
                thought, action, info = parse_response(response, verbose=conf["verbose"])
                state_log["thought"].append(thought)
                if action == "answer":
                    return info
                elif action == "retrieve":
                    state_log["subquestion"].append(info)
                    documents, layer_information = TreeRAG.retrieve(info, data.query_to_doc_ids[query_id], tokenizer_lock=tokenizer_lock)
                    state_log["retrieved_nodes"].append(layer_information)
                    return get_agentic_answer(query, documents, state_log, response)
                else:
                    tqdm.write(f"Unexpected agent action: {action}")
                    return "Error"

            if conf["max_retrieval_time"] > 0:
                # iterative retrieval & QA
                state_log = {
                    "subquestion": [query],
                    "retrieved_nodes": [layer_information],
                    "thought": []
                }
                agent_prompt = AgentPrompt(ans_type=conf["answer_type"],
                    max_retrieval_time=conf["max_retrieval_time"],
                    thought_max_length=conf["max_response_length"],
                )
                answer = get_agentic_answer(query, documents, state_log)
                qa_time = time.time() - start_time

                top_k_scores = {} # node['node_index']: node['score']
                for layer_information in state_log["retrieved_nodes"]:
                    for node in layer_information:
                        if node['layer_number'] == 0:
                            if node['node_index'] in top_k_scores.keys() and node['score'] > top_k_scores[node['node_index']]:
                                top_k_scores[node['node_index']] = node['score']
                            else:
                                top_k_scores.setdefault(node['node_index'], node['score'])
                top_k_scores = dict(sorted(top_k_scores.items(), key=lambda x: x[1], reverse=True)[:top_k])
                if isinstance(TreeRAG.tree, List):
                    context = [TreeRAG.tree[data.query_to_doc_ids[query_id]].all_nodes[top_k_node_index].text for top_k_node_index in top_k_scores.keys()]
                else:
                    context = [TreeRAG.tree.all_nodes[top_k_node_index].text for top_k_node_index in top_k_scores.keys()]

                output = "\n".join([f"\nid: {query_id}",
                                    f"question: {query}",
                                    f'sub-questions: {"\n\t".join(state_log["subquestion"][1:])}', 
                                    f'thoughts: {"\n\t".join(state_log["thought"])}',
                                    f"answer: {answer}",
                                    f"gold answer: {data.gold_answers[query_id] if data.gold_answers is not None else 'NA'}",
                                    "\n"])
                
            else:
                # single retrieval and QA
                documents = '\n'.join(documents)
                qa_message = get_qa_template(query, documents, type=conf["answer_type"], 
                                             add_abstract_to_context=conf["abstract_layer_as_context"] > 0,
                                             thought_max_length=conf["max_response_length"])
                raw_answer = TreeRAG.qa(question=qa_message, max_tokens=2 * conf["max_response_length"])

                try:
                    thought, answer = raw_answer.rsplit('Answer:', maxsplit=1)
                    answer = answer.strip(' .')
                except (IndexError, ValueError):
                    thought = ''
                    answer = raw_answer

                qa_time = time.time() - start_time

                top_k_scores = {}
                for node in layer_information:
                    if node['layer_number'] == 0:
                        top_k_scores[node['node_index']] = node['score']
                top_k_scores = dict(sorted(top_k_scores.items(), key=lambda x: x[1], reverse=True)[:top_k])
                if isinstance(TreeRAG.tree, List):
                    context = [TreeRAG.tree[data.query_to_doc_ids[query_id]].all_nodes[top_k_node_index].text for top_k_node_index in top_k_scores.keys()]
                else:
                    context = [TreeRAG.tree.all_nodes[top_k_node_index].text for top_k_node_index in top_k_scores.keys()]
                # if only <k documents are retrieved (may caused by deduplication) then fill with empty docs
                if len(context) < top_k:
                    for _ in range(top_k - len(context)):
                        context.append('')
                
                output = "\n".join([f"\nid: {query_id}",
                                    f"question: {query}",
                                    f"thoughts: {thought}",
                                    f"answer: {answer}",
                                    f"gold answer: {data.gold_answers[query_id] if data.gold_answers is not None else 'NA'}",
                                    "\n"])

            if conf["verbose"]:
                tqdm.write(output)
            logger.write(output)

            return query_id, answer, context, qa_time
        
        # QA with multithreading
        if conf["multithreading_qa_batch_size"] > 1:
            tokenizer_lock = Lock()
            # preload the model in case of multiple loading by threads
            if hasattr(conf["embed_model"], "load_model"):
                conf["embed_model"].load_model()
            if conf["rerank"] and conf["rerank_model"] is not None and hasattr(conf["rerank_model"], "load_model"):
                conf["rerank_model"].load_model()

            bar = tqdm(range(0, len(data.all_queries), conf["multithreading_qa_batch_size"]), desc="qa")
            for i in bar:
                with ThreadPoolExecutor() as executor:
                    future_qa_results = [
                        executor.submit(qa, query, query_id, tokenizer_lock)
                        for query_id, query in enumerate(data.all_queries[i : i + conf["multithreading_qa_batch_size"]], i)
                    ]
                    for future in as_completed(future_qa_results):
                        query_id, answer, context, qa_time = future.result()
                        all_answers[query_id] = answer
                        all_contexts[query_id] = context
                        all_qa_time.append(qa_time)
        else:
            bar = tqdm(data.all_queries, desc="qa")
            for query_id, query in enumerate(bar):
                _, answer, context, qa_time = qa(query, query_id)
                all_answers[query_id] = answer
                all_contexts[query_id] = context
                all_qa_time.append(qa_time)

        bar.close()

        results = {
            "answers": [ans[1] for ans in sorted(all_answers.items())],
            "retrieved_docs": [cont[1] for cont in sorted(all_contexts.items())],
            "time": {
                "tb_time": TreeRAG.tb_time,
                "tr_time": TreeRAG.tr_time,
                "qa_time": sum(all_qa_time) / len(all_qa_time),
            },
        }
        if conf["save_dir"] is not None:
            save_answers(conf, results)
        
        retrieval_stats = (f"Total times of retrieval: {TreeRAG.retrieve_count}\n"
            f"Average tree retrieval time: {TreeRAG.time_dict['tree'] / TreeRAG.retrieve_count:.4f}s\n"
            f"Average sparse retrieval time: {TreeRAG.time_dict['sparse'] / TreeRAG.retrieve_count:.4f}s\n"
            f"Average rerank time: {TreeRAG.time_dict['rerank'] / TreeRAG.retrieve_count:.4f}s\n"
        )
        print(retrieval_stats)
        logger.writelines(retrieval_stats.splitlines())

    # ============================= 4) evaluation =============================
    scores = evaluator.evaluate(answers=results.get("answers", None),
                                retrieved_docs=results.get("retrieved_docs", None),
                                metrics=conf["evaluation_metrics"])
    
    if "time" in results:
        final_eval_output = (
            f"Evaluation results: {scores}\n"
            f'Tree building time: {"NA" if results["time"]["tb_time"] < 0 else f"{results['time']['tb_time']:.2f}s"}\n'
            f'Single retrieval time: {"NA" if results["time"]["tr_time"] < 0 else f"{results['time']['tb_time']:.2f}s"}\n'
            f'Average QA time: {"NA" if results["time"]["qa_time"] < 0 else f"{results['time']['tb_time']:.2f}s"}\n'
        )
    else:
        final_eval_output = (
            f"Evaluation results: {scores}\n"
        )
    print(final_eval_output)
    
    logger.writelines(final_eval_output.splitlines())
    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, metavar='CONFIG FILE NAME', 
                        help='Config file name (without ".py") under the "conf" directory.')
    args = parser.parse_args()

    if args.config is not None:
        conf = read_config(conf_name=args.config)
    else:
        conf = read_config()
    
    main()
