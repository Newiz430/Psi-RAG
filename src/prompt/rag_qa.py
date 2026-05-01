one_shot_rag_qa_docs = ("""An in-context example is provided below. \n"""
    """=================ONE-SHOT IN-CONTEXT EXAMPLE=================\n"""
    """Retrieved documents: \n\n"""
    """Hotel Tallcorn\nThe Hotel Tallcorn is located in Marshalltown, Iowa. Today it is called the Tallcorn Towers Apartments. Built in 1928 by the Eppley Hotel Company, local citizens contributed $120,000 to ensure the successful completion of this seven-story hotel. It was completed in connection to the seventy-fifth anniversary of Marshalltown. The hotel's sale in 1956 from the Eppley chain to the Sheraton Corporation was part of the second largest hotel sale in United States history. The Tallcorn was listed as a contributing property in the Marshalltown Downtown Historic District on the National Register of Historic Places in 2002.\n"""
    """Hotel Bond\nHotel Bond is a historic hotel, built in two stages in 1913 and 1921, in downtown Hartford, Connecticut by hotelier Harry S. Bond. It is located near Bushnell Park, and was considered the grandest hotel in Hartford during its heyday. The second section is a 12 story building attached to the 6 story first section. A Statler Hotel opened in the area in 1954, creating competition, and the Bond Hotel company declared bankruptcy shortly after that. It was bought by the California-based Masaglia Hotel chain, which began an incremental renovation program. In 1964 it was sold to a Cincinnati, Ohio investment group which announced extensive renovation plans. However, the financing plans fell through and the hotel was again in bankruptcy. The building was sold at auction to the Roman Catholic Archdiocese of Hartford in 1965, and it became the home of the Saint Francis Hospital School of Nursing. The Bond Ballroom reopened in 2001, with the rest of the building becoming a Homewood Suites by Hilton in 2006.\n"""
    """Ritz-Carlton Jakarta\nThe Ritz-Carlton Jakarta is a hotel and skyscraper in Jakarta, Indonesia and 14th Tallest building in Jakarta. It is located in city center of Jakarta, near Mega Kuningan, adjacent to the sister JW Marriott Hotel. It is operated by The Ritz-Carlton Hotel Company. The complex has two towers that comprises a hotel and the Airlangga Apartment respectively. The hotel was opened in 2005. \n"""
)

no_retrieval_system = (
    'Your task is to answer the user question, following the instructions below: \n'
    '(1) Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. \n'
    '(2) Always conclude with "Answer: <answer>" to present a concise, definitive response. \n'
    '(3) NO UNNECESSARY WORDS: After your reasoning, you MUST end your response with "Answer: <answer>" in a new line. \n'
    "(4) If you cannot answer the question based on your knowledge, be honest and answer with \"I don't know\" only. Do NOT provide additional uncertain information. \n"
    '(5) Your "Thought: " response should be NO MORE THAN {thought_max_length} WORDS.\n'
)

rag_qa_system_short = (
    'As an advanced reading comprehension assistant, your task is to answer the user question based on retrieved documents, following the instructions below: \n'
    '(1) Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. \n'
    '(2) Always conclude with "Answer: <answer>" to present a concise, definitive response. \n'
    '(3) NO UNNECESSARY WORDS: After your reasoning, you MUST end your response with "Answer: <answer>" in a new line. <answer> is in most cases a word or a phrase. Avoid ambiguous words like "approximately". \n'
    '(4) If you cannot answer the question based on the retrieved documents, answer with "Not mentioned" only. Do NOT provide additional uncertain information. \n'
    '(5) Your "Thought: " response should be NO MORE THAN {thought_max_length} WORDS.\n'
)

rag_qa_system_medium = (
    'As an advanced reading comprehension assistant, your task is to answer the user question based on retrieved documents, following the instructions below: \n'
    '(1) Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. \n'
    '(2) Always conclude with "Answer: <answer>" to present a concise, definitive response. \n'
    '(3) NO UNNECESSARY WORDS: After your reasoning, you MUST end your response with "Answer: <answer>" in a new line. <answer> is in most cases a narrative sentence or simply a word / phrase according to the user question. Avoid ambiguous words like "approximately". \n'
    '(4) If you cannot answer the question based on the retrieved documents, answer with "Not mentioned" only. Do NOT provide additional uncertain information. \n'
    '(5) Your "Thought: " response should be NO MORE THAN {thought_max_length} WORDS.\n'
)

abs_ctx1 = ", including high-level summaries from different parts of the corpus"
abs_ctx2 = "·For each high-level summary, a [LAYER] is prepended to indicate their breadth. A higher [LAYER] summarizes more documents in a higher level. Note that summaries are not given in order. \n"
abs_ctx3 = "and summaries "
rag_qa_system_long = (
    'As an advanced reading comprehension and information retrieval agent, your task is to answer summative user questions based on retrieved documents{abs_ctx1}: \n'
    '·For each document, an [ID] is prepended to indicate the relative order in the document corpus. \n{abs_ctx2}'
    "These documents {abs_ctx3}provide a condensed version of the original long corpus to help you with summative tasks. "
    "\nRequirements: \n"
    '(1) Organize the documents {abs_ctx3}to form a logically clear summary of the main content or plot of the original long corpus. '
    '(2) Your answer should be comprehensive and coherent. Your summative answer should be an independent paragraph, enabling the user to comprehend the content or plot of the original corpus. '
    'Do NOT include additional uncertain or made-up information in your answer. \n'
    '(3) Your answer should be NO MORE THAN {response_max_length} WORDS. \n' 
)

rag_qa_system_auto = (
    'As an advanced reading comprehension assistant, your task is to answer the user question based on retrieved documents. \n'
    'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. Instructions: \n'
    '(1) Always conclude with "Answer: <answer>" to present a concise, definitive response. \n'
    '(2) NO UNNECESSARY WORDS: After your reasoning, you MUST end your response with "Answer: <answer>" in a new line. Avoid ambiguous words like "approximately". \n'
    '(3) If you cannot answer the question based on the retrieved documents, answer with "Not mentioned" only. Do NOT provide additional uncertain information. \n'
    '(4) Your "Thought: " response should be NO MORE THAN {response_max_length} WORDS.\n'
    'For summative tasks, you should organize the documents to form a logically clear summary of the main content or plot of the original long corpus. '
    'Your answer should be comprehensive and coherent. Your summative answer should be an independent paragraph, enabling the user to comprehend the content or plot of the original corpus. '
    'Do NOT include additional uncertain or made-up information in your answer. Your answer should also be NO MORE THAN {response_max_length} WORDS. \n'
)

one_shot_rag_qa_input_short = (
    "\nUser question: "
    "When was Neville A. Stanton's employer founded?"
    '\nThought: '
)

one_shot_rag_qa_input_medium = (
    "\nUser question: "
    ""
    '\nThought: '
)

one_shot_rag_qa_output_short = (
    'From "Neville A. Stanton", The employer of Neville A. Stanton is University of Southampton. According to "Southampton", the University of Southampton was founded in 1862. '
    "\nAnswer: 1862"
    """\n=================ONE-SHOT IN-CONTEXT EXAMPLE END=================\n"""
)


def get_qa_template(query, documents, type="short", thought_max_length=500, **kwargs):
    if documents is None or len(documents) == 0:
        return [
            {"role": "system", "content": no_retrieval_system.format(thought_max_length=thought_max_length)},
            {"role": "user", "content": (
                f"User question: {query}\n"
                "Thought: "
            )}
        ]
    if type == "auto":
        return [
            {"role": "system", "content": rag_qa_system_auto.format(response_max_length=thought_max_length)},
            {"role": "user", "content": one_shot_rag_qa_docs + one_shot_rag_qa_input_short}, 
            {"role": "assistant", "content": one_shot_rag_qa_input_short},
            {"role": "user", "content": (
                f"Retrieved documents: \n\n{documents}\n\nUser question: {query}\n"
                "Thought: "
            )}
        ]
    elif type == "medium":
        return [
            {"role": "system", "content": rag_qa_system_medium.format(thought_max_length=thought_max_length)},
            {"role": "user", "content": one_shot_rag_qa_docs + one_shot_rag_qa_input_short}, 
            {"role": "assistant", "content": one_shot_rag_qa_input_short},
            {"role": "user", "content": (
                f"Retrieved documents: \n\n{documents}\n\nUser question: {query}\n"
                "Thought: "
            )}
        ]
    elif type == "long":
        return [
            {"role": "system", "content": rag_qa_system_long.format(
                abs_ctx1=abs_ctx1 if kwargs.get("add_abstract_to_context", False) else "",
                abs_ctx2=abs_ctx2 if kwargs.get("add_abstract_to_context", False) else "",
                abs_ctx3=abs_ctx3 if kwargs.get("add_abstract_to_context", False) else "",
                response_max_length=int(thought_max_length * 0.7))},
            {"role": "user", "content": (
                f"Retrieved documents: \n\n{documents}\n\nUser question: {query}\n"
            )}
        ]
    else:
        return [
            {"role": "system", "content": rag_qa_system_short.format(thought_max_length=thought_max_length)},
            {"role": "user", "content": one_shot_rag_qa_docs + one_shot_rag_qa_input_short},
            {"role": "assistant", "content": one_shot_rag_qa_input_short},
            {"role": "user", "content": (
                f"Retrieved documents: \n\n{documents}\n\nUser question: {query}\n"
                "Thought: "
            )}
        ]
