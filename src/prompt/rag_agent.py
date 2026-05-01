one_shot_rag_docs_1 = ("""An in-context example is provided below. \n"""
    """=================ONE-SHOT IN-CONTEXT EXAMPLE=================\n"""
    """Retrieved documents: \n\n"""
    """Hotel Tallcorn\nThe Hotel Tallcorn is located in Marshalltown, Iowa. Today it is called the Tallcorn Towers Apartments. Built in 1928 by the Eppley Hotel Company, local citizens contributed $120,000 to ensure the successful completion of this seven-story hotel. It was completed in connection to the seventy-fifth anniversary of Marshalltown. The hotel's sale in 1956 from the Eppley chain to the Sheraton Corporation was part of the second largest hotel sale in United States history. The Tallcorn was listed as a contributing property in the Marshalltown Downtown Historic District on the National Register of Historic Places in 2002.\n"""
    """Hotel Bond\nHotel Bond is a historic hotel, built in two stages in 1913 and 1921, in downtown Hartford, Connecticut by hotelier Harry S. Bond. It is located near Bushnell Park, and was considered the grandest hotel in Hartford during its heyday. The second section is a 12 story building attached to the 6 story first section. A Statler Hotel opened in the area in 1954, creating competition, and the Bond Hotel company declared bankruptcy shortly after that. It was bought by the California-based Masaglia Hotel chain, which began an incremental renovation program. In 1964 it was sold to a Cincinnati, Ohio investment group which announced extensive renovation plans. However, the financing plans fell through and the hotel was again in bankruptcy. The building was sold at auction to the Roman Catholic Archdiocese of Hartford in 1965, and it became the home of the Saint Francis Hospital School of Nursing. The Bond Ballroom reopened in 2001, with the rest of the building becoming a Homewood Suites by Hilton in 2006.\n"""
    """Ritz-Carlton Jakarta\nThe Ritz-Carlton Jakarta is a hotel and skyscraper in Jakarta, Indonesia and 14th Tallest building in Jakarta. It is located in city center of Jakarta, near Mega Kuningan, adjacent to the sister JW Marriott Hotel. It is operated by The Ritz-Carlton Hotel Company. The complex has two towers that comprises a hotel and the Airlangga Apartment respectively. The hotel was opened in 2005. \n"""
)

one_shot_rag_docs_2 = (
    """Retrieved documents: \n\n"""
    """The Oberoi Group\nThe Oberoi Group is a hotel company with its head office in Delhi. Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands.\n"""
    """Mohan Singh Oberoi\nRai Bahadur Mohan Singh Oberoi (15 August 1898 – 3 May 2002) was an Indian hotelier, the founder and chairman of Oberoi Hotels & Resorts, India's second-largest hotel company, with 35 hotels in India, Sri Lanka, Nepal, Egypt, Australia and Hungary.\n"""
)

one_shot_rag_input = (
    '\nUser question: The Oberoi family is part of a hotel company that has a head office in what city?\nRetrieval time remaining: {current_retrieval_time}'
    '\nThought: '
)

one_shot_rag_output_1 = (
    'According to the user question, I need to first figure out the hotel company that the Oberoi family is part of. However, the retrieved documents do not include any details regarding the Oberoi family. \n'
    'Retrieve: Which hotel company is the Oberoi family part of?'
)

one_shot_rag_output_2 = (
    'According to "Mohan Singh Oberoi", Mohan Singh Oberoi is the founder and chairman of Oberoi Hotels & Resorts. '
    'From "The Oberoi Group", Oberoi Hotels & Resorts is a brand and its company, The Oberoi Group, is a hotel company with its head office in Delhi. \n'
    'Answer: Delhi'
    """\n=================ONE-SHOT IN-CONTEXT EXAMPLE END=================\n"""
)


answer_length_instruction = {
    "short": "Your <answer> should be in most cases a word or a phrase. ",
    "medium": "Your <answer> should be in most cases a narrative sentence or simply a word / phrase according to the user question. ",
    "long": "Your <answer> should be a comprehensive and coherent paragraph, enabling the user to comprehend the content or plot of the original corpus. ", # unused
    "auto": "",
}
rag_agent_system = (
    'As an advanced reading comprehension and information retrieval agent, your task is to answer the user question by retrieving documents and multi-step reasoning, following the instructions below: \n'
    '(1) Your response starts after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions step-by-step. '
    'If you believe you have found the answer, conclude with "Answer: <answer>" to present a concise, definitive response. \n'
    '(2) You can request for retrieving supporting documents from a relevant database a maximum of {max_retrieval_time} times. Some retrieved documents are provided the first time. '
    'When the question involves content you do not understand, or current documents do not provide enough information, request retrieval by responding with "Retrieve: <retrieve_query>", '
    'where <retrieve_query> is a sub-question derived from the user question, which you believe must be resolved to answer the user question. \n'
    '(3) The retrieval results will be provided in the format "Retrieved documents: ...", and you should continue your reasoning based on the user question, your previous reasoning steps, and all retrieved documents. '
    'Obtain your answer with as few retrieval attempts as possible, which means you need to make your <retrieve_query> as precise as possible. If newly retrieved documents do not help, try reorganizing your <retrieve_query>. '
    'If you run out of retrieval attempts and still cannot answer the question, answer with "Not mentioned" only. Do NOT provide additional uncertain information. \n' 
    '(4) NO UNNECESSARY WORDS: After your reasoning, you MUST end your response with "Answer: <answer>" or "Retrieve: <retrieve_query>" in a new line. {answer_length_instruction}Avoid ambiguous words like "approximately". \n'
    '(5) Your "Thought: " response should be NO MORE THAN {thought_max_length} WORDS. \n'
)
rag_agent_system_chat = (
    'As an advanced reading comprehension and information retrieval agent, your task is to answer the user question by retrieving documents and multi-step reasoning, following the instructions below: \n'
    '(1) Your response starts after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions step-by-step. '
    'If you believe you have found the answer, conclude with "Answer: <answer>" to present a concise, definitive response. \n'
    '(2) You can request for retrieving supporting documents from a relevant database a maximum of {max_retrieval_time} times. Some retrieved documents are provided the first time. '
    'When the question involves content you do not understand, or current documents do not provide enough information, request retrieval by responding with "Retrieve: <retrieve_query>", '
    'where <retrieve_query> is a sub-question derived from the user question, which you believe must be resolved to answer the user question. \n'
    '(3) The retrieval results will be provided in the format "Retrieved documents: ...", and you should continue your reasoning based on the user question, your previous reasoning steps, and all retrieved documents. '
    'Obtain your answer with as few retrieval attempts as possible, which means you need to make your <retrieve_query> as informative as possible. '
    'If newly retrieved documents do not help, try reorganizing your <retrieve_query>. Be honest and admit your ambiguity; do NOT provide uncertain information. \n' 
    '(4) NO UNNECESSARY WORDS: After your reasoning, you MUST end your response with "Answer: <answer>" or "Retrieve: <retrieve_query>" in a new line. {answer_length_instruction}Avoid ambiguous words like "approximately". \n'
    '(5) Your "Thought: " response should be NO MORE THAN {thought_max_length} WORDS. \n'
)

class AgentPrompt:
    def __init__(self, 
                 ans_type="short",
                 one_shot_in_context=True,
                 max_retrieval_time=3, 
                 thought_max_length=500) -> None:
        
        self.ans_type = ans_type
        self.current_retrieval_time = max_retrieval_time
        self.max_retrieval_time = max_retrieval_time
        self.thought_max_length = thought_max_length
        self.one_shot_in_context = one_shot_in_context
        self.answer_required_suffix = ', you MUST end your response with "Answer: <answer>"!'

        self.template = lambda d, q, t: (
            f"Retrieved documents: \n\n{d}\nUser question: {q}\n"
            f"Retrieval time remaining: {t}"
            f"{self.answer_required_suffix if t == 0 else ''}\n"
            "Thought: "
        )
        
        self.current_template = None

    def get_template(self, query, documents, answer=None):
        
        if self.current_template is None:
            if self.ans_type == "auto":
                self.current_template = [{"role": "system", "content": rag_agent_system_chat.format(max_retrieval_time=self.max_retrieval_time, 
                                                                                                    thought_max_length=self.thought_max_length,
                                                                                                    answer_length_instruction=answer_length_instruction[self.ans_type])}]
            else:
                self.current_template = [{"role": "system", "content": rag_agent_system.format(max_retrieval_time=self.max_retrieval_time, 
                                                                                               thought_max_length=self.thought_max_length,
                                                                                               answer_length_instruction=answer_length_instruction[self.ans_type])}]
            if self.one_shot_in_context:
                self.current_template.extend(
                    [{"role": "user", "content": one_shot_rag_docs_1 + one_shot_rag_input.format(current_retrieval_time=self.max_retrieval_time)},
                        {"role": "assistant", "content": one_shot_rag_output_1},
                        {"role": "user", "content": one_shot_rag_docs_2 + one_shot_rag_input.format(current_retrieval_time=self.max_retrieval_time - 1) + 
                                                    f"{self.answer_required_suffix if self.max_retrieval_time == 1 else ''}"},
                        {"role": "assistant", "content": one_shot_rag_output_2},]
                )
            self.current_template.append(
                {"role": "user", "content": self.template(documents, query, self.current_retrieval_time)}
            )

        elif answer is not None:
            self.current_retrieval_time -= 1
            self.current_template.append({"role": "assistant", "content": answer})
            self.current_template.append({"role": "user", "content": self.template(documents, query, self.current_retrieval_time)})
        
        else:
            raise ValueError("The assistant did not give an answer to the former question. ")

        return self.current_template
