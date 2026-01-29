one_shot_rag_docs = ("""=================ONE-SHOT IN-CONTEXT EXAMPLE=================\n"""
    """Hotel Tallcorn\nThe Hotel Tallcorn is located in Marshalltown, Iowa. Today it is called the Tallcorn Towers Apartments. Built in 1928 by the Eppley Hotel Company, local citizens contributed $120,000 to ensure the successful completion of this seven-story hotel. It was completed in connection to the seventy-fifth anniversary of Marshalltown. The hotel's sale in 1956 from the Eppley chain to the Sheraton Corporation was part of the second largest hotel sale in United States history. The Tallcorn was listed as a contributing property in the Marshalltown Downtown Historic District on the National Register of Historic Places in 2002.\n"""
    """The Oberoi Group\nThe Oberoi Group is a hotel company with its head office in Delhi. Founded in 1934, the company owns and/or operates 30+ luxury hotels and two river cruise ships in six countries, primarily under its Oberoi Hotels & Resorts and Trident Hotels brands.\n"""
    """Hotel Bond\nHotel Bond is a historic hotel, built in two stages in 1913 and 1921, in downtown Hartford, Connecticut by hotelier Harry S. Bond. It is located near Bushnell Park, and was considered the grandest hotel in Hartford during its heyday. The second section is a 12 story building attached to the 6 story first section. A Statler Hotel opened in the area in 1954, creating competition, and the Bond Hotel company declared bankruptcy shortly after that. It was bought by the California-based Masaglia Hotel chain, which began an incremental renovation program. In 1964 it was sold to a Cincinnati, Ohio investment group which announced extensive renovation plans. However, the financing plans fell through and the hotel was again in bankruptcy. The building was sold at auction to the Roman Catholic Archdiocese of Hartford in 1965, and it became the home of the Saint Francis Hospital School of Nursing. The Bond Ballroom reopened in 2001, with the rest of the building becoming a Homewood Suites by Hilton in 2006.\n"""
    """Mohan Singh Oberoi\nRai Bahadur Mohan Singh Oberoi (15 August 1898 – 3 May 2002) was an Indian hotelier, the founder and chairman of Oberoi Hotels & Resorts, India's second-largest hotel company, with 35 hotels in India, Sri Lanka, Nepal, Egypt, Australia and Hungary.\n"""
    """Ritz-Carlton Jakarta\nThe Ritz-Carlton Jakarta is a hotel and skyscraper in Jakarta, Indonesia and 14th Tallest building in Jakarta. It is located in city center of Jakarta, near Mega Kuningan, adjacent to the sister JW Marriott Hotel. It is operated by The Ritz-Carlton Hotel Company. The complex has two towers that comprises a hotel and the Airlangga Apartment respectively. The hotel was opened in 2005."""
)

one_shot_abs_output = (
    """The Hotel Tallcorn in Marshalltown, Iowa, built in 1928, is now an apartment building. It was part of one of the 
second-largest hotel sales in U.S. history in 1956. Hotel Bond is a historic hotel in Hartford, Connecticut, built in two stages. 
After facing competition, it went through multiple ownership changes and bankruptcies. It is now a Homewood Suites by Hilton, 
with its ballroom having reopened in 2001. The Oberoi Group, founded in 1934, is a luxury hotel company based in Delhi. 
It owns and operates over 30 hotels and two cruise ships. It was founded by Mohan Singh Oberoi, who started India's second-largest 
hotel company. The Ritz-Carlton Jakarta is a skyscraper hotel in Indonesia, opened in 2005. It consists of two towers, 
one for the hotel and one for apartments, and is located near the JW Marriott Hotel.
\n=================ONE-SHOT IN-CONTEXT EXAMPLE END================="""
)

one_shot_kw_output = (
    """historic hotels,hotel companies,bankruptcy and renovation,brand operations,Indian hospitality industry,Indonesia skyscraper,
hotel locations,hotel building time,hotel architecture,hotel founders,hotel ownership history,Marshalltown Iowa,Hotel Tallcorn,
Eppley Hotel Company,Sheraton sale,The Oberoi Group,Mohan Singh Oberoi,Hotel Bond,Hartford Connecticut,Bond Ballroom,Ritz-Carlton Jakarta
\n=================ONE-SHOT IN-CONTEXT EXAMPLE END================="""
)

abs_system = lambda abs_max_length: (
    "As an advanced document summarization assistant, your task is to summarize the given list of documents, including as many key details and relationships as possible. \n"
    "(1) Your answer should always start with \"Summary:\". NO UNNECESSARY WORDS: Always directly output the summary instead of useless modifiers like 'Here is a summary of the given list ...'. \n"
    "(2) Be faithful to the original documents. Unnecessary paraphrasing and adding content not mentioned in the original documents are NOT allowed. \n"
    f"(3) Your summary should be NO MORE THAN {abs_max_length} WORDS. Keep in mind that a longer summary is not always better. \n"
)

abs_system_kw_leaf = lambda max_num_keywords: (
    "As an advanced document summarization assistant, your task is to summarize the given list of text paragraphs "
    "into multiple key phrases or words. "
    "These key phrases include: \n"
    "·Key phrases that overview the types of information covered in the text; \n"
    "·Key phrases that reflect the common themes of these paragraphs; \n"
    "·Key phrases that distinguish the specific content of each paragraph. \n"
    "\nRequirements: \n"
    "·Generate key phrases mentioned above in order. "
    "All key phrases should be strictly formatted as ONE line of text, separated by commas, "
        "equivalent to the output format of Python code ``','.join(list_of_keywords)``. \n"
    "·Different key phrases should be distinct from one another. \n"
    "·NO UNNECESSARY WORDS: Always output the direct key phrase list instead of useless modifiers like "
        "'Here are the key phrases according to the documents: ...' or 'Information key phrases: ... Theme key phrases: ...'. \n"
    f"·The number of key phrases should be NO MORE THAN {max_num_keywords}. "
)

abs_system_kw_branch = lambda max_num_keywords: (
    "As an advanced document summarization assistant, your task is to handpick or summarize the given list of keywords or phrases. "
    "These key phrases include: \n"
    "·Key phrases that overview the types of information covered in the text; \n"
    "·Key phrases that reflect the common themes of these paragraphs; \n"
    "·Key phrases that distinguish the specific content of each paragraph. \n"
    "\nRequirements: \n"
    "·Generate key phrases mentioned above in order. "
    "All key phrases should be strictly formatted as ONE line of text, separated by commas, "
        "equivalent to the output format of Python code ``','.join(list_of_keywords)``. \n"
    "·Different key phrases should be distinct from one another. \n"
    "·NO UNNECESSARY WORDS: Always output the direct key phrase list instead of useless modifiers like "
        "'Here are the key phrases according to the documents: ...' or 'Information key phrases: ... Theme key phrases: ...'. \n"
    f"·The number of key phrases should be NO MORE THAN {max_num_keywords}. "
)

def get_abs_template(documents, keyword=True, leaf=True, abs_max_length=20):

    if not keyword:
        abs_system_prompt = abs_system(abs_max_length=abs_max_length)
        return [
            {
                "role": "system", 
                "content": abs_system_prompt,
            }, {
                "role": "user",
                "content": one_shot_rag_docs,
            }, {
                "role": "assistant",
                "content": one_shot_abs_output,
            }, {
                "role": "user",
                "content": documents,
            }, 
        ]
    
    if leaf:
        abs_system_prompt = abs_system_kw_leaf(max_num_keywords=abs_max_length)
    else:
        abs_system_prompt = abs_system_kw_branch(max_num_keywords=abs_max_length)
    
    return [
        {
            "role": "system", 
            "content": abs_system_prompt,
        }, {
            "role": "user",
            "content": one_shot_rag_docs,
        }, {
            "role": "assistant",
            "content": one_shot_kw_output,
        }, {
            "role": "user",
            "content": documents,
        }, 
    ]