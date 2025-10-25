# ==================================================================
# ============================ PROMPTS =============================
# ==================================================================
query_analysis_prompt: str = """
<SYSTEM>
    <ROLE>
        You're an expert at determining whether a user query requires information from a vector store or a web search.
    </ROLE>
    <TOPICS>{topics}</TOPICS>

    <GUIDELINES>
    - If the query is related to the topics above, choose 'vectorstore'.
    - If the query is not covered by the topics above, choose 'websearch'.
    - Base your decision solely on the content of the query.
    </GUIDELINES>
</SYSTEM>
"""

retrieval_grading_prompt: str = """
<SYSTEM>
    <ROLE>
        You're an expert at determining whether the retrieved documents from a vector store is relevant to the user query.
    </ROLE>
    <VALID_OUTPUT>{valid_output}</VALID_OUTPUT>

    <GUIDELINES>
    - If the documents are relevant to the user query, choose 'yes'.
    - If the documents are not relevant to the user query, choose 'no'.
    </GUIDELINES>

</SYSTEM>
"""

query_n_retrieved_docs_prompt: str = """
<QUERY>{query}</QUERY>
<RETRIEVED_DOCUMENTS>{retrieved_documents}</RETRIEVED_DOCUMENTS>

Are the retrieved documents relevant to the user query?
"""

query_n_response_prompt: str = """
<QUERY>{query}</QUERY>
<RESPONSE>{response}</RESPONSE>

Is the response relevant to the user query?
"""

rag_response_generator_prompt: str = """
<ROLE>
    You're an expert at generating accurate and concise answers to user queries based on retrieved documents.
</ROLE>

    <QUERY>{query}</QUERY>
    <RETRIEVED_DOCUMENTS>{retrieved_documents}</RETRIEVED_DOCUMENTS>

    <GUIDELINES>
    - Limit your summary to a maximum of 5 sentences.
    - Use only the information provided in the retrieved documents.
    </GUIDELINES>
"""

hallucination_prompt: str = """
<SYSTEM>
    <ROLE>
        You're an expert at determining whether the generated response is accurate and relevant to the user query.
    </ROLE>
    <VALID_OUTPUT>{valid_output}</VALID_OUTPUT>

    <GUIDELINES>
    - If the response is NOT relevant to the user query, choose 'yes'.
    - If the response is relevant to the user query, choose 'no'.
    </GUIDELINES>

</SYSTEM>
"""

query_rewriter_prompt: str = """
<ROLE>
    You're an expert at rewriting user queries to improve vector search retrieval.
</ROLE>

<ORIGINAL_QUERY>{original_query}</ORIGINAL_QUERY>

<GUIDELINES>
- Rewrite the query to be more specific and clear.
- Ensure the rewritten query captures the user's intent accurately.
- There must be no preamble, just the single rewritten query.
</GUIDELINES>
"""

websearch_prompt: str = """
<SYSTEM>
    <ROLE>
        You are an expert assistant specialized in generating a concise summary of web search results.
    </ROLE>

    <GUIDELINES>
    - Summarize the search results accurately and concisely.
    - Limit your summary to a maximum of 5 sentences.
    </GUIDELINES>

</SYSTEM>
"""

vectorstore_routing_prompt = """
<INSTR>
    Analyze this query and determine which retriever to use.
    <QUERY>{query}</QUERY>
</INSTR>
    """
