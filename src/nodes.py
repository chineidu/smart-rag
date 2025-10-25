from typing import Any, Literal

from langchain_core.documents.base import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_qdrant import QdrantVectorStore

from prompts import (
    hallucination_prompt,
    query_analysis_prompt,
    query_n_response_prompt,
    query_n_retrieved_docs_prompt,
    query_rewriter_prompt,
    rag_response_generator_prompt,
    retrieval_grading_prompt,
    vectorstore_routing_prompt,
    websearch_prompt,
)
from schemas.input_schema import (
    GradeRetrievalSchema,
    HallucinationSchema,
    RouteQuerySchema,
    VectorSearchTypeSchema,
)
from schemas.types import DataSource, VectorSearchType, YesOrNo
from src import create_logger
from src.state import State
from src.utilities.llm_utils import classifier_model, remote_llm
from src.utilities.tools import search_tool
from src.utilities.utils import (
    convert_langchain_messages_to_dicts,
    get_structured_output,
)
from src.utilities.vectorstores import initialize_vectorstores

logger = create_logger("nodes")
# =================================================================
# ========================== GLOBAL VARS ==========================
# =================================================================
topics: list[str] = [_topic.value for _topic in VectorSearchType]
valid_output: list[str] = [_typ.value for _typ in YesOrNo]
vectorstore_ai: QdrantVectorStore
vectorstore_football: QdrantVectorStore

try:
    vectorstore_ai, vectorstore_football = initialize_vectorstores()
except Exception as e:
    logger.error(f"Failed to initialize vector stores: {e}", exc_info=True)
    raise


async def classify_query_node(state: State) -> dict[str, Any]:
    """Classify the user query to determine the data source to use."""
    logger.info("Calling ===> classify_query_node <===")

    query = state.get("query")
    sys_msg = SystemMessage(content=query_analysis_prompt.format(topics=topics))
    messages = convert_langchain_messages_to_dicts(
        [sys_msg, HumanMessage(content=query)]
    )
    query_type: RouteQuerySchema = await get_structured_output(
        messages=messages,
        model=classifier_model,
        schema=RouteQuerySchema,
    )  # type: ignore

    logger.info(
        f"✅ Classified query to use data source: {query_type.data_source.value}"
    )
    return {"other_info": {"source_type": query_type.data_source.value}}


async def llm_call_node(state: State) -> dict[str, Any]:
    """Call LLM with tools to get an initial response."""
    logger.info("Calling ===> llm_call_node <===")

    query = state.get("query")
    if not query and "messages" in state:
        messages = state["messages"]
        query = messages[-1] if isinstance(messages, list) else messages

    llm_with_tools = remote_llm.bind_tools([search_tool])
    response = await llm_with_tools.ainvoke(query)

    return {
        "query": query,
        # Messages key is the default key for tools
        "messages": [response],
    }


async def generate_web_search_response(state: State) -> dict[str, Any]:
    """Generate response based on web search results."""
    logger.info("Calling ===> generate_web_search_response <===")

    message: str = state.get("messages", [])[-1].content  # type: ignore
    if not message:
        return {
            "response": "I couldn't find relevant information to answer your query."
        }
    sys_msg = SystemMessage(content=websearch_prompt)
    prompt: str = f"SEARCH RESULTS:\n{message}"

    response = await remote_llm.ainvoke([sys_msg, HumanMessage(content=prompt)])

    return {
        "query": state["query"],
        "response": response.content,
    }


async def retrieve_documents(state: State) -> dict[str, Any]:
    """Retrieve documents by intelligently selecting the appropriate retriever."""
    max_chars: int = 1_000
    logger.info("Calling ===> retrieve_documents <===")

    query = state.get("query")
    prompt: str = vectorstore_routing_prompt.format(query=query)

    user_msg = {"role": "user", "content": prompt}

    retriever_choice = await get_structured_output(
        messages=[user_msg],
        model=classifier_model,
        schema=VectorSearchTypeSchema,
    )  # type: ignore
    retriever_choice: str = retriever_choice.vector_search_type.value  # type: ignore

    logger.info(f"✅ Retriever choice: {retriever_choice}")

    # Retrieve documents based on the routing decision
    if retriever_choice == VectorSearchType.FOOTBALL.value:
        retrieved_docs = vectorstore_football.similarity_search(query, k=3)
        logger.info(
            f"✅ Used football retriever, found {len(retrieved_docs)} documents"
        )
    elif retriever_choice == VectorSearchType.AI.value:
        retrieved_docs = vectorstore_ai.similarity_search(query, k=3)
        logger.info(f"✅ Used AI retriever, found {len(retrieved_docs)} documents")
    else:
        return {"response": "I couldn't find the vectorstore to answer your query."}

    # Format documents for message display
    formatted_docs = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\nContent: {doc.page_content[:max_chars]} [truncated]\n"
        for doc in retrieved_docs
    )

    return {
        "query": query,
        "documents": retrieved_docs,
        "messages": [f"Retrieved {len(retrieved_docs)} documents:\n{formatted_docs}"],
    }


async def grade_documents(state: State) -> dict[str, Any]:
    """Grade the relevance of retrieved documents."""
    logger.info("Calling ===> grade_documents <===")

    query = state.get("query")
    documents = state.get("documents", [])

    if not documents:
        logger.warning("⚠️ No documents to grade")
        return {"other_info": {"retrieval_relevance": YesOrNo.NO.value}}

    # Grade each document
    relevant_docs: list[Document] = []
    for doc in documents:
        doc_content = f"Source: {doc.metadata}\nContent: {doc.page_content}"

        sys_msg = SystemMessage(
            content=retrieval_grading_prompt.format(
                valid_output=valid_output, retrieved_documents=doc_content
            )
        )
        grading_query = query_n_retrieved_docs_prompt.format(
            query=query, retrieved_documents=doc_content
        )

        messages = convert_langchain_messages_to_dicts(
            [sys_msg, HumanMessage(content=grading_query)]
        )
        grade: GradeRetrievalSchema = await get_structured_output(
            messages=messages,
            model=classifier_model,
            schema=GradeRetrievalSchema,
        )  # type: ignore

        if grade.is_relevant.value == YesOrNo.YES.value:
            relevant_docs.append(doc)

    logger.info(f"✅ Graded documents: {len(relevant_docs)}/{len(documents)} relevant")

    return {
        "documents": relevant_docs,
        "other_info": {
            "retrieval_relevance": (
                YesOrNo.YES.value if relevant_docs else YesOrNo.NO.value
            )
        },
    }


def should_continue_to_retrieve(state: State) -> Literal["retrieve", "web_search"]:
    """Decide whether to retrieve from vectorstore or perform web search."""
    source_type = state.get("other_info", {}).get("source_type", DataSource.WEBSEARCH)

    if source_type == DataSource.VECTORSTORE.value:
        return "retrieve"
    return "web_search"


def should_continue_to_generate(
    state: State,
) -> Literal["generate", "rewrite", "failed"]:
    """Decide whether to generate response or rewrite query based on retrieval relevance."""
    relevance = state.get("other_info", {}).get("retrieval_relevance", YesOrNo.NO)
    runs: int = state.get("runs", 0)

    if runs <= 3:
        if relevance == YesOrNo.YES.value:
            return "generate"
        return "rewrite"

    return "failed"


async def generate_response(state: State) -> dict[str, Any]:
    """Generate response based on retrieved documents."""
    logger.info("Calling ===> generate_response <===")

    query = state.get("query")
    documents = state.get("documents", [])

    if not documents:
        return {
            "response": "I couldn't find relevant information to answer your query."
        }

    if documents:
        # Format documents for the prompt
        formatted_docs = "\n\n".join(
            f"Document {i + 1}:\n{doc.page_content}" for i, doc in enumerate(documents)
        )

    prompt = rag_response_generator_prompt.format(
        query=query,
        retrieved_documents=formatted_docs,  # type: ignore
    )

    response = await remote_llm.ainvoke(prompt)

    return {"response": response.content}


async def check_hallucination_node(state: State) -> dict[str, Any]:
    """Check if the generated response contains hallucinations."""
    logger.info("Calling ===> check_hallucination_node <===")

    query = state.get("query")
    runs: int = state.get("runs", 0)

    response = state.get("response")

    sys_msg = SystemMessage(
        content=hallucination_prompt.format(valid_output=valid_output)
    )
    check_query = query_n_response_prompt.format(query=query, response=response)

    messages = convert_langchain_messages_to_dicts(
        [sys_msg, HumanMessage(content=check_query)]
    )
    result: HallucinationSchema = await get_structured_output(
        messages=messages,
        model=classifier_model,
        schema=HallucinationSchema,
    )  # type: ignore

    logger.info(f"✅ Hallucination check: {result.is_hallucinating.value}")

    return {
        "runs": runs + 1,
        "other_info": {"is_hallucinating": result.is_hallucinating.value},
    }


def should_continue_to_final_answer(
    state: State,
) -> Literal["answer", "rewrite", "failed"]:
    """Decide whether to finalize answer, rewrite query, or fail based on hallucination check."""
    is_hallucinating = state.get("other_info", {}).get("is_hallucinating", YesOrNo.YES)
    runs: int = state.get("runs", 0)

    if runs <= 3:
        if is_hallucinating == YesOrNo.NO.value:
            return "answer"
        return "rewrite"

    return "failed"


async def rewrite_query(state: State) -> dict[str, Any]:
    """Rewrite the query to improve retrieval."""
    logger.info("Calling ===> rewrite_query <===")
    runs: int = state.get("runs", 0)

    query = state.get("query")
    prompt = query_rewriter_prompt.format(original_query=query)
    response = await remote_llm.ainvoke(prompt)

    rewritten = response.content
    logger.info(f"Original: {query}\nRewritten: {rewritten}")
    logger.warning(f"⚠️ Runs: {runs + 1}")

    return {
        "query": query,
        "runs": runs + 1,
        "other_info": {"rewritten_query": rewritten},
    }


def failed_node(state: State) -> dict[str, Any]:
    """Finalize the answer."""
    logger.info("Calling ===> failed_node <===")

    return {
        "response": state.get(
            "response", "I couldn't find relevant information to answer your query."
        )
    }


def answer_node(state: State) -> dict[str, Any]:
    """Finalize the answer."""
    logger.info("Calling ===> answer_node <===")

    response = state.get(
        "response", "I couldn't find relevant information to answer your query."
    )

    return {"response": response}
