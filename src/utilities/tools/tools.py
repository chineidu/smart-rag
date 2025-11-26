import asyncio
from typing import Any, Coroutine

from langchain_core.documents import Document
from qdrant_client import models
from qdrant_client.models import Filter

from src import create_logger
from src.startup import get_bm25_setup, get_vectorstore_setup
from src.utilities.tools.helpers import (
    aduckduckgo_search,
    keyword_search,
    rerank_documents,
    tavily_search_tool,
)

logger = create_logger(name="retrieval")


async def avector_search_tool(
    query: str,
    filter: str | None = None,
    k: int = 3,
) -> list[Document]:
    """Perform a vector search with metadata filtering.

    Parameters
    ----------
    query : str
        The search query string.
    filter : str or None, default=None
        The metadata filter value for 'metadata.section'.
    k : int, default=3
        The number of top similar documents to retrieve.

    Returns
    -------
    list[Document]
        A list of retrieved Document objects.
    """
    _vs_setup = get_vectorstore_setup()
    if _vs_setup is not None and _vs_setup.is_ready():
        vectorstore = _vs_setup.get_vectorstore()

    if vectorstore is None:
        raise RuntimeError(
            "Vector store not initialized. Call set_vectorstore_setup(VectorStoreSetup) "
            "during app startup and ensure it is ready before using avector_search_tool."
        )
    key: str = "metadata.section"
    _filter: Filter | None = (
        models.Filter(
            must=[models.FieldCondition(key=key, match=models.MatchValue(value=filter))]
        )
        if filter
        else None
    )
    return await vectorstore.asimilarity_search(query, k=k, filter=_filter)


async def akeyword_search_tool(
    query: str,
    filter: str | None = None,  # noqa: ARG001
    k: int = 3,
) -> list[Document]:
    """Perform keyword search asynchronously using BM25 and return top k documents.

    Parameters
    ----------
    query : str
        The search query string.
    filter : str | None, default=None
        For function signature compatibility. (Not used in keyword search)
    k : int, default=3
        The number of top similar documents to retrieve.

    Returns
    -------
    list[Document]
        A list of retrieved Document objects.
    """
    return await asyncio.to_thread(keyword_search, query, k)


async def ahybrid_search_tool(
    query: str, filter: str | None = None, k: int = 5
) -> list[Document]:
    """
    Asynchrounously combine vector and keyword search results using Reciprocal Rank Fusion (RRF).

    Parameters
    ----------
    query : str
        The search query string.
    filter : str or None, optional
        Optional filter expression passed to the vector search, by default None.
    k : int, optional
        Maximum number of documents to return, by default 5.

    Returns
    -------
    list[Document]
        Top-k documents ranked by fused scores.

    Notes
    -----
    RRF is a simple, unsupervised method for merging ranked lists.
    The constant ``K`` (set to 61) controls the steepness of the rank
    discount curve and is taken from the original RRF paper.
    """
    K: int = 61  # Default for RRF

    _vs_setup = get_vectorstore_setup()
    if _vs_setup is not None and _vs_setup.is_ready():
        vectorstore = _vs_setup.get_vectorstore()

    if vectorstore is None:
        raise RuntimeError(
            "Vector store not initialized. Call set_vectorstore_setup(VectorStoreSetup) "
            "during app startup and ensure it is ready before using avector_search_tool."
        )

    _bm25_setup = get_bm25_setup()
    if _bm25_setup is not None and _bm25_setup.is_ready():
        bm25_dict: dict[str, Any] | None = _bm25_setup.get_model_dict()
    if bm25_dict is None:
        raise RuntimeError(
            "BM25 model not initialized. Call set_bm25_setup(BM25Setup) during app startup "
            "and ensure it is ready before using keyword_search."
        )
    logger.info("Performing hybrid search")

    tasks: list[Coroutine[Any, Any, list[Document]]] = [
        avector_search_tool(query=query, filter=filter, k=k),  # type: ignore
        akeyword_search_tool(query=query, k=k),  # type: ignore
    ]
    semantic_docs, kw_docs = await asyncio.gather(*tasks)

    # Create a unified document dictionary from both search results
    all_docs: dict[str, Document] = {}
    for doc in semantic_docs + kw_docs:
        doc_id = doc.metadata["chunk_id"]
        if doc_id not in all_docs:
            all_docs[doc_id] = doc

    # Results of vector and kw search
    res_ids: list[list[str]] = [
        [doc.metadata["chunk_id"] for doc in semantic_docs],
        [doc.metadata["chunk_id"] for doc in kw_docs],
    ]
    # Calculate Reciprocal Rank Fusion (RRF)
    rrf_dict: dict[str, float] = {}

    for doc_list in res_ids:
        # Grab each doc_id
        for idx, doc_id in enumerate(doc_list):
            if doc_id not in rrf_dict:
                rrf_dict[doc_id] = 0
            # Add (1 / (idx + k)) to each retrieved doc
            rrf_dict[doc_id] += 1 / (idx + K)
    # Sort result using RRF score in descending order
    ranked_ids: list[str] = sorted(
        rrf_dict.keys(), key=lambda x: rrf_dict[x], reverse=True
    )[:k]

    return [all_docs[_id] for _id in ranked_ids]


async def arerank_documents(
    query: str, documents: list[Document], k: int = 3
) -> list[Document]:
    """Asynchronously rerank documents by relevance to query using CrossEncoder.

    Parameters
    ----------
    query : str
        The search query string.
    documents : list[Document]
        List of Document objects to rerank.
    k : int, optional
        Maximum number of documents to return, by default 3.

    Returns
    -------
    list[Document]
        Documents sorted by relevance score in descending order.
    """
    logger.info("Performing document reranking")
    return await asyncio.to_thread(rerank_documents, query, documents, k)


async def aduckduckgo_web_search_tool(
    query: str, fetch_full_page: bool = False, k: int = 5, max_chars: int | None = None
) -> list[Document]:
    """Asynchronously search DuckDuckGo and optionally fetch full page content.

    Parameters
    ----------
    query : str
        The search query string.
    fetch_full_page : bool, default=False
        If True, fetch and parse full HTML content for each result.
    k : int, optional
        Maximum number of documents to return, by default 5.
    max_chars : int or None, default=None
        Maximum characters to return per result. If None, no truncation.

    Returns
    -------
    list[Document]
    """
    max_chars = 8_000 if not max_chars else max_chars

    search_response: dict[str, list[dict[str, Any]]] = await aduckduckgo_search(
        query=query, fetch_full_page=fetch_full_page, k=k, max_chars=max_chars
    )

    formatted_results: list[Document] = [
        Document(
            page_content=f"Title: {result['title']}\nContent: {result['raw_content']}",
            metadata={
                "url": result["url"],
                "title": result["title"],
            },
        )
        for result in search_response["results"]
    ]
    return formatted_results


async def atavily_web_search_tool(
    query: str, fetch_full_page: bool = False, k: int = 5, max_chars: int | None = None
) -> list[Document]:
    """Asynchronously search using Tavily and optionally fetch full page content.

    Parameters
    ----------
    query : str
        The search query string.
    fetch_full_page : bool, default=False
        If True, fetch and parse full HTML content for each result.
    k : int, optional
        Maximum number of documents to return, by default 5.
    max_chars : int or None, default=None
        Maximum characters to return per result. If None, no truncation.

    Returns
    -------
    list[Document]
    """
    max_chars = 8_000 if not max_chars else max_chars
    search_response: dict[str, list[dict[str, Any]]] = await tavily_search_tool(
        query=query, fetch_full_page=fetch_full_page, k=k, max_chars=max_chars
    )

    formatted_results: list[Document] = [
        Document(
            page_content=f"Title: {result['title']}\nContent: {result['raw_content']}",
            metadata={
                "url": result["url"],
                "title": result["title"],
            },
        )
        for result in search_response["results"]
    ]
    return formatted_results
