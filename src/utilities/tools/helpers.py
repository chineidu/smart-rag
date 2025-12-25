import asyncio
import json
import os
import re
from typing import Any, Coroutine

import numpy as np
from bs4 import BeautifulSoup
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.documents import Document
from langchain_tavily import TavilySearch
from markdownify import markdownify as md
from tokenizers import (  # type: ignore
    Regex,
    Tokenizer,
    normalizers,
)
from tokenizers import (
    models as t_models,
)

from src import create_logger
from src.config import app_settings
from src.startup import get_bm25_setup, get_reranker_setup, get_vectorstore_setup
from src.utilities.client import HTTPXClient

logger = create_logger("helpers")

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CustomTokenizer:
    """A class for ..."""

    # pattern_digits: str = r"[0-9]+"
    pattern_punctuation: str = r"[^\w\s\\\/]"  # Includes `\`, `/`
    pattern_spaces: str = r"\s{2,}"
    pattern_split: str = r"\W"

    unk_str: str = "[UNK]"

    def __init__(self, to_lower: bool = False) -> None:
        """Initialize with a WordPiece tokenizer and normalizer sequence."""
        self.to_lower = to_lower
        self.tokenizer = Tokenizer(t_models.WordPiece(unk_token=self.unk_str))  # type: ignore

        # Create the custom normalizer
        transformations_list = []
        if self.to_lower:
            transformations_list.append(normalizers.Lowercase())

        transformations_list.extend(
            [  # type: ignore
                normalizers.NFD(),
                normalizers.Replace(Regex(self.pattern_punctuation), " "),
                normalizers.StripAccents(),
                normalizers.Strip(),
                # Last step
                normalizers.Replace(Regex(self.pattern_spaces), " "),
            ]
        )
        self.tokenizer.normalizer = normalizers.Sequence(  # type: ignore
            transformations_list  # type: ignore
        )

    def split_on_patterns(self, text: str) -> str:
        """Split a string on a pattern and join the parts with spaces.

        Parameters
        ----------
        text : str
            Input text to be split.

        Returns
        -------
        str
            Processed text with pattern-based splits.
        """
        parts: list[str] = re.split(self.pattern_split, text, flags=re.I)
        # Remove empty strings and join by spaces
        output: str = " ".join(filter(lambda x: x != "", [p.strip() for p in parts]))
        return output

    def format_data(self, data: str) -> str:
        """Format a single text string using pattern splitting and normalization.

        Parameters
        ----------
        data : str
            Input text to be formatted.

        Returns
        -------
        str
            Normalized and formatted text.
        """
        text: str = self.split_on_patterns(data)
        return self.tokenizer.normalizer.normalize_str(text)

    def batch_format_data(self, data: list[str]) -> list[str]:
        """Format a batch of text strings.

        Parameters
        ----------
        data : list[str]
            List of input texts to be formatted.

        Returns
        -------
        list[str]
            List of normalized and formatted texts.
        """
        return [self.format_data(row) for row in data]


def keyword_search(query: str, k: int = 3) -> list[Document]:
    """Perform keyword search using BM25 and return top k documents."""

    _vs_setup = get_vectorstore_setup()
    vectorstore = None
    bm25_dict = None

    if _vs_setup is not None and _vs_setup.is_ready():
        vectorstore = _vs_setup.get_vectorstore()
    if vectorstore is None:
        raise RuntimeError(
            "Vector store not initialized. Call set_vectorstore_setup(VectorStoreSetup) "
            "during app startup and ensure it is ready before using avector_search_tool."
        )
    custom_tokenizer = CustomTokenizer()

    _bm25_setup = get_bm25_setup()
    if _bm25_setup is not None and _bm25_setup.is_ready():
        bm25_dict = _bm25_setup.get_model_dict()
    if bm25_dict is None:
        raise RuntimeError(
            "BM25 model not initialized. Call set_bm25_setup(BM25Setup) during app startup "
            "and ensure it is ready before using keyword_search."
        )

    # Tokenize the query
    tokenized_query: list[str] = custom_tokenizer.format_data(query).split()
    doc_scores = bm25_dict["bm25"].get_scores(tokenized_query)
    # Sort in descending order and select the top k
    top_k_idxs: np.ndarray = np.argsort(doc_scores)[::-1][:k]

    return [bm25_dict["doc_dict"][bm25_dict["doc_ids"][i]] for i in top_k_idxs]


def rerank_documents(
    query: str, documents: list[Document], k: int = 3
) -> list[Document]:
    """Rerank documents by relevance to query using CrossEncoder.

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
    _reranker_setup = get_reranker_setup()
    reranker = None

    if _reranker_setup is not None and _reranker_setup.is_ready():
        reranker = _reranker_setup.get_model()
    if reranker is None:
        raise RuntimeError(
            "Reranker model not initialized. Call set_reranker_setup(CrossEncoderSetup) during "
            "app startup "
            "and ensure it is ready before using rerank_documents."
        )

    # Prepare pairs of (query, document content) for scoring
    pairs: list[tuple[str, str]] = [(query, doc.page_content) for doc in documents]
    # Get relevance scores from the CrossEncoder
    scores: list[float] | np.ndarray = reranker.predict(pairs)  # type: ignore

    # Combine documents with their scores
    doc_score_pairs: list[tuple[Document, float]] = list(zip(documents, scores))
    # Sort documents by score in descending order
    ranked_docs: list[Document] = [
        doc for doc, _ in sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    ][:k]
    return ranked_docs


def truncate_content(content: str | None, max_chars: int | None = None) -> str | None:
    """Truncate content to max_chars with ellipsis indicator."""
    if not content:
        return None

    if max_chars:
        return (
            f"{content[:max_chars]} [truncated]..."
            if len(content) > max_chars
            else content
        )
    return content


def extract_main_content_from_html(content: str) -> str:
    """Extract main content from HTML by removing noise and finding article body.

    Parameters
    ----------
    content : str
        Raw HTML content string.

    Returns
    -------
    str
        BeautifulSoup element containing the main content area.
        Falls back to body element if no main content found.

    Notes
    -----
    Removes scripts, styles, navigation, headers, footers, and ads.
    Searches for common content containers: main, article, or content divs.
    """
    soup = BeautifulSoup(content, "html.parser")

    # Remove unwanted elements
    for tag in soup(
        [
            "script",
            "style",
            "nav",
            "header",
            "footer",
            "aside",
            "iframe",
            "noscript",
        ]
    ):
        tag.decompose()

    # Try to find main content area (common patterns)
    main_content = None
    for selector in [
        soup.find("main"),
        soup.find("article"),
        soup.find(
            "div",
            class_=lambda x: x
            and any(
                c in str(x).lower()  # type: ignore
                for c in ["content", "article", "post", "story"]
            ),
        ),
        soup.find(
            "div",
            id=lambda x: x
            and any(
                c in str(x).lower()  # type: ignore
                for c in ["content", "article", "post", "main"]
            ),
        ),
    ]:
        if selector and selector.get_text(strip=True):
            main_content = selector
            break

    # Fall back to body if no main content found
    if not main_content:
        main_content = soup.find("body") or soup

    return str(main_content)


async def afetch_raw_content(url: str) -> str | None:
    """Fetch HTML content from a URL and convert to markdown.

    Parameters
    ----------
    url : str
        The URL to fetch content from.

    Returns
    -------
    str | None
        Markdown-converted content if successful, None otherwise.

    Notes
    -----
    Uses browser-like headers to avoid bot detection and a 15-second timeout.
    Extracts main content from common article/content tags.
    """
    # Browser-like headers to avoid bot detection
    headers: dict[str, str] = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,"
        "*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Cache-Control": "max-age=0",
    }

    try:
        async with HTTPXClient(timeout=15) as client:
            response = await client.get(url, headers=headers)

            # Check if request was successful
            if not response.get("success"):
                return None

            # Response might be dict or str
            if isinstance(response["data"], dict):
                content = json.dumps(response["data"])
            else:
                content = response["data"]
            html_content = content

            # Parse HTML and extract main content
            main_content: str = extract_main_content_from_html(content=html_content)

            # Convert to markdown
            markdown_content = md(
                str(main_content),
                heading_style="ATX",
                bullets="-",
                strip=["script", "style"],
            )

            # Clean up excessive whitespace
            lines: list[str] = [
                line.strip() for line in markdown_content.split("\n") if line.strip()
            ]
            cleaned = "\n\n".join(lines)

            return cleaned if cleaned and len(cleaned) > 100 else None

    except Exception as e:
        print(f"Warning: Failed to fetch full page content for {url}: {str(e)}")
        return None


async def aduckduckgo_search(
    query: str, fetch_full_page: bool = False, k: int = 5, max_chars: int | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Search DuckDuckGo and optionally fetch full page content.

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
    dict[str, list[dict[str, Any]]]
        Dictionary with "results" key containing list of search results.
        Each result has title, url, content, and raw_content fields.

    Notes
    -----
    When fetch_full_page=True, uses browser-like headers and smart content
    extraction to avoid bot detection and JS-blocking issues.
    """

    try:
        search = DuckDuckGoSearchResults(output_format="list", num_results=k)  # type: ignore
        _raw_results = await search.ainvoke(query)

        # format the data
        raw_results: list[dict[str, Any]] = [
            {
                "title": row["title"],
                "url": row["link"],
                "content": row["snippet"],
                "raw_content": row["snippet"],
            }
            for row in _raw_results
        ]

        if fetch_full_page:
            # Fetch full pages concurrently for better performance
            tasks: list[Coroutine[Any, Any, Any]] = [
                afetch_raw_content(row["url"]) for row in raw_results
            ]
            full_contents = await asyncio.gather(*tasks)

            raw_results = [
                {
                    **row,
                    "raw_content": truncate_content(
                        content=full_content
                        or row["content"],  # Fall back to content if fetch fails
                        max_chars=max_chars,
                    ),
                }
                for row, full_content in zip(raw_results, full_contents)
            ]
        return {"results": raw_results}

    except Exception as e:
        print(f"Duckduckgo search failed: {str(e)}")
        return {"results": []}


async def tavily_search_tool(
    query: str, fetch_full_page: bool = False, k: int = 5, max_chars: int | None = None
) -> dict[str, list[dict[str, Any]]]:
    """Search the web using TavilySearch and return formatted results.

    Parameters
    ----------
    query : str
        The search query string.
    fetch_full_page : bool, default=False
        If True, include full raw_content from Tavily results.
    k : int, optional
        Maximum number of documents to return, by default 5.
    max_chars : int or None, default=None
        Maximum characters to return per result. If None, no truncation.

    Returns
    -------
    dict[str, list[dict[str, Any]]]
        Dictionary with "results" key containing list of search results.
        Each result has title, url, content, and raw_content fields.

    Notes
    -----
    Tavily automatically provides raw_content when available.
    No additional fetching needed - Tavily handles this internally.
    """

    tavily_search = TavilySearch(
        api_key=app_settings.TAVILY_API_KEY.get_secret_value(),
        max_results=k,
        topic="general",
        # include_raw_content tells Tavily to fetch full page content
        include_raw_content=fetch_full_page,
    )
    search_response = await tavily_search.ainvoke({"query": query})  # type: ignore
    # return search_response
    raw_results: list[dict[str, Any]] = [
        {
            "title": row["title"],
            "url": row["url"],
            "content": row["content"],
            "raw_content": truncate_content(
                content=row.get("raw_content") or row["content"],
                max_chars=max_chars,
            ),
        }
        for row in search_response["results"]
    ]

    return {"results": raw_results}


def _normalize_data(documents: list[Document]) -> list[Document]:
    """Normalize document text content using CustomTokenizer.

    Parameters
    ----------
    documents : list[Document]
        List of Document objects to normalize.

    Returns
    -------
    list[Document]
        List of Document objects with normalized text content.
    """
    custom_tokenizer = CustomTokenizer(to_lower=True)
    normalized_documents: list[Document] = []

    for doc in documents:
        normalized_text = custom_tokenizer.format_data(doc.page_content)
        normalized_doc = Document(
            page_content=normalized_text,
            metadata=doc.metadata,
        )
        normalized_documents.append(normalized_doc)

    return normalized_documents
