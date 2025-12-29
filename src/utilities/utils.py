import asyncio
import json
import os
import re
import unicodedata
from glob import glob
from pathlib import Path
from re import Match, Pattern
from typing import TYPE_CHECKING, Any, Callable, Coroutine, cast
from uuid import uuid4

from bs4 import BeautifulSoup
from langchain_community.document_loaders import (
    CSVLoader,
    JSONLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langsmith import traceable
from markdownify import markdownify as md
from pydantic import BaseModel

from src import create_logger
from src.prompts import PromptsBuilder
from src.schemas.base import ModelList
from src.schemas.nodes_schema import (
    Decision,
    Plan,
    RetrieverMethod,
    ReWrittenQuery,
    Step,
)
from src.schemas.types import (
    EventsType,
    FileFormatsType,
    MemoryData,
    RetrieverMethodType,
    T,
)
from src.utilities.client import HTTPXClient, get_instructor_openrouter_client
from src.utilities.model_config import RemoteModel
from src.utilities.tools.tools import (
    ahybrid_search_tool,
    akeyword_search_tool,
    arerank_documents,
    avector_search_tool,
)

logger = create_logger("utilities")


if TYPE_CHECKING:
    from src.state import State, StepState

prompt_builder = PromptsBuilder()


def merge_dicts(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Merge two dictionaries, with new values updating existing ones."""
    if existing is None:
        return new
    # Update existing dict with new dict values
    return {**existing, **new}


@traceable
async def get_structured_output(
    messages: list[dict[str, Any]],
    model: RemoteModel | None,
    schema: type[T],
) -> BaseModel:
    """
    Retrieves structured output from a chat completion model.

    Parameters
    ----------
    messages : list[dict[str, Any]]
        The list of messages to send to the model for the chat completion.
    model : RemoteModel
        The remote model to use for the chat completion (e.g., 'gpt-4o').
    schema : PydanticModel
        The Pydantic schema to enforce for the structured output.

    Returns
    -------
    BaseModel
        An instance of the provided Pydantic schema containing the structured output.

    Notes
    -----
    This is an asynchronous function that awaits the completion of the API call.
    """
    model = model if model else RemoteModel.GEMINI_2_5_FLASH_LITE

    aclient = await get_instructor_openrouter_client()
    return await aclient.chat.completions.create(
        model=model,
        response_model=schema,
        messages=messages,  # type: ignore
        max_retries=5,
    )


def convert_langchain_messages_to_dicts(
    messages: list[HumanMessage | SystemMessage | AIMessage],
) -> list[dict[str, str]]:
    """Convert LangChain messages to a list of dictionaries.

    Parameters
    ----------
    messages : list[HumanMessage | SystemMessage | AIMessage]
        List of LangChain message objects to convert.

    Returns
    -------
    list[dict[str, str]]
        List of dictionaries with 'role' and 'content' keys.
        Roles are mapped as follows:
        - HumanMessage -> "user"
        - SystemMessage -> "system"
        - AIMessage -> "assistant"

    """
    role_mapping: dict[str, str] = {
        "SystemMessage": "system",
        "HumanMessage": "user",
        "AIMessage": "assistant",
    }

    converted_messages: list[dict[str, str]] = []
    for msg in messages:
        message_type: str = msg.__class__.__name__
        # Default to "user" if unknown
        role: str = role_mapping.get(message_type, "user")
        converted_messages.append({"role": role, "content": msg.content})  # type: ignore

    return converted_messages


def load_csv_doc(filepath: str | Path, _jq_schema: str | None = None) -> list[Document]:
    """This is used to load CSV documents."""
    loader = CSVLoader(filepath)
    docs: list[Document] = loader.load()  # type: ignore
    return docs


def load_pdf_doc(filepath: str | Path, _jq_schema: str | None = None) -> list[Document]:
    """This is used to load a PDF document using PyPDFLoader."""
    loader = PyPDFLoader(filepath)
    docs: list[Document] = loader.load()  # type: ignore
    return docs


def load_json_doc(filepath: str | Path, jq_schema: str | None = None) -> list[Document]:
    """This is used to load JSON documents."""
    if jq_schema is None:
        raise ValueError("jq_schema is required for loading JSON documents")
    loader = JSONLoader(filepath, jq_schema=jq_schema)
    docs: list[Document] = loader.load()  # type: ignore
    return docs


def load_txt_doc(filepath: str | Path, _jq_schema: str | None = None) -> list[Document]:
    """This is used to load txt documents."""
    loader = TextLoader(filepath)
    docs: list[Document] = loader.load()  # type: ignore
    return docs


type FlatFileFn = Callable[[str | Path, str | None], list[Document]]

file_formats_dict: dict[FileFormatsType, FlatFileFn] = {
    FileFormatsType.CSV: load_csv_doc,
    FileFormatsType.JSON: load_json_doc,
    FileFormatsType.PDF: load_pdf_doc,
    FileFormatsType.TXT: load_txt_doc,
}


def load_document(
    filepath: str | Path, jq_schema: str | None, format: FileFormatsType
) -> list[Document]:
    """This is used to load (CSV, JSON, PDF, TXT) documents."""
    flat_file_fn = file_formats_dict[format]
    return flat_file_fn(filepath, jq_schema)


_ESCAPE_RE: re.Pattern[str] = re.compile(
    r"(?:\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|\\U[0-9a-fA-F]{8})"
)
_ZERO_WIDTH: set[str] = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}


def normalize_header_string(text: str) -> str:
    """Normalize header-like strings with minimal, safe transforms.

    Applies targeted unicode-escape decoding when present, replaces NBSP, removes
    zero-width characters, normalizes (NFKC), and collapses whitespace.
    """
    # Targeted backslash-escape decoding (avoid decoding unrelated backslashes)
    if _ESCAPE_RE.search(text):

        def _sub(m: re.Match[str]) -> str:
            token = m.group(0)
            try:
                return token.encode().decode("unicode_escape")
            except Exception:
                return token

        text = _ESCAPE_RE.sub(_sub, text)

    # Replace non‑breaking space with normal space
    text = text.replace("\u00a0", " ")

    # Remove zero‑width characters
    if any(ch in text for ch in _ZERO_WIDTH):
        text = "".join(ch for ch in text if ch not in _ZERO_WIDTH)

    # Unicode normalize and collapse whitespace
    text = unicodedata.normalize("NFKC", text)
    return " ".join(text.split())


def clean_xbrl_noise(text: str) -> str:
    """Aggressively remove XBRL noise while preserving document structure.

    This function removes all XBRL/XML metadata and keeps only the meaningful
    HTML content that can be converted to readable markdown.

    Parameters
    ----------
    text : str
        The input HTML or XML text containing XBRL data.

    Returns
    -------
    str
        The cleaned text with XBRL noise removed.
    """

    body_match = re.search(r"<body[^>]*>(.*)</body>", text, re.DOTALL | re.IGNORECASE)
    if body_match:
        text = "<body>" + body_match.group(1) + "</body>"

    try:
        soup = BeautifulSoup(text, "html.parser")

        # Remove <head> entirely - it contains most XBRL metadata
        for head in soup.find_all("head"):
            head.decompose()

        # Remove all script and style tags
        for tag in soup(["script", "style", "meta", "link"]):
            tag.decompose()

        # Remove XML/XBRL namespaced elements (tags with colons)
        for tag in soup.find_all():
            if tag.name and ":" in tag.name:
                tag.decompose()

        # Remove hidden XBRL data elements (usually display:none or specific XBRL classes)
        for tag in soup.find_all(style=re.compile(r"display:\s*none", re.I)):
            tag.decompose()

        for tag in soup.find_all(class_=re.compile(r"xbrl|hidden", re.I)):
            tag.decompose()

        # Remove specific XBRL attribute clutter
        for tag in soup.find_all():
            if tag.name:
                # Remove XBRL attributes
                attrs_to_remove = []
                for attr in tag.attrs:
                    if (
                        ":" in attr
                        or attr.startswith("xmlns")
                        or attr in ["contextref", "unitref", "decimals"]
                    ):
                        attrs_to_remove.append(attr)  # noqa: PERF401
                for attr in attrs_to_remove:
                    del tag[attr]

        # Get the cleaned HTML
        cleaned: str = str(soup)

    except Exception as e:
        print(f"Warning: HTML parsing failed: {e}")
        cleaned = text

    # Post-processing regex cleanup for any remaining XBRL noise

    # Remove namespace URLs that got left behind
    cleaned = re.sub(
        r'http://[^\s<>"]+(?:xbrl|fasb|sec\.gov)[^\s<>"]*', "", cleaned, flags=re.I
    )

    # Remove XBRL namespace tokens (us-gaap:Something, iso4217:USD, etc.)
    cleaned = re.sub(
        r"\b(?:us-gaap|nvda|srt|stpr|fasb|xbrli|iso4217|xbrl|dei|ix|country|xbrldi|link):[A-Za-z0-9_\-:()]+(?:Member)?\b",
        "",
        cleaned,
        flags=re.I,
    )

    # Remove long numeric strings (CIK numbers, etc.) - 10+ digits
    cleaned = re.sub(r"\b\d{10,}\b", "", cleaned)
    # Remove date patterns that are concatenated without separators (2023-01-292022-01-30)
    cleaned = re.sub(r"(?:\d{4}-\d{2}-\d{2}){2,}", "", cleaned)
    # Remove very long alphanumeric strings (40+ chars) that indicate concatenated tags
    cleaned = re.sub(r"\b[A-Za-z0-9_\-]{40,}\b", "", cleaned)
    # Remove XML/namespace declarations
    cleaned = re.sub(r'xmlns[:\w]*="[^"]*"', "", cleaned)
    cleaned = re.sub(r'xml:\w+="[^"]*"', "", cleaned)
    # Remove "pure" standalone (XBRL unit)
    cleaned = re.sub(r"\bpure\b(?!\s+\w)", "", cleaned)
    # Clean up multiple colons and extra punctuation
    cleaned = re.sub(r":{2,}", ":", cleaned)
    return re.sub(r"\s*:\s*:\s*", " ", cleaned)


async def download_and_parse_data(
    url: str,
    raw_doc_path: Path | str,
    cleaned_doc_path: Path | str,
    force_download: bool = False,
) -> None:
    """Download and parse HTML/XBRL documents with aggressive noise removal.

    Parameters
    ----------
        url : str
            The remote URL to download
        raw_doc_path : Path | str
            Output path for the raw bytes/text
        cleaned_doc_path : Path | str
            Output path for the cleaned markdown/text
        force_download : bool, default=False
            When True, re-download and re-clean even if file(s) exist

    Returns
    -------
        None
    """
    if isinstance(raw_doc_path, str):
        raw_doc_path = Path(raw_doc_path)
    if isinstance(cleaned_doc_path, str):
        cleaned_doc_path = Path(cleaned_doc_path)

    # Safe, identifiable user agent:
    USER_AGENT: str = (
        "MyCompany MyDownloader/1.0 (+https://mycompany.example; dev@mycompany.example)"
    )
    headers: dict[str, str] = {"User-Agent": USER_AGENT, "Accept": "application/json"}

    # If raw document exists and we are not forcing re-download
    if raw_doc_path.exists() and raw_doc_path.is_file() and not force_download:
        print(f"Raw file already exists: {raw_doc_path}. Skipping download.")
    else:
        # Ensure the path exists
        raw_doc_path.parent.mkdir(parents=True, exist_ok=True)

        async with HTTPXClient() as client:
            response: dict[str, Any] = await client.get(url, headers=headers)

        if not response["success"]:
            print(f"Failed to download {url}: {response.get('error')}")
            return

        # Response data may be a dict or string; store as text
        raw_content: Any = response["data"]
        if not isinstance(raw_content, str):
            # Coerce to text safely
            try:
                raw_content = json.dumps(raw_content, ensure_ascii=False)
            except Exception:
                raw_content = str(raw_content)

        raw_doc_path.write_text(raw_content, encoding="utf-8")
        print(f"Saved raw content to {raw_doc_path}")

    # Convert the raw HTML/text into a cleaned markdown or plain text
    raw_text: str = raw_doc_path.read_text(encoding="utf-8")

    # Use the aggressive cleaner to remove XBRL noise
    cleaned_html = clean_xbrl_noise(raw_text)

    # For HTML content, convert to markdown with better formatting
    try:
        # Configure markdownify to preserve more structure
        cleaned_text: str = md(
            cleaned_html,
            heading_style="ATX",  # Use # for headers
            bullets="-",  # Use - for bullet points
            strong_em_symbol="**",  # Use ** for bold
            strip=["script", "style"],  # Remove script and style tags
        )
    except Exception as e:
        # If markdownify fails, try basic text extraction
        print(f"Warning: Markdown conversion failed: {e}")
        try:
            soup = BeautifulSoup(cleaned_html, "html.parser")
            cleaned_text = soup.get_text("\n", strip=True)
        except Exception:
            cleaned_text = cleaned_html

    # Post-processing cleanup on the markdown text
    # Remove lines that are mostly XBRL noise (lots of colons, short tokens)
    lines: list[str] = cleaned_text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        # Skip lines with excessive XBRL patterns
        if len(line) < 10:  # Keep very short lines (might be intentional)
            cleaned_lines.append(line)
            continue

        # Count suspicious patterns
        colon_count = line.count(":")
        token_count = len(
            re.findall(r"\b[A-Z][a-z]+(?:[A-Z][a-z]+)+\b", line)
        )  # CamelCase tokens

        # If line has too many colons or camelCase tokens relative to length, skip it
        if colon_count > len(line) / 20 or (token_count > 5 and len(line.split()) < 20):
            continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)

    # Remove excessive blank lines (more than 2 consecutive)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    # Remove leading/trailing whitespace from each line
    cleaned_text = "\n".join(line.strip() for line in cleaned_text.split("\n"))

    # Final whitespace cleanup
    cleaned_text = cleaned_text.strip()

    # Ensure the path exists
    cleaned_doc_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned_doc_path.write_text(cleaned_text, encoding="utf-8")

    print(f"Saved cleaned content to {cleaned_doc_path}")
    return


def extract_10k_sections(documents: list[Document]) -> tuple[list[str], list[str]]:
    """Extract 10-K sections with title and content separately (line-by-line comments)

    Parameters
    ----------
    documents : list[Document]
        The list of documents to extract sections from.

    Returns
    -------
    tuple[list[str], list[str]]
        A tuple containing two lists:
        - List of section titles.
        - List of section contents.
    """

    # Get the entire document text from the TextLoader's first document
    raw_text: str = documents[0].page_content  # the string to search for ITEM headers

    # Header pattern: match 'ITEM 1.' or 'ITEM 1A.' etc. at the beginning of a line
    # ^\s*            -> allow leading whitespace before the header
    # ITEM\s+         -> the literal word ITEM followed by at least one space
    # \d+             -> the item number (one or more digits)
    # [A-Z]?           -> optional letter (A, B, etc.) after the number
    # \.               -> period following the number (escaped dot)
    # [\t ]+          -> at least one whitespace char (tab/space) after the dot
    # [^\n\r]*        -> the remainder of the heading line (until newline)
    # re.MULTILINE     -> ^ anchors at the beginning of each line
    header_pattern: Pattern[str] = re.compile(
        r"^\s*(ITEM\s+\d+[A-Z]?\.[\t ]+[^\n\r]*)", re.MULTILINE
    )

    # run finditer which returns match objects with start()/end() locations
    matches: list[Match[str]] = list(
        header_pattern.finditer(raw_text)
    )  # convert to list for indexing

    # Prepare lists to hold the results
    section_titles: list[
        str
    ] = []  # will store the header lines like 'ITEM 1. BUSINESS'
    # will store the textual content of each section (no header)
    section_content: list[str] = []

    # Walk through each header match, capturing both title and the content after it
    for i, match in enumerate(matches):
        title: str = match.group(
            1
        ).strip()  # capture the heading text and strip whitespace
        # Normalize the header to handle NBSP/zero-width and consistent spacing
        title = normalize_header_string(title)
        section_titles.append(title)

        # The content begins right after the matched heading line
        start_pos: int = match.end()  # numeric index where this header finishes

        # Determine where this section ends: next header start or the end of the document
        if i + 1 < len(matches):
            end_pos: int = matches[i + 1].start()  # next header's start position
        else:
            end_pos = len(raw_text)  # or EOF if this is the last header

        # Use the start/end slices to get the body text and strip leading/trailing whitespace
        content: str = raw_text[start_pos:end_pos].strip()  # remove extra whitespace
        section_content.append(content)  # store the cleaned body in the sections list

    # Confirmation print for quick inspection when the cell runs
    print(f"Found {len(section_titles)} ITEM sections.")

    return (section_titles, section_content)


def chunk_data_default(
    documents: list[Document],
    chunk_size: int = 1_000,
    chunk_overlap: int = 100,
    add_start_index: bool = True,
) -> list[Document]:
    """Chunk document sections into smaller pieces with metadata.

    Parameters
    ----------
    documents : list[Document]
            List of documents to split. Each item should be a Document containing textual content
            (for example via a `page_content` or `text` attribute). Original document metadata is
            preserved and propagated to resulting chunks.
    chunk_size : int, default=1000
            Maximum number of characters per chunk.
    chunk_overlap : int, default=100
        Number of overlapping characters between consecutive chunks.
    add_start_index : bool, default=True
        Whether to add a `start_index` field in metadata to track original position.

    Returns
    -------
    list[Document]
        List of Document chunks with metadata including source, section, and chunk_id.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # chunk size (characters)
        chunk_overlap=chunk_overlap,  # chunk overlap (characters)
        add_start_index=add_start_index,  # track index in original document
    )

    _documents: list[Document] = text_splitter.split_documents(documents)
    doc_chunks_with_metadata: list[Document] = []

    # Update the metadata for each chunk
    for doc in _documents:
        chunk_id: str = str(uuid4())  # unique ID for this chunk
        doc_chunks_with_metadata.append(  # noqa: PERF401
            Document(
                page_content=doc.page_content,
                metadata={
                    **doc.metadata,
                    # Add unique chunk ID
                    "chunk_id": chunk_id,
                },
            )
        )
    print(f"Created {len(doc_chunks_with_metadata)} document chunks with metadata.")

    return doc_chunks_with_metadata


def chunk_data_by_sections(
    source: str,
    section_titles: list[str],
    section_content: list[str],
    chunk_size: int = 1_000,
    chunk_overlap: int = 100,
    add_start_index: bool = True,
) -> list[Document]:
    """Chunk document sections into smaller pieces with metadata.

    Parameters
    ----------
    source : str
        The original document source (for source metadata).
    section_titles : list[str]
        List of section titles extracted from the document.
    section_content : list[str]
        List of section contents corresponding to the titles.

    Returns
    -------
    list[Document]
        List of Document chunks with metadata including source, section, and chunk_id.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # chunk size (characters)
        chunk_overlap=chunk_overlap,  # chunk overlap (characters)
        add_start_index=add_start_index,  # track index in original document
    )

    doc_chunks_with_metadata: list[Document] = []

    # Loop thru each section's content and its title
    for title, content in zip(section_titles, section_content):
        section_chunks: list[str] = text_splitter.split_text(content)

        # Loop thru each chunk to add metadata
        for chunk in section_chunks:
            chunk_id: str = str(uuid4())  # unique ID for this chunk
            doc_chunks_with_metadata.append(  # noqa: PERF401
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source,  # original document path
                        # Ensure section titles are normalized in metadata
                        "section": normalize_header_string(title),
                        "chunk_id": chunk_id,  # unique chunk ID
                    },
                )
            )
    print(f"Created {len(doc_chunks_with_metadata)} document chunks with metadata.")

    return doc_chunks_with_metadata


def load_all_documents(
    filepaths: str | list[str],
    jq_schema: str | None,
    format: FileFormatsType | str,
    filepaths_is_glob: bool = False,
) -> list[Document]:
    """
    Load all documents from specified file paths.

    Parameters
    ----------
    filepaths : str or list[str]
        Path to a single file or a list of paths.
    jq_schema : str or None
        JQ schema to apply when loading documents.
    format : FileFormatsType | str
        Format of the files to be loaded.
    filepaths_is_glob : bool, default=False
        If True, treat `filepaths` as a glob pattern to match multiple files.

    Returns
    -------
    list[Document]
        List of loaded Document objects.
    """

    if isinstance(format, str):
        format = FileFormatsType(format)

    if filepaths_is_glob:
        if isinstance(filepaths, list):
            # If already a list, glob each path
            all_files: list[str] = []
            for fp in filepaths:
                all_files.extend(glob(f"{fp}/*.{format.value}"))
            filepaths = all_files
        else:
            # If single string, glob it directly
            filepaths = glob(f"{filepaths}/*.{format.value}")

    elif isinstance(filepaths, str):
        filepaths = [filepaths]

    docs: list[Document] = [
        doc
        for fp in filepaths
        for doc in load_document(filepath=fp, jq_schema=jq_schema, format=format)
    ]
    print(f"Loaded {len(docs)} documents from {len(filepaths)} filepaths.")

    return docs


def dedupe_model_list(model_list: ModelList[T]) -> ModelList[T]:
    """De-duplicate a ModelList by converting to JSON for comparison.

    Parameters
    ----------
    model_list : ModelList[T]
        Input ModelList to deduplicate.

    Returns
    -------
    ModelList[T]
        De-duplicated ModelList.
    """
    seen = set()
    unique_items: list[T] = []

    for item in model_list.items:
        # Convert to dict for comparison - use mode='json' for full serialization
        item_dict = (
            item.model_dump(mode="json") if hasattr(item, "model_dump") else dict(item)
        )  # type: ignore

        try:
            key = json.dumps(item_dict, sort_keys=True, separators=(",", ":"))
        except TypeError:
            # Fallback for non-serializable objects
            key = str(sorted(item_dict.items()))

        if key not in seen:
            seen.add(key)
            unique_items.append(item)  # type: ignore

    return ModelList[T](items=unique_items)


def dedupe_dicts(lst: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """De-duplicate a list of dictionaries.

    Parameters
    ----------
    lst : list[dict[str, Any]]
        Input list of dictionaries or BaseModel instances.

    Returns
    -------
    list[dict[str, Any]]
        De-duplicated list of dictionaries.
    """
    seen = set()
    out: list[dict[str, Any]] = []
    for data in lst:
        # Convert BaseModel to dict if needed
        if isinstance(data, BaseModel):
            # Use mode='json' to ensure nested objects are also serialized
            item_dict = (
                data.model_dump(mode="json")
                if hasattr(data, "model_dump")
                else dict(data)
            )
        else:
            item_dict = data

        try:
            key = json.dumps(item_dict, sort_keys=True, separators=(",", ":"))
        except TypeError:
            # Fallback: convert to string representation if JSON serialization fails
            key = str(sorted(item_dict.items()))

        if key not in seen:
            seen.add(key)
            out.append(item_dict)

    return out


def append_memory(existing: MemoryData, new: MemoryData) -> MemoryData:
    """Merge new memory data into existing memory, appending lists of dicts and merging dicts.

    Parameters
    ----------
    existing: MemoryData
        The existing memory data.
    new: MemoryData
        The new memory data to merge.

    Returns
    -------
    MemoryData
        The merged memory data.
    """
    # Handle empty cases
    if not existing or (isinstance(existing, (list, ModelList)) and len(existing) == 0):
        return new

    if not new or (isinstance(new, (list, ModelList)) and len(new) == 0):
        return existing

    # Type checking
    if type(existing) is not type(new):
        raise ValueError(
            f"Existing and new memory data must be of the same type. Got {type(existing)} and {type(new)}"
        )

    if isinstance(existing, list) and isinstance(new, list):
        if isinstance(existing[0], BaseModel) or isinstance(new[0], BaseModel):
            existing, new = (
                ModelList[BaseModel](items=existing),  # type: ignore
                ModelList[BaseModel](items=new),  # type: ignore
            )

    # Case 1: Both are ModelList - merge and deduplicate
    if isinstance(existing, ModelList) and isinstance(new, ModelList):
        # Combine items
        combined = ModelList[T](items=existing.items + new.items)  # type: ignore
        # Deduplicate and return
        return dedupe_model_list(combined).items  # type: ignore

    # Case 2: Both are list of dicts or BaseModels - merge as before
    if isinstance(existing, list) and isinstance(new, list):
        if not existing or not new:
            return existing or new

        # Check if items are dicts or BaseModels (for backwards compatibility)
        if isinstance(existing[0], (dict, BaseModel)) and isinstance(
            new[0], (dict, BaseModel)
        ):
            combined = existing + new
            return dedupe_dicts(combined)

    # Case 3: Both are dicts - merge dictionaries
    if isinstance(existing, dict) and isinstance(new, dict):
        result: dict[str, Any] = existing.copy()

        for key, new_value in new.items():
            # Skip None or empty values
            if new_value is None or new_value == "" or new_value == []:
                continue

            existing_value = result.get(key, None)

            # If key doesn't exist, just add it
            if existing_value is None:
                result[key] = new_value
                continue

            # Lists: combine and remove duplicates
            if isinstance(new_value, list):
                combined = existing_value + new_value
                result[key] = (
                    dedupe_dicts(combined)
                    if isinstance(combined[0], dict)
                    else list(dict.fromkeys(combined))
                )

            # Dicts: merge
            elif isinstance(new_value, dict):
                result[key] = {**existing_value, **new_value}

            # Everything else: new value overwrites
            else:
                result[key] = new_value

        return result

    raise ValueError(f"Unsupported memory data type: {type(existing)}")


def merge_step_states(
    existing: "list[StepState]", new: "list[StepState]"
) -> "list[StepState]":
    """Intelligently merge step states by step_index, combining duplicate steps.

    This reducer merges steps with the same step_index, combining their:
    - rewritten_queries (deduplicated)
    - retrieved_documents (deduplicated)
    - summaries (concatenated if different)

    Special behavior: If new contains a single None value, returns empty list (reset).

    Parameters
    ----------
    existing : list[StepState]
        Existing list of step states.
    new : list[StepState]
        New step states to merge.

    Returns
    -------
    list[StepState]
        Merged list of step states.

    Examples
    --------
    >>> existing = [{'step_index': 0, 'summary': ''}]
    >>> new = [{'step_index': 0, 'summary': 'Content here'}]
    >>> merged = merge_step_states(existing, new)
    >>> print(merged[0]['summary'])  # 'Content here'
    """
    # Reset detection: if new list contains None, clear state
    # e.g., existing = [...], new = [None] -> return []
    if new and len(new) == 1 and new[0] is None:  # type: ignore
        return []

    # Create a mapping of step_index to step for efficient lookup
    step_map: dict[int, StepState] = {}

    # Add existing steps to map
    for step in existing:
        step_map[step["step_index"]] = step.copy()

    # Merge new steps
    for new_step in new:
        step_idx = new_step["step_index"]

        if step_idx in step_map:
            # Step exists, merge it
            existing_step = step_map[step_idx]

            # Merge question (prefer non-empty)
            if (
                not existing_step.get("question", "").strip()
                and new_step.get("question", "").strip()
            ):
                existing_step["question"] = new_step["question"]

            # Merge rewritten_queries (deduplicate)
            existing_queries = existing_step.get("rewritten_queries", [])
            new_queries = new_step.get("rewritten_queries", [])
            existing_step["rewritten_queries"] = list(
                dict.fromkeys(existing_queries + new_queries)
            )

            # Merge retrieved_documents (deduplicate using append_memory)
            existing_docs = existing_step.get("retrieved_documents", [])
            new_docs = new_step.get("retrieved_documents", [])
            if existing_docs or new_docs:
                existing_step["retrieved_documents"] = append_memory(  # type: ignore
                    existing_docs,
                    new_docs,
                )

            # Merge summaries
            existing_summary = existing_step.get("summary", "").strip()
            new_summary = new_step.get("summary", "").strip()

            if not existing_summary and new_summary:
                existing_step["summary"] = new_summary
            elif existing_summary and new_summary and existing_summary != new_summary:
                # Both have different content - combine
                existing_step["summary"] = f"{existing_summary}\n\n{new_summary}"
            elif existing_summary and not new_summary:
                # Keep existing
                pass

        else:
            # New step, add it
            step_map[step_idx] = new_step.copy()

    # Convert back to sorted list by step_index
    return sorted(step_map.values(), key=lambda x: x["step_index"])


def merge_documents(existing: list[Document], new: list[Document]) -> list[Document]:
    """Merge and deduplicate Document lists.

    A cleaner wrapper around append_memory specifically for Documents.

    Special behavior: If new contains a single None value, returns empty list (reset).

    Parameters
    ----------
    existing : list[Document]
        Existing documents.
    new : list[Document]
        New documents to merge.

    Returns
    -------
    list[Document]
        Merged and deduplicated documents.
    """
    # Reset detection: if new list contains None, clear documents
    if new and len(new) == 1 and new[0] is None:  # type: ignore
        return []

    if not existing:
        return new
    if not new:
        return existing
    return append_memory(existing, new)  # type: ignore


def concatenate_strings(existing: str, new: str) -> str:
    """Concatenate two strings with a separator, handling empty strings.

    Parameters
    ----------
    existing : str
        Existing string.
    new : str
        New string to append.

    Returns
    -------
    str
        Concatenated string.
    """
    separator: str = "\n\n"

    existing = existing.strip() if existing else ""
    new = new.strip() if new else ""

    if not existing:
        return new
    if not new:
        return existing

    # Avoid duplicates
    if existing == new:
        return existing

    return f"{existing}{separator}{new}"


# ---------------------------------------------------------
# -------------- HELPER FUNCTIONS FOR NODES ---------------
# ---------------------------------------------------------
def deduplicate(documents: list[Document]) -> list[Document]:
    """Deduplicate documents based on 'chunk_id' in metadata."""
    docs_dict: dict[str, Document] = {}

    if not documents[0].metadata:
        raise ValueError(
            "Cannot deduplicate documents without 'chunk_id' in metadata. Please ensure "
            "documents have 'chunk_id' in their metadata."
        )
    for doc in documents:
        if (_id := doc.metadata["chunk_id"]) not in docs_dict:
            docs_dict[_id] = doc
    return list(docs_dict.values())


def format_documents(documents: list[Document]) -> str:
    """Format documents for synthesis input."""
    delimiter: str = "===" * 20
    try:
        docs: list[str] = [
            f"[Source]: {doc.metadata['source']}\n[Content]: {doc.page_content}\n{delimiter}"
            for doc in documents
        ]
    except KeyError:
        docs = [
            f"[Source]: {doc.metadata['url']}\n[Content]: {doc.page_content}\n{delimiter}"
            for doc in documents
        ]
    formated_docs: str = "\n\n".join(docs)

    return formated_docs


type RetrieverFn = Callable[[str, str | None, int], Coroutine[Any, Any, list[Document]]]
retrieval_method_dicts: dict[str, RetrieverFn] = {
    RetrieverMethodType.VECTOR_SEARCH: avector_search_tool,
    RetrieverMethodType.KEYWORD_SEARCH: akeyword_search_tool,
    RetrieverMethodType.HYBRID_SEARCH: ahybrid_search_tool,
}


async def aretrieve_internal_documents(
    method: RetrieverMethodType | str,
    rewritten_queries: list[str],
    target_section: str | None,
    k: int,
) -> list[Document]:
    """Retrieve internal documents using the specified retrieval method.

    Parameters
    ----------
    method : RetrieverMethodType | str
        Retrieval method to use (`vector_search`, `keyword_search`, or `hybrid_search`).
    rewritten_queries : list[str]
        Query variations produced by the query rewriter for this step.
    target_section : str | None
        Target section filter for internal document search. Only applied when
        method is `vector_search` or `hybrid_search`; ignored for pure keyword search.
    k : int
        Number of top documents to retrieve per query before deduplication.

    Returns
    -------
    list[Document]
        List of unique retrieved documents across all query variations.

    Raises
    ------
    ValueError
        If the provided method is not supported.
    """

    method = (
        method
        if isinstance(method, RetrieverMethodType)
        else RetrieverMethodType(method)
    )
    retrieval_fn = retrieval_method_dicts.get(method)
    if retrieval_fn is None:
        raise ValueError(f"Unsupported retrieval method: {method}")

    tasks: list[Coroutine[Any, Any, list[Document]]] = [
        # Expected signature. Order: query, target_section, k is important!
        retrieval_fn(
            query,
            target_section,
            k,
        )
        for query in rewritten_queries
    ]
    all_docs: list[list[Document]] = await asyncio.gather(*tasks)
    # Flatten the docs
    retrieved_docs: list[Document] = [doc for sublist in all_docs for doc in sublist]  # type: ignore

    return deduplicate(documents=retrieved_docs)


async def query_rewriter(question: str, search_keywords: list[str]) -> ReWrittenQuery:
    """Re-write the user's question into multiple query variations."""
    prompt = prompt_builder.query_rewriter_prompt(
        question=question, search_keywords=", ".join(search_keywords)
    )
    messages = convert_langchain_messages_to_dicts(messages=[HumanMessage(prompt)])
    response = await get_structured_output(
        messages=messages, model=None, schema=ReWrittenQuery
    )
    return cast(ReWrittenQuery, response)


async def determine_retrieval_type(question: str) -> RetrieverMethod:
    """Determine the optimal retrieval method for the given question."""
    prompt = prompt_builder.retriever_type_prompt(question=question)
    messages = convert_langchain_messages_to_dicts(messages=[HumanMessage(prompt)])
    response = await get_structured_output(
        messages=messages, model=None, schema=RetrieverMethod
    )
    return cast(RetrieverMethod, response)


def convert_context_to_str(step_state: list["StepState"]) -> str:
    """This function converts the list of StepState dictionaries into a single string.

    Parameters
    ----------
    step_state : list[StepState]
        The list of StepState dictionaries representing the research history.

    Returns
    -------
    str
        A single string representation of the research history.
    """
    return "\n\n".join(
        [
            f"Step {s['step_index']}: {s['question']}\nSummary: {s['summary']}"
            for s in step_state
        ]
    )


def format_plan(plan: Plan | None) -> str:
    """Format the plan into a string representation.

    Parameters
    ----------
    plan : Plan
        The multi-step plan to be formatted.

    Returns
    -------
    str
        A string representation of the plan.
    """
    if plan is None:
        return ""
    return json.dumps([step.model_dump() for step in plan.steps])


async def get_decision(question: str, plan: Plan | None, history: str) -> Decision:
    """This node is used to determine whether to continue with the plan or finish.

    Parameters
    ----------
    question : str
        The original user question.
    plan : Plan
        The multi-step plan object.
    history : str
        The history of completed steps.

    Returns
    -------
    Decision
        The decision object containing the next action and rationale.
    """
    sys_msg = prompt_builder.decision_prompt(
        question=question, plan=format_plan(plan=plan)
    )
    history_query: str = f"<COMPLETED_STEPS>{history}</COMPLETED_STEPS>"

    messages: list[dict[str, str]] = convert_langchain_messages_to_dicts(
        messages=[SystemMessage(sys_msg), HumanMessage(history_query)]
    )
    response = await get_structured_output(
        messages=messages, model=None, schema=Decision
    )
    return cast(Decision, response)


async def rerank_retrieved_documents(state: "State") -> dict[str, Any]:
    """Rerank documents by relevance to query."""
    k: int = 3
    question: str = state["original_question"]
    retrieved_documents: list[Document] = state["retrieved_documents"]
    # Get the details of the current step
    current_step_idx: int = state["current_step_index"]
    current_step: Step = state["plan"].steps[current_step_idx]
    logger.info(
        f"Retrieving documents for reranking for Step {current_step_idx}: {current_step.question}"
    )

    reranked_docs: list[Document] = await arerank_documents(
        query=question, documents=retrieved_documents, k=k
    )

    return {"reranked_documents": reranked_docs}


def format_generated_content(node: str, data: dict[str, Any]) -> str | None:
    """Format the generated content based on the node type.

    Parameters
    ----------
    node : str
        The type of node generating the content.
    data : dict[str, Any]
        The data produced by the node.

    Returns
    -------
    str | None
        A formatted string representation of the generated content.
    """
    try:
        if node == EventsType.VALIDATE_QUERY.value:
            query: str = data[node]["step_state"][-1]["question"]
            is_related_to_context: bool = data[node]["is_related_to_context"]
            return json.dumps(
                {"query": query, "is_related_to_context": is_related_to_context}
            )

        if node == EventsType.GENERATE_PLAN.value:
            if data[node] is None:
                return "No plan generated."
            return format_plan(Plan.model_validate(data[node]["plan"]))

        if node in [
            EventsType.INTERNET_SEARCH.value,
            EventsType.RETRIEVE_INTERNAL_DOCS.value,
        ]:
            _data = data[node]["step_state"][-1]
            re_written_queries = _data["rewritten_queries"]
            num_retrieved_docs = len(_data["reranked_documents"])
            sources = [
                doc.metadata.get("source", "unknown")
                for doc in _data["reranked_documents"]
            ]
            return json.dumps(
                {
                    "re_written_queries": re_written_queries,
                    "num_retrieved_docs": num_retrieved_docs,
                    "sources": sources,
                }
            )

        if node == EventsType.COMPRESS_DOCUMENTS.value and data[node]:
            return data[node]["synthesized_context"]

        if node == EventsType.REFLECT.value and data[node]:
            current_step: int = data[node]["step_state"][-1]["step_index"]
            summary: str = data[node]["step_state"][-1]["summary"]
            return json.dumps({"current_step": current_step, "summary": summary})

        if node == EventsType.FINAL_ANSWER.value and data[node]:
            return data[node]["final_answer"]

        if node == EventsType.UPDATE_LT_MEMORY.value:
            return "Long-term memory updated successfully."

        if node == EventsType.OVERALL_CONVO_SUMMARIZATION.value:
            return "Conversation summarized successfully."

    except KeyError as e:
        print(f"KeyError in format_generated_content: {e}")

    return None  # Default return if no conditions met


def log_system_info() -> None:
    """Log GPU information if available."""
    # Only import torch if needed
    import torch

    # Initialize memory variables
    allocated_gb = 0.0
    reserved_gb = 0.0

    # Check CUDA GPU
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        allocated_gb = torch.cuda.memory_allocated() / (1024**3)
        reserved_gb = torch.cuda.memory_reserved() / (1024**3)

        logger.info(
            f"🔥 CUDA GPU Device: {device_name} ({total_memory_gb:.2f} GB total)"
        )
        logger.info(
            f"   Memory - Allocated: {allocated_gb:.2f}GB, Reserved: {reserved_gb:.2f}GB"
        )
        return

    # Fallback to CPU
    import psutil

    cpu_count = os.cpu_count() or 0
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    total_memory_gb = memory.total / (1024**3)
    available_memory_gb = memory.available / (1024**3)
    used_memory_gb = memory.used / (1024**3)
    memory_percent = memory.percent

    logger.info("🚨 Using CPU for model inference (no GPU acceleration)")
    logger.info(f"💻 CPU Info: {cpu_count} cores, Usage: {cpu_percent}%")
    logger.info(
        f"   Memory - Total: {total_memory_gb:.2f}GB, Used: {used_memory_gb:.2f}GB, "
        f"Available: {available_memory_gb:.2f}GB ({memory_percent}% used)"
    )
