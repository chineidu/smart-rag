import json
import re
import unicodedata
from glob import glob
from pathlib import Path
from re import Match, Pattern
from typing import Any, Callable
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

from src.schemas.types import FileFormatsType
from src.utilities.client import HTTPXClient, get_instructor_openrouter_client
from src.utilities.model_config import RemoteModel

type PydanticModel = type[BaseModel]


def merge_dicts(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    """Merge two dictionaries, with new values updating existing ones."""
    if existing is None:
        return new
    # Update existing dict with new dict values
    return {**existing, **new}


@traceable
async def get_structured_output(
    messages: list[dict[str, Any]],
    model: RemoteModel,
    schema: PydanticModel,
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
        List of Document chunks with metadata including source_doc, section, and chunk_id.
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
                        "source_doc": source,  # original document path
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
    format: FileFormatsType,
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
    format : FileFormatsType
        Format of the files to be loaded.
    filepaths_is_glob : bool, default=False
        If True, treat `filepaths` as a glob pattern to match multiple files.

    Returns
    -------
    list[Document]
        List of loaded Document objects.
    """
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
