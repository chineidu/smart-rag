import asyncio

from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src import create_logger
from src.schemas.types import FileFormatsType
from src.utilities.utils import (
    chunk_data_by_sections,
    extract_10k_sections,
    load_all_documents,
)

logger = create_logger("vectorstores")


class VectorStoreSetup:
    def __init__(self) -> None:
        self._vectorstore: QdrantVectorStore | None = None
        self._initialized: bool = False
        self._documents: list[Document] | None = None
        # Use an asyncio.Lock to avoid blocking the event loop during async initialization
        self._lock: asyncio.Lock = asyncio.Lock()

    def get_vectorstore(self) -> QdrantVectorStore | None:
        """Return the cached vector store instance if initialized.

        Returns
        -------
        QdrantVectorStore | None
            The cached vector store instance or None if not yet set up.
        """
        return self._vectorstore

    def get_documents(self) -> list[Document] | None:
        """Return the cached documents if available.

        Returns
        -------
        list[Document] | None
            The cached list of Document objects or None if not yet set.
        """
        return self._documents

    def chunk_documents(
        self,
        documents: list[Document],
        source: str | None = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 100,
        add_start_index: bool = True,
        split_by_sections: bool = False,
    ) -> list[Document]:
        """
        Split a list of Document objects into smaller character-based chunks.

        Parameters
        ----------
        documents : list[Document]
            List of documents to split. Each item should be a Document containing textual content
            (for example via a `page_content` or `text` attribute). Original document metadata is
            preserved and propagated to resulting chunks.
        source : str
            Source identifier to be used in section-based chunking.
        chunk_size : int, default=1000
            Maximum number of characters per chunk.
        chunk_overlap : int, default=100
            Number of overlapping characters between consecutive chunks.
        add_start_index : bool, default=True
            Whether to add a `start_index` field in metadata to track original position.
        split_by_sections : bool, default=False
            If True, split documents based on extracted sections; otherwise, use character-based splitting.

        Returns
        -------
        list[Document]
        """
        if split_by_sections:
            if source is None:
                raise ValueError(
                    "Source must be provided when split_by_sections is True."
                )

            section_titles, section_content = extract_10k_sections(documents)
            return chunk_data_by_sections(
                source=source,
                section_titles=section_titles,
                section_content=section_content,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                add_start_index=add_start_index,
            )

        return RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,  # chunk size (characters)
            chunk_overlap=chunk_overlap,  # chunk overlap (characters)
            add_start_index=add_start_index,  # track index in original document
        ).split_documents(documents)

    async def asetup_vectorstore(
        self,
        filepaths: str | list[str],
        jq_schema: str | None,
        format: FileFormatsType | str,
        embedding_model: Embeddings,
        client: QdrantClient,
        collection: str,
        filepaths_is_glob: bool = False,
        force_recreate: bool = False,
    ) -> QdrantVectorStore | None:
        """Set up a Qdrant vector store with embedded documents.

        Parameters
        ----------
        filepaths : str or list[str]
            Path to a single file or a list of paths.
        jq_schema : str or None
            JQ schema to apply when loading documents.
        format : FileFormatsType | str
            Format of the files to be loaded.
        embedding_model : Embeddings
            Embeddings implementation used for document encoding.
        client : QdrantClient
            Qdrant client instance for collection management.
        collection : str
            Name of the Qdrant collection to use or create.
        filepaths_is_glob : bool, optional
            If True, treat `filepaths` as a glob pattern to match multiple files, by default False
        Returns
        -------
        QdrantVectorStore
            Qdrant vector store instance with embedded documents.
        """
        # Import here to avoid circular dependencies
        from src.utilities.tools.helpers import _normalize_data

        try:
            # If already initialized and cached, reuse
            if self._initialized and self._vectorstore:
                return self._vectorstore

            async with self._lock:  # Ensure coroutine-safe initialization
                # Load and embed documents, then cache the vector store
                logger.info(f"Loading documents from {filepaths} with format {format}")
                docs: list[Document] = load_all_documents(
                    filepaths=filepaths,
                    jq_schema=jq_schema,
                    format=format,
                    filepaths_is_glob=filepaths_is_glob,
                )
                chunked_docs: list[Document] = self.chunk_documents(
                    documents=docs,
                    source="Nvidia_10K_Filings",
                    split_by_sections=True,
                )
                normalized_docs: list[Document] = _normalize_data(chunked_docs)
                sample_text: str = "sample text"
                sample_embedding: list[
                    list[float]
                ] = await embedding_model.aembed_documents([sample_text])
                vector_size: int = len(sample_embedding[0])

                if force_recreate and client.collection_exists(collection):
                    client.delete_collection(collection_name=collection)

                if client.collection_exists(collection):
                    logger.info(
                        f"Collection {collection!r} already exists. Skipping creation."
                    )
                    vectorstore = QdrantVectorStore(
                        client=client,
                        collection_name=collection,
                        embedding=embedding_model,
                    )

                else:
                    client.create_collection(
                        collection_name=collection,
                        vectors_config=VectorParams(
                            size=vector_size, distance=Distance.COSINE
                        ),
                    )
                    vectorstore = QdrantVectorStore(
                        client=client,
                        collection_name=collection,
                        embedding=embedding_model,
                    )
                    logger.info(
                        f"Creating collection {collection!r} with vector size {vector_size}.."
                    )
                    _ = await vectorstore.aadd_documents(documents=chunked_docs)
                    logger.info(
                        f"Qdrant vector store set up with collection {collection!r}"
                    )

                self._vectorstore = vectorstore
                # For keyword search and other uses
                self._documents = normalized_docs
                self._initialized = True

                return self._vectorstore

        except Exception as e:
            logger.error(f"Error setting up vectorstore: {e}", exc_info=True)
            return None

    def is_ready(self) -> bool:
        """Check if the vector store setup is complete.

        Returns
        -------
        bool
            True if the vector store is initialized; False otherwise.
        """
        return self._initialized and self._vectorstore is not None

    async def close(self) -> None:
        """Asynchronously close and clean up vector store resources.

        Use this method to ensure the close runs under the same async lock that guards
        initialization to avoid race conditions.
        """
        async with self._lock:
            if self._initialized:
                logger.info("📝 Closing vector store resources...")
                self._vectorstore = None
                self._initialized = False
