from glob import glob

from langchain_core.documents.base import Document
from langchain_core.embeddings import Embeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src import PACKAGE_PATH, create_logger
from src.config import app_config, app_settings
from src.utilities.embeddings import TogetherEmbeddings
from src.utilities.utils import load_pdf_doc

logger = create_logger("vectorstores")


def text_split_documents(docs: list[Document]) -> list[Document]:
    """
    Split a list of Document objects into smaller character-based chunks.

    Parameters
    ----------
    docs : list[Document]
        List of documents to split. Each item should be a Document containing textual content
        (for example via a `page_content` or `text` attribute). Original document metadata is
        preserved and propagated to resulting chunks.

    Returns
    -------
    list[Document]
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # chunk size (characters)
        chunk_overlap=50,  # chunk overlap (characters)
        add_start_index=True,  # track index in original document
    )
    return text_splitter.split_documents(docs)


def setup_vectorstore(
    filepaths: str | list[str],
    embeddings: Embeddings,
    client: QdrantClient,
    collection: str,
    filepaths_is_glob: bool = False,
) -> tuple[QdrantVectorStore, list[Document]]:
    """
    Set up a Qdrant vector store from one or more PDF filepaths using the provided embeddings and client.

    Parameters
    ----------
    filepaths : str or list of str
        Path to a single PDF file or a list of paths. If a single string is provided, it will be
        wrapped into a list internally. Each filepath is processed by `load_pdf_doc` to produce
        document objects to be indexed.
    embeddings : Embeddings
        Embeddings implementation used to compute vector dimensionality and to encode documents
        for storage. The dimensionality is inferred by calling `embeddings.embed_query("sample text")`.
    client : QdrantClient
        Qdrant client instance used to query and create collections. If the specified collection
        does not exist, a collection will be created on the client with vector parameters derived
        from the embeddings.
    collection : str
        Name of the Qdrant collection to use or create.
    filepaths_is_glob : bool, default=False
        If True, treat `filepaths` as a glob pattern to match multiple PDF files.

    Returns
    -------
    tuple[QdrantVectorStore, list[Document]]
    """
    if filepaths_is_glob:
        if isinstance(filepaths, list):
            # If already a list, glob each path
            all_files = []
            for fp in filepaths:
                all_files.extend(glob(f"{fp}/*.pdf"))
            filepaths = all_files
        else:
            # If single string, glob it directly
            filepaths = glob(f"{filepaths}/*.pdf")
    elif isinstance(filepaths, str):
        filepaths = [filepaths]

    docs: list[Document] = [
        doc for fp in filepaths for doc in load_pdf_doc(filepath=fp)
    ]
    logger.info(f"Loaded {len(docs)} documents from {len(filepaths)} filepaths.")

    vector_size: int = len(embeddings.embed_query("sample text"))

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    vectorstore = QdrantVectorStore(
        client=client, collection_name=collection, embedding=embeddings
    )
    logger.info(
        f"Qdrant vector store set up with collection {collection!r} and vector size {vector_size}"
    )
    return (vectorstore, docs)


def embed_and_store_documents(
    vectorstore: QdrantVectorStore, docs: list[Document]
) -> QdrantVectorStore:
    """
    Embed and store documents in the vector store.

    Parameters
    ----------
    vectorstore : QdrantVectorStore
        The vector store instance to add documents to (modified in-place).
    docs : list[Document]
        Documents to embed and store.

    Returns
    -------
    QdrantVectorStore
        The same vectorstore instance (for method chaining).
    """
    vectorstore.add_documents(documents=docs)
    logger.info(f"Embedded and stored {len(docs)} documents.")
    return vectorstore


class VectorStoreSetup:
    def __init__(
        self,
        collection_name: str,
        filepaths: str | list[str],
        embeddings: Embeddings,
        client: QdrantClient,
        filepaths_is_glob: bool = False,
    ) -> None:
        """Class to set up and manage a Qdrant vector store with document embedding and storage.

        Parameters
        ----------
        collection_name : str
            Name of the Qdrant collection to use or create.
        filepaths : str or list of str
            Path to a single PDF file or a list of paths.
        embeddings : Embeddings
            Embeddings implementation used for document encoding.
        client : QdrantClient
            Qdrant client instance for collection management.
        filepaths_is_glob : bool, default=False
            If True, treat `filepaths` as a glob pattern to match multiple PDF files.
        """
        self.collection_name: str = collection_name
        self.filepaths: str | list[str] = filepaths
        self.embeddings: Embeddings = embeddings
        self.client: QdrantClient = client
        self.filepaths_is_glob: bool = filepaths_is_glob
        self._docs: list[Document] | None = None
        self._vectorstore: QdrantVectorStore | None = None

    def setup(self) -> None:
        """Set up the vector store and load documents."""
        self._vectorstore, self._docs = setup_vectorstore(
            filepaths=self.filepaths,
            embeddings=self.embeddings,
            client=self.client,
            collection=self.collection_name,
            filepaths_is_glob=self.filepaths_is_glob,
        )

    def embed_and_store(self) -> QdrantVectorStore:
        """Embed and store documents in the vector store."""
        if self._docs is None:
            raise ValueError("Documents are not loaded. Call setup() first.")
        split_docs = text_split_documents(docs=self._docs)

        if self._vectorstore is None:
            raise ValueError("Vector store is not set up. Call setup() first.")

        return embed_and_store_documents(self._vectorstore, split_docs)

    def get_vectorstore(self) -> QdrantVectorStore:
        """Get the Qdrant vector store instance."""
        if self._vectorstore is None:
            raise ValueError("Vector store is not set up. Call setup() first.")
        return self._vectorstore


def initialize_vectorstores() -> tuple[QdrantVectorStore, QdrantVectorStore]:
    """
    Initialize AI news and football news vector stores.

    Returns
    -------
    tuple[QdrantVectorStore, QdrantVectorStore]
        A tuple containing (ai_vectorstore, football_vectorstore).

    Raises
    ------
    Exception
        If vectorstore initialization fails for either collection.
    """
    embeddings = TogetherEmbeddings(
        together_api_key=app_settings.TOGETHER_API_KEY,
        model=app_config.llm_model_config.embedding_model.model_name,
    )
    # TODO: Replace with actual Qdrant client connection parameters
    # For demonstration, using in-memory Qdrant client
    client = QdrantClient(":memory:")

    # AI news vector store setup
    filepath: str = str(
        PACKAGE_PATH / app_config.vectorstore_config.ai_config.filepaths
    )
    ai_news_vectorstore_setup = VectorStoreSetup(
        collection_name="ai_news",
        filepaths=filepath,
        embeddings=embeddings,
        client=client,
        filepaths_is_glob=True,
    )
    ai_news_vectorstore_setup.setup()
    vectorstore_ai: QdrantVectorStore = ai_news_vectorstore_setup.embed_and_store()
    logger.info("AI news vector store setup complete.")

    # Football news vector store setup
    filepath = str(
        PACKAGE_PATH / app_config.vectorstore_config.football_config.filepaths
    )
    football_news_vectorstore_setup = VectorStoreSetup(
        collection_name="football_news",
        filepaths=filepath,
        embeddings=embeddings,
        client=client,
        filepaths_is_glob=True,
    )
    football_news_vectorstore_setup.setup()
    vectorstore_football: QdrantVectorStore = (
        football_news_vectorstore_setup.embed_and_store()
    )
    logger.info("Football news vector store setup complete.")

    return (vectorstore_ai, vectorstore_football)


# Module-level instances initialized on import
vectorstore_ai: QdrantVectorStore
vectorstore_football: QdrantVectorStore

try:
    vectorstore_ai, vectorstore_football = initialize_vectorstores()
except Exception as e:
    logger.error(f"Failed to initialize vector stores: {e}", exc_info=True)
    raise
