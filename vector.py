import os
import logging
import pandas as pd
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

# -------------------- Setup logging --------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Setting up local RAG with Ollama and Chroma")

# -------------------- Configuration --------------------
embedding_model_name = "embeddinggemma"
chroma_collection_name = "restaurant_reviews"
csv_file_path = "./realistic_restaurant_reviews.csv"
db_location = "./chroma_langchain_db"
logger.info(f"Using embedding model: {embedding_model_name}")

# -------------------- Initialize Ollama Embeddings --------------------
logger.info("Initializing Ollama Embeddings")
try:
    embedding_function_ollama = OllamaEmbeddings(model=embedding_model_name)
    logger.info("Ollama Embeddings initialized successfully")
except Exception as e:
    logger.exception("Failed to initialize Ollama Embeddings")
    raise e

# -------------------- Initialize Chroma Vector Store --------------------
logger.info("Initializing Chroma vector store")
try:
    vector_store = Chroma(
        collection_name=chroma_collection_name,
        embedding_function=embedding_function_ollama,
        persist_directory=db_location,
        relevance_score_fn=lambda d: 1 - d, 
    )
    logger.info("Chroma vector store initialized successfully")
except Exception as e:
    logger.exception("Failed to initialize Chroma vector store")
    raise e


# -------------------- Load Data from CSV --------------------
def load_documents_from_csv(csv_path: str):
    if not os.path.exists(csv_path):
        logger.error(f"CSV file not found at path: {csv_path}")
        raise FileNotFoundError(f"CSV file not found at path: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = ["Title", "Review", "Rating", "Date"]
    for col in required_columns:
        if col not in df.columns:
            logger.error(f"CSV is missing required column: {col}")
            raise ValueError(f"CSV is missing required column: {col}")

    logger.info(f"Data loaded successfully with shape: {df.shape}")

    documents = []
    ids = []
    for i, row in df.iterrows():
        doc_content = f"{row['Title']} {row['Review']}"
        logger.debug(f"Creating Document {i}: {doc_content[:50]}...")  # first 50 chars
        doc = Document(
            page_content=doc_content,
            metadata={"rating": row["Rating"], "date": row["Date"]},
            id=str(i),
        )
        documents.append(doc)
        ids.append(str(i))

    logger.info(f"Total documents created: {len(documents)}")
    return documents, ids


# -------------------- Add documents to vector store if not already persisted --------------------
add_documents = not os.path.exists(db_location) or not os.listdir(db_location)
if add_documents:
    logger.info("Adding data to Chroma vector store from CSV")
    documents, ids = load_documents_from_csv(csv_file_path)
    logger.info(f"Adding {len(documents)} documents to vector store")
    logger.debug(
        f"Sample Document: {documents[0].page_content[:100]}..."
    )  # first 100 chars
    try:
        vector_store.add_documents(documents=documents, ids=ids)
        vector_store.persist()  # ensure data is saved
        logger.info("Data added and persisted to Chroma vector store successfully")
    except Exception as e:
        logger.exception("Failed to add documents to Chroma vector store")
        raise e
else:
    logger.info("Vector store already exists. Skipping document addition.")

# -------------------- Setup Retriever --------------------
retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 2, "score_threshold": -0.2},
)
logger.info("Retriever initialized successfully. Ready for queries.")
