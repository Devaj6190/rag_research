# ingest.py
import os
import json
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Folders & collection names
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS_DIR = os.path.join(BASE_DIR, "chunking", "data", "chunks")
COLLECTION_NAME = "paper_chunks_local"
PERSIST_DIR = os.path.join(BASE_DIR, "chromadb_data")

# Initialize local embedding function (offline, free)
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Wipe and recreate the collection for a clean ingest every time
import chromadb
raw_client = chromadb.PersistentClient(path=PERSIST_DIR)
try:
    raw_client.delete_collection(COLLECTION_NAME)
    print(f"Cleared existing collection '{COLLECTION_NAME}'")
except Exception:
    pass

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
    persist_directory=PERSIST_DIR
)

# Load all JSON chunks
documents = []
ids = []

for filename in os.listdir(CHUNKS_DIR):
    if not filename.endswith(".json"):
        continue
    filepath = os.path.join(CHUNKS_DIR, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        chunks = json.load(f)
        for chunk in chunks:
            doc = Document(
                page_content=chunk["text"],
                metadata=chunk.get("metadata", {})
            )
            documents.append(doc)
            ids.append(chunk["chunk_id"])

# Add to Chroma vector store in batches
BATCH_SIZE = 500
for i in range(0, len(documents), BATCH_SIZE):
    batch_docs = documents[i:i + BATCH_SIZE]
    batch_ids = ids[i:i + BATCH_SIZE]
    vector_store.add_documents(documents=batch_docs, ids=batch_ids)
    print(f"  Ingested {min(i + BATCH_SIZE, len(documents))}/{len(documents)} chunks...")

print(f"Ingested {len(documents)} chunks into collection '{COLLECTION_NAME}'")
print(f"Stored in folder: {PERSIST_DIR}")
