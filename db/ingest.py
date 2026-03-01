# ingest.py
import os
import json
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Folders & collection names
CHUNKS_DIR = "../chunking/data/chunks"  # where your JSON chunk files are
COLLECTION_NAME = "paper_chunks_local"  # new local collection
PERSIST_DIR = r"C:\Users\test6\Documents\chromadb_local"  # folder to store vector DB

# Initialize local embedding function (offline, free)
embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create or load the collection
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

# Add to Chroma vector store
vector_store.add_documents(documents=documents, ids=ids)

print(f"✅ Ingested {len(documents)} chunks into collection '{COLLECTION_NAME}'")
print(f"Stored in folder: {PERSIST_DIR}")
