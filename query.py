# query.py
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Use same collection/folder as ingest
COLLECTION_NAME = "paper_chunks_local"
PERSIST_DIR = r"C:\Users\tanma\Documents\chromadb_local"

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
    persist_directory=PERSIST_DIR
)

# Example query
query = "Does timing of cranioplasty affect hydrocephalus?"
results = vector_store.similarity_search(query=query, k=5)

print("\n🔍 Query:", query)
print("\n📄 Top Results:\n")

for i, doc in enumerate(results, 1):
    print(f"Result {i}:")
    print(doc.page_content[:400], "...\n")
    print("Metadata:", doc.metadata)
    print("-" * 80)
