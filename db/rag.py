# rag.py
import os
from typing import TypedDict
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from openai import OpenAI

load_dotenv()

COLLECTION_NAME = "paper_chunks_local"
PERSIST_DIR = r"C:\Users\test6\Documents\chromadb_local"
TOP_K = 5

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = Chroma(
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
    persist_directory=PERSIST_DIR
)

client = OpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com"
)

class RAGResult(TypedDict):
    answer: str
    sources: list[dict]  # each: {"text": str, "source": str}

def rag_query_with_sources(query: str) -> RAGResult:
    results = vector_store.similarity_search(query=query, k=TOP_K)
    context = "\n\n".join(doc.page_content for doc in results)
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "You are a research assistant. Answer the user's question using only the provided context. If the context does not contain enough information, say so."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ]
    )
    return RAGResult(
        answer=response.choices[0].message.content,
        sources=[{"text": doc.page_content, "source": doc.metadata.get("source", "unknown")} for doc in results]
    )


def rag_query(query: str) -> str:
    # Retrieve relevant chunks
    results = vector_store.similarity_search(query=query, k=TOP_K)
    context = "\n\n".join(doc.page_content for doc in results)

    # Call DeepSeek
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research assistant. Answer the user's question "
                    "using only the provided context. If the context does not "
                    "contain enough information, say so."
                )
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion: {query}"
            }
        ]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    query = input("Enter your question: ")
    answer = rag_query(query)
    print("\nAnswer:\n", answer)
