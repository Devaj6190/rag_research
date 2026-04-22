# RAG Research Assistant

A clinical research assistant powered by Retrieval-Augmented Generation (RAG). Upload or query against a curated database of TBI/DAI research papers to get specific, actionable clinical answers grounded in published literature.

## How it works

1. **Paper ingestion** — Research papers (PDFs or fetched from PubMed Central) are chunked into 500-character segments with overlap
2. **Embedding** — Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and stored in a ChromaDB vector database
3. **Retrieval** — On query, the top-5 most semantically similar chunks are retrieved
4. **Generation** — DeepSeek LLM generates a clinical answer grounded in the retrieved chunks, with source attribution

## Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI |
| Vector DB | ChromaDB |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, offline) |
| LLM | DeepSeek API |
| PDF parsing | Docling |
| Paper fetching | PubMed E-utilities + PMC efetch API |

## Setup

### 1. Clone and install

```bash
git clone https://github.com/Devaj6190/rag_research.git
cd rag_research
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux
pip install -r requirements.txt
```

### 2. Configure environment

Create a `.env` file in the project root:

```
DEEPSEEK_API_KEY=your_key_here
APP_USERNAME=your_username
APP_PASSWORD=your_password
```

Get a DeepSeek API key at [platform.deepseek.com](https://platform.deepseek.com).

### 3. Build the paper database

The vector database is not included in the repo — you build it locally:

```bash
# Generate keyword co-occurrence scores (determines which keywords pull more papers)
python keyword_graph.py

# Fetch papers from PubMed Central based on keywords in assets/keywords_final.txt
python fetch_papers.py --target 300

# Embed and store all chunks in ChromaDB
python db/ingest.py
```

`fetch_papers.py` searches PubMed using clinical keywords, fetches full-text XML from PMC for free open-access articles, and writes chunk JSON files to `chunking/data/chunks/`. It takes ~45-60 minutes for 300 articles.

You can also add your own PDFs manually:

```bash
# Place PDFs in chunking/data/raw_pdfs/, then:
python chunking/chunk.py
python db/ingest.py
```

### 4. Run the app

```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser and log in with the credentials from your `.env`.

## Project structure

```
rag_research/
├── app.py                        # FastAPI server + auth + file upload
├── fetch_papers.py               # PubMed keyword search + PMC full-text fetcher
├── keyword_graph.py              # Co-occurrence centrality analysis for keyword scoring
├── assets/
│   ├── keywords_final.txt        # Clinical keywords used for paper search
│   └── keyword_scores.json       # Pre-computed centrality scores (output of keyword_graph.py)
├── chunking/
│   ├── chunk.py                  # PDF → chunk JSON pipeline (uses Docling)
│   └── data/
│       ├── raw_pdfs/             # Manually added PDFs
│       └── chunks/               # Chunk JSON files (one per paper)
├── db/
│   ├── ingest.py                 # Load chunks → ChromaDB
│   ├── rag.py                    # Core RAG logic: retrieval + DeepSeek generation
│   └── query.py                  # CLI query tool
├── frontend/
│   ├── index.html                # Main UI
│   └── login.html                # Auth page
└── chromadb_data/                # Local vector DB (gitignored, built on first run)
```

## Features

- **Session-based auth** — login required, httponly cookie sessions
- **File upload** — attach a PDF or DOCX to any query for additional context
- **Source attribution** — every answer shows which paper chunks were used
- **Automated paper fetching** — keyword-driven PubMed search with co-occurrence scoring to prioritise clinically relevant papers
- **Clean re-ingest** — running `ingest.py` wipes and rebuilds the collection to prevent stale embeddings

## Re-fetching / expanding the database

To fetch more papers or reset:

```bash
# Remove existing PMC chunks
rm chunking/data/chunks/pmc_*.json   # Mac/Linux
del chunking\data\chunks\pmc_*.json  # Windows

# Re-fetch with higher target
python fetch_papers.py --target 500

# Re-ingest (automatically clears old DB first)
python db/ingest.py
```
