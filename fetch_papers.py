#!/usr/bin/env python3
"""
fetch_papers.py — search PubMed by keywords, fetch full-text XML from PMC,
extract body text, and write chunk JSON files ready for db/ingest.py.

Bypasses PDF entirely — uses the PMC efetch API which is reliable and free.

Usage:
    python fetch_papers.py [--target 100] [--keywords assets/keywords_final.txt]
"""

import argparse
import json
import os
import re
import time
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter

KEYWORDS_FILE = "assets/keywords_final.txt"
CHUNKS_DIR = "chunking/data/chunks"
NCBI_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
RATE_DELAY = 0.4  # NCBI allows 3 req/sec without an API key

CHUNK_SIZE = 500
CHUNK_OVERLAP = 75

JUNK_PREFIXES = ("here ", "or clinical", "symptoms", "diagnoses")


def load_centrality_scores(scores_path: str) -> dict[str, float]:
    """Load pre-computed co-occurrence centrality scores if available."""
    if not os.path.exists(scores_path):
        return {}
    with open(scores_path, encoding="utf-8") as f:
        return json.load(f).get("scores", {})


def load_keywords(path: str) -> list[tuple[str, int]]:
    """
    Returns list of (keyword, frequency) tuples sorted by frequency descending.
    Frequency = number of paragraphs the keyword appears in.
    """
    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Count frequency from paragraph sections (before the unique keywords list)
    para_section = content.split("=== ALL UNIQUE KEYWORDS ===")[0] if "=== ALL UNIQUE KEYWORDS ===" in content else content
    freq: dict[str, int] = {}
    for para in re.split(r"--- Paragraph \d+ ---", para_section):
        para_lower = para.lower()
        for kw in re.split(r"[,\n]", para):
            kw = kw.strip().rstrip(".,").strip().lower()
            if kw:
                freq[kw] = freq.get(kw, 0) + 1

    # Build valid keyword list from the deduplicated section
    if "=== ALL UNIQUE KEYWORDS ===" in content:
        section = content.split("=== ALL UNIQUE KEYWORDS ===")[1]
        raw_lines = [ln.strip().lstrip("-").strip() for ln in section.splitlines()]
    else:
        raw_lines = []
        for ln in content.splitlines():
            ln = ln.strip()
            if ln and not ln.startswith("---") and not ln.startswith("==="):
                for kw in ln.split(","):
                    raw_lines.append(kw.strip())

    seen: set[str] = set()
    valid: list[tuple[str, int]] = []
    skipped_count = 0

    for kw in raw_lines:
        kw = kw.strip().rstrip(".,").strip()
        if not kw:
            continue
        kw_lower = kw.lower()
        if kw_lower in seen:
            continue
        seen.add(kw_lower)

        if len(kw) < 4:
            skipped_count += 1
            continue
        if re.fullmatch(r"[\d\s%/\.\-,]+", kw):
            skipped_count += 1
            continue
        if any(kw_lower.startswith(p) for p in JUNK_PREFIXES):
            skipped_count += 1
            continue
        if kw.endswith(":"):
            skipped_count += 1
            continue
        if kw_lower.startswith("upmc "):
            skipped_count += 1
            continue

        valid.append((kw, freq.get(kw_lower, 1)))

    # Sort by frequency descending so repeated keywords are searched first
    valid.sort(key=lambda x: x[1], reverse=True)

    print(f"Keywords loaded  : {len(valid)}")
    print(f"Filtered out     : {skipped_count} (noise/non-searchable terms)")
    print(f"Top 5 by freq    : {[(k, f) for k, f in valid[:5]]}")
    return valid


def search_pubmed(keyword: str, max_results: int = 10) -> list[str]:
    """Return PMIDs matching keyword that have free full text in PMC."""
    query = f'"{keyword}" AND ("traumatic brain injury" OR "TBI" OR "brain injury" OR "diffuse axonal injury") AND "free full text"[filter]'
    try:
        resp = requests.get(
            f"{NCBI_BASE}/esearch.fcgi",
            params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"},
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"    [search error] {e}")
        return []


def pmid_to_pmcid(pmid: str) -> str | None:
    """Convert a PMID to a PMCID via elink."""
    try:
        resp = requests.get(
            f"{NCBI_BASE}/elink.fcgi",
            params={"dbfrom": "pubmed", "db": "pmc", "id": pmid, "retmode": "json"},
            timeout=15,
        )
        resp.raise_for_status()
        for ls in resp.json().get("linksets", []):
            for lsdb in ls.get("linksetdbs", []):
                if lsdb.get("dbto") == "pmc":
                    ids = lsdb.get("links", [])
                    if ids:
                        return str(ids[0])
    except Exception as e:
        print(f"    [elink error] PMID {pmid}: {e}")
    return None


def fetch_pmc_text(pmcid: str) -> str | None:
    """Fetch full-text article body from PMC efetch XML API."""
    try:
        resp = requests.get(
            f"{NCBI_BASE}/efetch.fcgi",
            params={"db": "pmc", "id": pmcid, "rettype": "xml", "retmode": "xml"},
            timeout=30,
        )
        resp.raise_for_status()
        xml = resp.text

        # Extract title
        title_match = re.search(r"<article-title>(.*?)</article-title>", xml, re.DOTALL)
        title = re.sub(r"<[^>]+>", "", title_match.group(1)).strip() if title_match else ""

        # Extract all paragraph text from the article body
        paragraphs = re.findall(r"<p\b[^>]*>(.*?)</p>", xml, re.DOTALL)
        body_parts = []
        if title:
            body_parts.append(title)
        for p in paragraphs:
            text = re.sub(r"<[^>]+>", " ", p)   # strip XML tags
            text = re.sub(r"\s+", " ", text).strip()
            if len(text) > 50:
                body_parts.append(text)

        if not body_parts:
            return None

        return "\n\n".join(body_parts)
    except Exception as e:
        print(f"    [efetch error] PMC{pmcid}: {e}")
        return None


def save_chunks(pmcid: str, text: str, chunks_dir: str) -> int:
    """Split text into chunks and write JSON. Returns number of chunks written."""
    stem = f"pmc_{pmcid}"
    filepath = os.path.join(chunks_dir, f"{stem}.json")
    if os.path.exists(filepath):
        return -1  # already processed

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    chunks_text = splitter.split_text(text)
    chunks = [
        {"chunk_id": f"{stem}_{i}", "text": c, "metadata": {"source": f"{stem}.txt"}}
        for i, c in enumerate(chunks_text)
        if len(c.strip()) > 50
    ]
    if not chunks:
        return 0
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    return len(chunks)


def count_chunks(chunks_dir: str) -> int:
    return len([f for f in os.listdir(chunks_dir) if f.endswith(".json") and f.startswith("pmc_")])


def main():
    parser = argparse.ArgumentParser(description="Fetch PubMed papers and build chunk JSON files")
    parser.add_argument("--target", type=int, default=100, help="Target number of new PMC articles (default: 100)")
    parser.add_argument("--keywords", default=KEYWORDS_FILE, help="Path to keywords txt file")
    args = parser.parse_args()

    os.makedirs(CHUNKS_DIR, exist_ok=True)

    centrality = load_centrality_scores("assets/keyword_scores.json")
    if centrality:
        print("Using co-occurrence centrality scores from keyword_graph.py")
    else:
        print("No centrality scores found — using raw frequency. Run keyword_graph.py first for better results.")

    print(f"\nLoading keywords from: {args.keywords}")
    keywords = load_keywords(args.keywords)
    print()

    existing = count_chunks(CHUNKS_DIR)
    needed = max(0, args.target - existing)
    print(f"PMC articles chunked : {existing}")
    print(f"Target               : {args.target}")
    print(f"Need to fetch        : {needed}\n")

    if needed == 0:
        print("Target already reached. Use --target to raise it.")
        return

    # Phase 1: scan ALL keywords and collect PMIDs
    # Repeated keywords get higher retmax so more papers are pulled for them
    print("Phase 1: Scanning all keywords...\n")
    all_pmids: list[str] = []  # ordered, duplicates removed, higher-freq keywords first
    seen_pmids: set[str] = set()

    for i, (keyword, freq) in enumerate(keywords):
        score = centrality.get(keyword.lower(), 0)
        if centrality:
            # centrality score 1.0 → retmax 25, 0.5 → retmax 13, 0.0 → retmax 3
            # Capped at 25 so even top-scoring generic symptoms don't dominate
            retmax = max(3, min(25, round(score * 25)))
        else:
            retmax = min(5 + (freq - 1) * 10, 50)
        print(f"[{i+1}/{len(keywords)}] {keyword!r} (freq={freq}, retmax={retmax})")
        pmids = search_pubmed(keyword, max_results=retmax)
        time.sleep(RATE_DELAY)
        for pmid in pmids:
            if pmid not in seen_pmids:
                seen_pmids.add(pmid)
                all_pmids.append(pmid)

    print(f"\nPhase 1 done. Unique PMIDs collected: {len(all_pmids)}")
    print(f"\nPhase 2: Downloading up to {args.target} articles...\n")

    # Phase 2: download from collected PMIDs until target reached
    fetched = skipped = failed = 0

    for pmid in all_pmids:
        if count_chunks(CHUNKS_DIR) >= args.target:
            print(f"\nTarget of {args.target} articles reached.")
            break

        pmcid = pmid_to_pmcid(pmid)
        time.sleep(RATE_DELAY)
        if not pmcid:
            failed += 1
            continue

        if os.path.exists(os.path.join(CHUNKS_DIR, f"pmc_{pmcid}.json")):
            skipped += 1
            continue

        text = fetch_pmc_text(pmcid)
        time.sleep(RATE_DELAY)
        if not text:
            failed += 1
            continue

        n = save_chunks(pmcid, text, CHUNKS_DIR)
        if n > 0:
            fetched += 1
            print(f"  + PMC{pmcid}  ({n} chunks)  total: {count_chunks(CHUNKS_DIR)}")
        elif n == -1:
            skipped += 1
        else:
            failed += 1

    final = count_chunks(CHUNKS_DIR)
    print(f"\n{'='*40}")
    print(f"Keywords scanned   : {len(keywords)}/{len(keywords)}")
    print(f"Unique PMIDs found : {len(all_pmids)}")
    print(f"Articles fetched   : {fetched}")
    print(f"Already existed    : {skipped}")
    print(f"Failed/no text     : {failed}")
    print(f"PMC chunks now     : {final}")
    if final < args.target:
        print(f"\nReached {final}/{args.target}. Run again or raise --target.")
    else:
        print(f"\nDone. Run python db/ingest.py to index the new chunks.")


if __name__ == "__main__":
    main()
