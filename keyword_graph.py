#!/usr/bin/env python3
"""
keyword_graph.py — analyse keyword co-occurrence centrality from the notes file.

Keywords that consistently appear with other condition-related keywords score high.
Generic symptoms (pain, anxiety) that appear scattered across unrelated paragraphs
score lower than core clinical terms (DAI, baclofen pump, cranioplasty).

Weighting: pairs in a short focused paragraph get more weight than pairs in a
long catch-all history paragraph (weight = 1 / paragraph_size).

Usage:
    python keyword_graph.py [--keywords assets/keywords_final.txt] [--top 50]
Outputs:
    assets/keyword_scores.json  — used by fetch_papers.py for retmax scaling
"""

import argparse
import json
import re
from collections import defaultdict

KEYWORDS_FILE = "assets/keywords_final.txt"
OUTPUT_FILE = "assets/keyword_scores.json"

JUNK_PREFIXES = ("here ", "or clinical", "symptoms", "diagnoses")


def is_valid(kw: str) -> bool:
    kw = kw.strip().rstrip(".,").strip()
    if len(kw) < 4:
        return False
    if re.fullmatch(r"[\d\s%/\.\-,]+", kw):
        return False
    if any(kw.lower().startswith(p) for p in JUNK_PREFIXES):
        return False
    if kw.endswith(":"):
        return False
    if kw.lower().startswith("upmc "):
        return False
    return True


def parse_paragraphs(content: str) -> list[list[str]]:
    """Extract keyword lists from each paragraph section."""
    para_section = content.split("=== ALL UNIQUE KEYWORDS ===")[0] if "=== ALL UNIQUE KEYWORDS ===" in content else content
    paragraphs = []
    for block in re.split(r"--- Paragraph \d+ ---", para_section):
        block = block.strip()
        if not block:
            continue
        # Skip LLM preamble lines
        lines = [ln for ln in block.splitlines() if ln.strip() and not ln.lower().startswith("here ")]
        text = " ".join(lines)
        kws = [kw.strip().rstrip(".,").strip().lower() for kw in text.split(",")]
        kws = [kw for kw in kws if is_valid(kw)]
        if len(kws) >= 2:
            paragraphs.append(kws)
    return paragraphs


def compute_centrality(paragraphs: list[list[str]]) -> dict[str, float]:
    """
    Weighted co-occurrence centrality.
    Each keyword's score = sum of (1/paragraph_size) for every paragraph it appears in,
    multiplied by how many other unique keywords it co-occurs with in that paragraph.
    Short focused paragraphs carry more weight than long catch-all history paragraphs.
    """
    scores: dict[str, float] = defaultdict(float)
    co_occurs: dict[str, set] = defaultdict(set)

    for kws in paragraphs:
        n = len(kws)
        weight = 1.0 / n
        for kw in kws:
            for other in kws:
                if other != kw:
                    co_occurs[kw].add(other)
                    scores[kw] += weight

    return dict(scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--keywords", default=KEYWORDS_FILE)
    parser.add_argument("--top", type=int, default=50, help="Print top N keywords")
    args = parser.parse_args()

    with open(args.keywords, encoding="utf-8") as f:
        content = f.read()

    paragraphs = parse_paragraphs(content)
    print(f"Paragraphs analysed: {len(paragraphs)}")

    scores = compute_centrality(paragraphs)

    # Normalise to 0-1
    max_score = max(scores.values()) if scores else 1
    normalised = {k: round(v / max_score, 4) for k, v in scores.items()}
    ranked = sorted(normalised.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop {args.top} keywords by co-occurrence centrality:\n")
    print(f"  {'Score':>6}  Keyword")
    print(f"  {'------':>6}  -------")
    for kw, score in ranked[:args.top]:
        print(f"  {score:>6.3f}  {kw}")

    # Find natural gap — keywords below 0.1 are likely noise
    core = [(k, s) for k, s in ranked if s >= 0.1]
    edge = [(k, s) for k, s in ranked if s < 0.1]
    print(f"\nCore keywords (score >= 0.1): {len(core)}")
    print(f"Edge keywords (score < 0.1) : {len(edge)}")
    if edge:
        print(f"Edge examples: {[k for k, _ in edge[:8]]}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump({"scores": dict(ranked)}, f, indent=2)
    print(f"\nSaved scores to {OUTPUT_FILE}")
    print("Now run: python fetch_papers.py --target 300")


if __name__ == "__main__":
    main()
