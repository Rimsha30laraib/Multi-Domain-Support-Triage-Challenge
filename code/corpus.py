"""
corpus.py — Loads and chunks the local support corpus from data/.

Supports .txt, .md, .html, .htm files.
Each file is split into overlapping word-chunks for retrieval.
"""

import re
import hashlib
from pathlib import Path

from config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


# File extensions to index
SUPPORTED_EXTENSIONS = {".txt", ".md", ".html", ".htm"}

# Map directory names → ecosystem labels
ECOSYSTEM_MAP = {
    "hackerrank": "hackerrank",
    "claude":     "claude",
    "visa":       "visa",
}


def detect_ecosystem(fpath: Path) -> str:
    """Infer which support ecosystem a file belongs to from its path."""
    lower = str(fpath).lower()
    for key, label in ECOSYSTEM_MAP.items():
        if key in lower:
            return label
    return "general"


def clean_text(raw: str) -> str:
    """Strip HTML tags and normalise whitespace."""
    text = re.sub(r"<[^>]+>", " ", raw)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, title: str, ecosystem: str, fpath: Path) -> list[dict]:
    """
    Split text into overlapping word-level chunks.

    Each chunk dict contains:
        id, text, title, ecosystem, source
    """
    words  = text.split()
    chunks = []
    start  = 0

    while start < len(words):
        end   = min(start + CHUNK_SIZE, len(words))
        chunk = " ".join(words[start:end])
        cid   = hashlib.md5(chunk.encode()).hexdigest()[:8]

        chunks.append({
            "id":        cid,
            "text":      chunk,
            "title":     title,
            "ecosystem": ecosystem,
            "source":    str(fpath),
        })
        start += CHUNK_SIZE - CHUNK_OVERLAP

    return chunks


def load_corpus(data_dir: Path = DATA_DIR) -> list[dict]:
    """
    Walk data_dir recursively and return a flat list of chunk dicts.

    Skips files with unsupported extensions and files with < 80 characters
    of content (likely empty stubs).
    """
    docs = []

    for fpath in sorted(data_dir.rglob("*")):
        if fpath.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        try:
            raw = fpath.read_text(errors="replace")
        except Exception as e:
            print(f"  [Corpus] Could not read {fpath}: {e}")
            continue

        text = clean_text(raw)
        if len(text) < 80:
            continue

        ecosystem = detect_ecosystem(fpath)
        chunks    = chunk_text(text, fpath.stem, ecosystem, fpath)
        docs.extend(chunks)

    return docs