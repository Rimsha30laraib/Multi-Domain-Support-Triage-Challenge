"""
retriever.py — ChromaDB-backed vector search index.
Replaces the hybrid TF-IDF + FAISS approach with persistent ChromaDB + sentence-transformers.
"""

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from pathlib import Path

from config import EMBED_MODEL, TOP_K, DATA_DIR
from corpus import load_corpus

CHROMA_DIR = Path(__file__).parent / ".chroma_store"


class HybridIndex:
    """
    ChromaDB-backed semantic search index.
    Named HybridIndex to stay compatible with agent.py (no changes needed there).
    """
    def __init__(self):
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name="support_corpus",
            metadata={"hnsw:space": "cosine"},
        )
        self.docs = []  # ← always initialized

    def build(self, chunks: list[dict]) -> None:
        self.docs = chunks  # ← set BEFORE the early return too

        if self.collection.count() >= len(chunks):
            print(f"  [Chroma] Already indexed {self.collection.count()} docs — skipping rebuild.")
            return
    # ... rest of build unchanged
    # def __init__(self):
    #     self.embedder = SentenceTransformer(EMBED_MODEL)

    #     # Persistent ChromaDB client — survives restarts
    #     self.client = chromadb.PersistentClient(
    #         path=str(CHROMA_DIR),
    #         settings=Settings(anonymized_telemetry=False)
    #     )

    #     self.collection = self.client.get_or_create_collection(
    #         name="support_corpus",
    #         metadata={"hnsw:space": "cosine"}   # cosine similarity
    #     )

    # def build(self, chunks: list[dict]) -> None:
    #     """Embed and upsert all chunks into ChromaDB."""

    #     # Skip if already populated
    #     if self.collection.count() >= len(chunks):
    #         print(f"  [Chroma] Collection already has {self.collection.count()} docs — skipping build.")
    #         return

        print(f"  [Chroma] Embedding {len(chunks)} chunks with {EMBED_MODEL} …")

        ids        = [c["id"]        for c in chunks]
        texts      = [c["text"]      for c in chunks]
        metadatas  = [{"title": c["title"], "ecosystem": c["ecosystem"], "source": c["source"]} for c in chunks]

        # Embed in batches of 256 to avoid memory spikes
        BATCH = 256
        for i in range(0, len(chunks), BATCH):
            batch_ids   = ids[i:i+BATCH]
            batch_texts = texts[i:i+BATCH]
            batch_meta  = metadatas[i:i+BATCH]
            batch_embs  = self.embedder.encode(batch_texts, show_progress_bar=False).tolist()

            self.collection.upsert(
                ids        = batch_ids,
                documents  = batch_texts,
                embeddings = batch_embs,
                metadatas  = batch_meta,
            )

        print(f"  [Chroma] Build complete. {self.collection.count()} docs stored.")

    def search(self, query: str, top_k: int = TOP_K, alpha: float = 0.5,
               ecosystem_filter: str = None) -> list[dict]:
        """
        Semantic search via ChromaDB.

        Args:
            query:            Ticket text to search against.
            top_k:            Number of results to return.
            alpha:            Unused (kept for API compatibility with agent.py).
            ecosystem_filter: Optional — e.g. "visa" to restrict results.

        Returns:
            List of chunk dicts with keys: id, text, title, ecosystem, source.
        """
        query_emb = self.embedder.encode([query]).tolist()

        where = {"ecosystem": ecosystem_filter} if ecosystem_filter else None

        results = self.collection.query(
            query_embeddings = query_emb,
            n_results         = top_k,
            where             = where,          # ← ChromaDB metadata filter
            include           = ["documents", "metadatas", "distances"],
        )

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "id":        "",                  # not returned by query
                "text":      doc,
                "title":     meta.get("title", ""),
                "ecosystem": meta.get("ecosystem", "general"),
                "source":    meta.get("source", ""),
                "score":     round(1 - dist, 4),  # cosine similarity
            })

        return chunks


def load_or_build_index() -> HybridIndex:
    """
    Load existing ChromaDB store or build from corpus.
    Call once at startup from main.py.
    """
    index  = HybridIndex()
    chunks = load_corpus(DATA_DIR)

    print(f"  [Corpus] {len(chunks)} chunks loaded from data/")
    index.build(chunks)

    return index