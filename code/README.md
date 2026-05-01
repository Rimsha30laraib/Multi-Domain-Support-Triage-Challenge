# Support Triage Agent — Hybrid RAG

Fully separate implementations using Groq (LLaMA-3).
Both share the same Hybrid RAG architecture (TF-IDF + Sentence Embeddings).

---

## Architecture

```
support_tickets.csv
       │
       ▼
┌─────────────────────────────────────────────┐
│             Hybrid RAG Retriever            │
│  ┌──────────────┐    ┌────────────────────┐ │
│  │  TF-IDF      │    │ Sentence-Transformer│ │
│  │  (bigrams,   │    │ all-MiniLM-L6-v2   │ │
│  │  sublinear)  │    │ (cosine similarity) │ │
│  └──────┬───────┘    └─────────┬──────────┘ │
│         │   α=0.45             │  (1-α)=0.55│
│         └──────────┬───────────┘            │
│                    │ Fused score             │
│                    ▼                         │
│              Top-K chunks                   │
└─────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────┐
│  Heuristic escalation gate   │
│  (regex: fraud, hack, …)     │
└──────────────┬───────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
   Escalate      LLM Generation
   immediately   (Gemini / Groq)
                      │
                      ▼
                 Structured JSON
                      │
                      ▼
                 output.csv
```

---

## Setup

### 1. Install dependencies

```bash
cd code
pip install -r requirements.txt
```

### 2. Set API key in .env

```bash
GROQ_API_KEY="your-key-here"
```
Get a key at: https://console.groq.com/keys

### 3. Place corpus in `data/`

```
data/
├── hackerrank/    ← .txt / .md / .html files from HackerRank help center
├── claude/        ← .txt / .md / .html files from Claude Help Center
└── visa/          ← .txt / .md / .html files from Visa support
```

### 4. Build vector store (ChromaDB)

This:

chunks documents
generates embeddings
stores them in ChromaDB

### 5. Run
```
python code/main.py
```

Results → `support_tickets/output.csv`

---

## Output schema

| Column | Allowed values |
| `ticket_id` | from input CSV |
| `status` | `replied` \| `escalated` |
| `product_area` | e.g. `billing`, `account access`, `fraud`, `API` |
| `response` | user-facing answer grounded in corpus |
| `justification` | internal routing explanation |
| `request_type` | `product_issue` \| `feature_request` \| `bug` \| `invalid` |

---

## Key design decisions

| Decision | Rationale |
|---|---|
| **Hybrid TF-IDF + Semantic** | TF-IDF catches exact keyword matches (e.g. error codes), semantic handles paraphrase/intent. Best of both worlds. |
| **α = 0.45** | Slight semantic bias; tune per corpus. |
| **Heuristic escalation gate** | Fast regex pass catches high-risk tickets before any LLM call — safer and cheaper. |
| **Structured JSON output prompt** | Constrains the LLM to produce parseable output with validated fields. Fallback escalates on parse failure. |
| **No live web calls** | All retrieval is from local `data/` directory only — per challenge requirements. |

---

## Tuning

| Parameter | Location | Effect |
|---|---|---|
| `ALPHA` | top of `main.py` | 0 = pure semantic, 1 = pure TF-IDF |
| `TOP_K` | top of `main.py` | More docs = richer context, slower |
| `CHUNK_SIZE` | top of `main.py` | Larger = more context per chunk, fewer chunks |
| `ESCALATION_PATTERNS` | top of `main.py` | Add/remove risk keywords |