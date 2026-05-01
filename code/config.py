"""
config.py — Central configuration for the Groq Support Triage Agent.
Edit this file to tune models, paths, retrieval weights, and escalation rules.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (one level above groq_agent/)
load_dotenv(Path(__file__).parent.parent / ".env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


# # ── API ──────────────────────────────────────────────────────────────────────
GROQ_MODEL = "llama-3.1-8b-instant"   # recommended replacement
# ── Embedding model (local, no API needed) ────────────────────────────────────
EMBED_MODEL = "all-MiniLM-L6-v2"

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K         = 8
CHUNK_SIZE    = 600
CHUNK_OVERLAP = 80

# ── Generation ────────────────────────────────────────────────────────────────
MAX_TOKENS  = 1024
TEMPERATURE = 0.1     # low = more deterministic, safer for triage

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR   = Path(__file__).parent.parent
DATA_DIR   = ROOT_DIR / "data"
TICKETS_IN = ROOT_DIR / "support_tickets" / "support_tickets.csv"
OUTPUT_CSV = ROOT_DIR / "support_tickets" / "output.csv"

# ── Output schema ─────────────────────────────────────────────────────────────
OUTPUT_FIELDS       = ["ticket_id","issue","subject","company","response","product_area","status","request_type","justification"]
VALID_STATUSES      = {"replied", "escalated"}
VALID_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}

# ── Rate limiting ─────────────────────────────────────────────────────────────
SLEEP_BETWEEN_TICKETS = 0.8   # Groq free tier ~30 RPM
LLM_RETRIES           = 3