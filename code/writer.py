"""
writer.py — CSV input reading and output writing for the triage agent.

Keeps all I/O logic in one place so main.py stays clean.
"""

import csv
from pathlib import Path

from config import TICKETS_IN, OUTPUT_CSV, OUTPUT_FIELDS


# ── Input ─────────────────────────────────────────────────────────────────────

def read_tickets(filepath: Path = TICKETS_IN) -> list[dict]:
    """
    Read support_tickets.csv and return a list of row dicts.

    Auto-detects column names by checking all common variants and
    falling back gracefully if no ID column exists (auto-numbers rows).
    Prints detected columns so mismatches are obvious at a glance.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"Tickets file not found: {filepath}")

    # All known variants for each logical field (case-sensitive matches)
    ID_COLS = [
        "ticket_id", "id", "ID", "Ticket_ID", "ticket id",
    ]
    TEXT_COLS = [
        "ticket", "description", "text", "body", "message",
        "content", "issue", "query", "subject",
        "Ticket", "Description", "Text", "Body", "Message",
        "Issue", "Query", "Subject",                          # ← capitalised variants
    ]

    rows = []
    with open(filepath, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []

        # ── Print headers so mismatches are instantly visible ────────────────
        print(f"  [Writer] CSV columns found: {headers}")

        # ── Detect ID column ─────────────────────────────────────────────────
        # Do NOT fall back to using a text column as the ID.
        # If no ID column exists, we auto-number rows instead.
        id_col = next((c for c in ID_COLS if c in headers), None)
        if id_col:
            print(f"  [Writer] Using '{id_col}' as ticket_id column")
        else:
            print("  [Writer] No ticket_id column found — rows will be auto-numbered")

        # ── Detect text column ───────────────────────────────────────────────
        # Only consider columns that are NOT the id_col.
        text_col = next(
            (c for c in TEXT_COLS if c in headers and c != id_col),
            None,
        )
        if not text_col:
            # Last resort: pick any column that isn't the id column
            text_col = next((c for c in headers if c != id_col), None)

        print(f"  [Writer] Using '{text_col}' as ticket text column")

        if not text_col:
            raise ValueError(
                f"Cannot find a ticket-text column in: {headers}\n"
                f"Open writer.py and add your column name to TEXT_COLS."
            )

        # ── Detect Subject and Company columns (case-insensitive) ───────────
        subject_col = next((c for c in headers if c.lower() == "subject"), None)
        company_col = next((c for c in headers if c.lower() == "company"), None)
        print(f"  [Writer] Using '{subject_col}' as subject column")
        print(f"  [Writer] Using '{company_col}' as company column")

        # ── Read rows ────────────────────────────────────────────────────────
        for i, raw in enumerate(reader, start=1):
            # Use the real ID column if found; otherwise auto-number
            if id_col:
                ticket_id = (raw.get(id_col) or "").strip() or str(i)
            else:
                ticket_id = str(i)

            ticket_text = (raw.get(text_col) or "").strip()
            subject     = (raw.get(subject_col) or "").strip() if subject_col else ""
            company     = (raw.get(company_col) or "").strip() if company_col else ""

            if not ticket_text:
                continue

            rows.append({
                "ticket_id":   ticket_id,
                "ticket_text": ticket_text,
                "subject":     subject,
                "company":     company,
            })

    return rows


# ── Output ────────────────────────────────────────────────────────────────────

class OutputWriter:
    """
    Context manager that opens the output CSV and provides a .write() method.

    Usage:
        with OutputWriter() as w:
            w.write(ticket_id, result)
    """

    def __init__(self, filepath: Path = OUTPUT_CSV):
        self.filepath = filepath
        self._file   = None
        self._writer = None

    def __enter__(self):
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file   = open(self.filepath, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=OUTPUT_FIELDS)
        self._writer.writeheader()
        return self

    def __exit__(self, *_):
        if self._file:
            self._file.close()

    def write(self, ticket_id: str, result: dict) -> None:
        """Write one result row to the CSV."""
        self._writer.writerow({
            "ticket_id":     ticket_id,
            "issue":         result.get("issue", ""),
            "subject":       result.get("subject", ""),
            "company":       result.get("company", ""),
            "status":        result.get("status", ""),
            "product_area":  result.get("product_area", ""),
            "response":      result.get("response", ""),
            "justification": result.get("justification", ""),
            "request_type":  result.get("request_type", ""),
        })
        # Flush after each row so partial results are saved if the run fails mid-way
        self._file.flush()