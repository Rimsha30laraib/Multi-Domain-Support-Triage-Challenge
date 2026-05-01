"""
agent.py — The triage orchestrator for the Groq Support Agent.

Wires together:
  safety    → pre-LLM escalation gate
  retriever → hybrid TF-IDF + semantic search
  responder → prompt construction, Groq call, JSON parsing

This module contains no business logic of its own — it simply coordinates
the other modules in the correct order.
"""

from groq import Groq

from config import TOP_K
from safety import should_escalate, escalation_response, classify_request_type
from retriever import HybridIndex
from responder import generate


class TriageAgent:
    """
    Main agent class.  Instantiate once, then call .process() per ticket.

    Args:
        index:  A pre-built HybridIndex (from retriever.load_or_build_index).
        client: An initialised Groq client (from responder.init_groq).
    """

    def __init__(self, index: HybridIndex, client: Groq):
        self.index  = index
        self.client = client

    def process(self, ticket_text: str, subject: str = "", company: str = "") -> dict:
        """
        Triage a single support ticket.

        Pipeline:
          1. Safety gate  — escalate immediately on high-risk keywords.
          2. Retrieval    — find the TOP_K most relevant corpus chunks.
          3. Generation   — call Groq with ticket + context; parse JSON.
          4. Enrich       — heuristic request_type sanity check.
          5. Attach       — staple original issue/subject/company onto result.

        Args:
            ticket_text: The main issue text (Issue column).
            subject:     The ticket subject line (Subject column).
            company:     The company/product the ticket relates to (Company column).

        Returns:
            dict with keys:
              issue, subject, company,
              status, product_area, response, justification, request_type
        """
        
        # ── Step 1: Safety gate ───────────────────────────────────────────────
        if should_escalate(ticket_text):
            result = escalation_response()
            result["request_type"] = classify_request_type(ticket_text)

        else:
            # ── Step 2: Retrieve relevant docs ────────────────────────────────
            # retrieved = self.index.search(ticket_text, top_k=TOP_K, alpha=ALPHA)
            # AFTER:
            eco = company.lower() if company.lower() in ("hackerrank", "claude", "visa") else None
            retrieved = self.index.search(ticket_text, top_k=TOP_K, ecosystem_filter=eco)
            
            # ── Step 3: LLM generation + parsing ─────────────────────────────
            # Pass subject & company so the prompt has full context
            result = generate(ticket_text, retrieved, self.client,
                              subject=subject, company=company)

            # ── Step 4: Sanity check on request_type ─────────────────────────
            heuristic_type = classify_request_type(ticket_text)
            if heuristic_type in ("bug", "feature_request", "invalid"):
                result["request_type"] = heuristic_type

        # ── Step 5: Attach original CSV fields to result ──────────────────────
        result["issue"]   = ticket_text
        result["subject"] = subject
        result["company"] = company

        return result