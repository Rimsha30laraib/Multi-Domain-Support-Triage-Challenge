"""
responder.py — Builds prompts, calls the Groq API (LLaMA-3), and parses structured output.

Responsibilities:
  1. format_context()  — converts retrieved chunks into a readable context block
  2. build_messages()  — assembles system + user message list for chat completion
  3. call_groq()       — sends the request and returns raw text
  4. parse_response()  — extracts and validates the JSON output
  5. generate()        — top-level function that orchestrates 1-4
"""

import re
import json
import time
import sys

from groq import Groq

from config import (
    GROQ_API_KEY, GROQ_MODEL, MAX_TOKENS, TEMPERATURE,
    VALID_STATUSES, VALID_REQUEST_TYPES, LLM_RETRIES,
)
from safety import classify_request_type


# ── Prompt templates ──────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a senior support triage agent for a multi-product company.
You handle tickets for three ecosystems:
  • HackerRank — developer hiring platform
  • Claude      — AI assistant by Anthropic
  • Visa        — global payment network

Core rules:
  - Ground every answer ONLY in the provided documentation snippets.
  - Never invent policies, URLs, timelines, or fees not mentioned in the docs.
  - Escalate when: no relevant documentation found, billing disputes needing
    human review, account security issues, legal matters, or any situation
    where a wrong answer could cause significant harm.
  - Reply when: the documentation clearly and safely resolves the question.
  - Respond with a SINGLE valid JSON object — no markdown fences, no preamble.
"""

_USER_TEMPLATE = """\
### Support Ticket
{ticket}

### Retrieved Documentation
{context}

### Required Output (JSON only — no markdown, no extra text)
{{
  "status":        "replied" | "escalated",
  "product_area":  "<specific area, e.g. billing / account access / assessments / card payments / API>",
  "response":      "<2–5 sentence user-facing reply, grounded in the docs above>",
  "justification": "<1–2 sentence internal note explaining the routing decision>",
  "request_type":  "product_issue" | "feature_request" | "bug" | "invalid"
}}
"""


# ── Context formatter ─────────────────────────────────────────────────────────

def format_context(retrieved: list[dict]) -> str:
    """Convert retrieved chunks into a numbered context block."""
    if not retrieved:
        return "No relevant documentation found."

    parts = []
    for i, doc in enumerate(retrieved, 1):
        snippet = doc["text"][:500].strip()
        parts.append(
            f"[Doc {i} | ecosystem: {doc['ecosystem']} | title: {doc['title']}]\n{snippet}"
        )
    return "\n\n".join(parts)


# ── Message builder ───────────────────────────────────────────────────────────

def build_messages(ticket: str, context: str, subject: str = "", company: str = "") -> list[dict]:
    """Assemble the system + user message list for the Groq chat API."""
    return [
        {"role": "system", "content": _SYSTEM_PROMPT.strip()},
        {"role": "user",   "content": _USER_TEMPLATE.format(
            ticket=ticket.strip(),
            context=context,
            subject=subject.strip(),
            company=company.strip(),
        )},
    ]

# ── Groq API call ─────────────────────────────────────────────────────────────

def init_groq() -> Groq:
    """Initialise and return the Groq client. Call once at startup."""
    if not GROQ_API_KEY:
        print("[Error] GROQ_API_KEY is not set. Export it before running.")
        sys.exit(1)
    return Groq(api_key=GROQ_API_KEY)


def call_groq(client: Groq, messages: list[dict]) -> str:
    """
    Send messages to Groq with exponential-backoff retries.

    Returns raw response text or empty string on total failure.
    """
    for attempt in range(LLM_RETRIES):
        try:
            chat = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=messages,
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
            )
            return chat.choices[0].message.content.strip()
        except Exception as e:
            wait = 2 ** attempt
            print(f"  [Groq] Attempt {attempt + 1} failed: {e}. Retrying in {wait}s …")
            time.sleep(wait)

    print("  [Groq] All retries exhausted. Returning empty string.")
    return ""


# ── Response parser ───────────────────────────────────────────────────────────

def parse_response(raw: str, ticket_text: str) -> dict:
    """
    Parse and validate the JSON output from the LLM.

    Falls back to a safe escalation dict on parse failure or missing fields.
    """
    try:
        cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
        result  = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return _fallback_response(ticket_text, reason="JSON parse error")

    if result.get("status") not in VALID_STATUSES:
        result["status"] = "escalated"

    if result.get("request_type") not in VALID_REQUEST_TYPES:
        result["request_type"] = classify_request_type(ticket_text)

    for key in ("status", "product_area", "response", "justification", "request_type"):
        if not result.get(key):
            return _fallback_response(ticket_text, reason=f"missing field '{key}'")

    return result


def _fallback_response(ticket_text: str, reason: str) -> dict:
    return {
        "status":        "escalated",
        "product_area":  "unknown",
        "response": (
            "We were unable to process your request automatically. "
            "A support agent will review your ticket and follow up shortly."
        ),
        "justification": f"Automated fallback escalation ({reason}); human review required.",
        "request_type":  classify_request_type(ticket_text),
    }


# ── Top-level generate ────────────────────────────────────────────────────────

def generate(ticket: str, retrieved: list[dict], client: Groq, subject: str = "", company: str = "") -> dict:
    """
    Full responder pipeline:
      1. Format retrieved docs into context.
      2. Build system + user messages.
      3. Call Groq (LLaMA-3).
      4. Parse + validate the JSON response.

    Args:
        ticket:    Raw support ticket text.
        retrieved: List of chunk dicts from HybridIndex.search().
        client:    Initialised Groq client instance.
        subject:   Ticket subject line (Subject column).
        company:   Company/product the ticket relates to (Company column).

    Returns:
        Validated result dict with keys:
        status, product_area, response, justification, request_type.
    """
    context  = format_context(retrieved)
    messages = build_messages(ticket, context, subject=subject, company=company)
    raw      = call_groq(client, messages)

    if not raw:
        return _fallback_response(ticket, reason="empty LLM response")

    return parse_response(raw, ticket)