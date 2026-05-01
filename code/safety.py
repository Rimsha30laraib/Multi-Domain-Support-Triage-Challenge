"""
safety.py — Pre-LLM safety gate and request-type classifier.

Two responsibilities:
  1. should_escalate()   — fast regex check for high-risk tickets that must
                           NEVER be answered by an LLM alone.
  2. classify_request_type() — keyword heuristic to label a ticket as
                               product_issue | feature_request | bug | invalid
                               (used as a fallback / sanity check).

Design principle: when in doubt, escalate. A false-positive escalation is
far less harmful than a hallucinated policy answer or a missed fraud signal.
"""

import re

# ── High-risk escalation patterns ─────────────────────────────────────────────
# Any ticket matching these should be routed to a human agent immediately,
# before calling the LLM.  Add or remove patterns as the corpus evolves.

_ESCALATION_REGEX = re.compile(
    r"\b("
    # Security / account compromise
    r"fraud|unauthorized(?:\s+charge)?|account\s+compromised|hack(?:ed)?|"
    r"identity\s+theft|phishing|breach|data\s+leak|"
    # Legal / regulatory
    r"legal|lawsuit|sue|court|gdpr|personal\s+data\s+request|"
    r"discrimination|accessibility\s+complaint|"
    # Financial disputes
    r"chargeback|dispute|refund\s+denied|double\s+charged|overcharged|"
    # Threats
    r"threaten|police|law\s+enforcement|"
    # Critical system failures
    r"production\s+down|critical\s+bug|data\s+loss|"
    # Assessment integrity (HackerRank-specific)
    r"assessment\s+result\s+wrong|false\s+positive|cheating\s+flag(?:ged)?|"
    r"wrongly\s+disqualified"
    r")\b",
    re.IGNORECASE,
)

# ── Request-type patterns (ordered — first match wins) ────────────────────────
_REQUEST_TYPE_RULES: list[tuple[str, re.Pattern]] = [
    ("invalid",         re.compile(r"^\s*(test|hello+|hi+|hey|dummy|asdf|foo|bar|ignore|n/?a)\s*$", re.I)),
    ("bug",             re.compile(r"\b(bug|broken|error|crash|not\s+working|glitch|fail(?:ing|ed)?|exception|500|502|503)\b", re.I)),
    ("feature_request", re.compile(r"\b(feature\s+request|suggest(?:ion)?|would\s+(?:like|love)|wish(?:\s+you\s+had)?|please\s+add|could\s+you\s+add|enhancement|roadmap)\b", re.I)),
]
_DEFAULT_REQUEST_TYPE = "product_issue"


# ── Public API ────────────────────────────────────────────────────────────────

def should_escalate(ticket_text: str) -> bool:
    """
    Return True if the ticket contains high-risk keywords that require
    immediate escalation to a human agent — bypassing LLM generation entirely.

    Args:
        ticket_text: Raw ticket string.

    Returns:
        True  → escalate without calling the LLM.
        False → safe to proceed with retrieval + LLM response.
    """
    return bool(_ESCALATION_REGEX.search(ticket_text))


def classify_request_type(ticket_text: str) -> str:
    """
    Heuristic request-type classifier.

    Used as:
      - A fallback when the LLM output cannot be parsed.
      - A sanity-check to override an implausible LLM classification.

    Returns one of: 'product_issue', 'feature_request', 'bug', 'invalid'
    """
    for rtype, pattern in _REQUEST_TYPE_RULES:
        if pattern.search(ticket_text):
            return rtype
    return _DEFAULT_REQUEST_TYPE


def escalation_response() -> dict:
    """
    Standard pre-built response dict for heuristic escalations.
    This is returned directly to the writer without any LLM call.
    """
    return {
        "status":        "escalated",
        "product_area":  "security / sensitive",
        "response": (
            "Thank you for reaching out. Your request involves a sensitive issue "
            "that requires direct attention from our specialised support team. "
            "A human agent will review your case and contact you shortly. "
            "Please do not share additional personal or financial details through "
            "this channel."
        ),
        "justification": (
            "Heuristic escalation: ticket matched a high-risk keyword pattern "
            "(fraud, security, legal, billing dispute, or assessment integrity). "
            "Routed to human agent without LLM involvement."
        ),
    }