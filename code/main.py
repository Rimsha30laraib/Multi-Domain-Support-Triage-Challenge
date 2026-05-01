"""
main.py — Entry point for the Groq Support Triage Agent.

Run:
    export GROQ_API_KEY="your-key"
    python main.py

Module responsibilities:
    config.py    — all constants and settings
    corpus.py    — document loading and chunking
    retriever.py — hybrid TF-IDF + semantic index
    safety.py    — escalation heuristics and request-type classifier
    responder.py — Groq API calls (LLaMA-3) and JSON parsing
    agent.py     — triage orchestrator
    writer.py    — CSV I/O
    main.py      — startup, loop, progress reporting  ← YOU ARE HERE
"""

import time
from config import TICKETS_IN, OUTPUT_CSV, SLEEP_BETWEEN_TICKETS
from retriever import load_or_build_index
from responder import init_groq
from agent import TriageAgent
from writer import read_tickets, OutputWriter


def main():
    print("=" * 62)
    print("  Support Triage Agent  ·  Groq (LLaMA-3) + Hybrid RAG")
    print("=" * 62)

    print("\n[1/4] Initialising Groq client …")
    groq_client = init_groq()

    print("\n[2/4] Loading / building corpus index …")
    index = load_or_build_index()
    print(f"       {len(index.docs):,} chunks indexed across all ecosystems.")

    print("\n[3/4] Reading tickets …")
    tickets = read_tickets()
    print(f"       {len(tickets)} ticket(s) to process from {TICKETS_IN.name}")

    print("\n[4/4] Running triage …\n")

    agent = TriageAgent(index=index, client=groq_client)
    with OutputWriter() as writer:
        for i, row in enumerate(tickets, 1):
            tid     = row["ticket_id"]
            text    = row["ticket_text"]
            subject = row.get("subject", "")
            company = row.get("company", "")

            print(f"  [{i:>3}/{len(tickets)}] #{tid}: {text[:72].rstrip()} …")

            result = agent.process(text, subject=subject, company=company)

            writer.write(tid, result)
            print(
                f"           → status={result['status']:<10}  "
                f"area={result['product_area']:<28}  "
                f"type={result['request_type']}"
            )

            if i < len(tickets):
                time.sleep(SLEEP_BETWEEN_TICKETS)

                
    print(f"\n✓  All done. Output written to: {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()