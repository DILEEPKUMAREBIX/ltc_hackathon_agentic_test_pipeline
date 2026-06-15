"""
Agent 2: Requirement Analysis Agent
---------------------------------------
Reads a (mock) JIRA ticket describing a new feature/requirement, pulls
relevant codebase context from Agent 1's vector store, and asks Claude
to produce a structured list of test scenarios (JSON).

This JSON is the CONTRACT consumed by Agent 3 (test writer).

Usage:
    python analyze.py --ticket sample_ticket.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic

# Allow importing Agent 1's query() function
AGENT1_DIR = Path(__file__).parent.parent / "agent1_understanding"
sys.path.insert(0, str(AGENT1_DIR))
from ingest import query as rag_query  # noqa: E402

MODEL = "claude-sonnet-4-6"

SCENARIO_SCHEMA_DESC = """
Return ONLY a JSON array (no markdown, no preamble). Each element must have:
{
  "id": "TS-001",
  "title": "short title",
  "type": "functional" | "edge_case" | "negative" | "regression",
  "layer": "ui" | "api" | "both",
  "preconditions": "string",
  "steps": ["step 1", "step 2", ...],
  "expected_result": "string",
  "related_files": ["file paths from context, if relevant"]
}
"""


def load_ticket(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def gather_context(ticket: dict, n_results: int = 5) -> str:
    """Query Agent 1's vector store using the ticket title + description
    to find the most relevant existing code/doc chunks."""
    search_text = f"{ticket.get('title', '')} {ticket.get('description', '')}"
    hits = rag_query(search_text, n_results=n_results)

    context_blocks = []
    for h in hits:
        context_blocks.append(f"--- {h['source']} ---\n{h['text']}")
    return "\n\n".join(context_blocks)


def build_prompt(ticket: dict, context: str) -> str:
    return f"""You are a senior QA engineer analyzing a new feature request
for an existing application.

EXISTING CODEBASE CONTEXT (retrieved via RAG, may be partial):
{context}

NEW REQUIREMENT (JIRA ticket):
Title: {ticket.get('title')}
Description: {ticket.get('description')}
Acceptance Criteria:
{json.dumps(ticket.get('acceptance_criteria', []), indent=2)}

TASK:
Analyze how this new requirement interacts with the existing system shown
in the context above. Identify:
1. Functional test scenarios covering the acceptance criteria
2. Edge cases (boundary values, empty/null inputs, large inputs)
3. Negative tests (invalid input, unauthorized access, error handling)
4. Regression scenarios (existing flows that could break due to this change)

For each scenario, classify whether it should be tested at the UI layer,
API layer, or both, based on the codebase context.

{SCENARIO_SCHEMA_DESC}
"""


def analyze(ticket: dict, n_context: int = 5) -> list:
    context = gather_context(ticket, n_results=n_context)
    prompt = build_prompt(ticket, context)

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    # Strip markdown code fences if present
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[len("json"):]
        text = text.strip()

    try:
        scenarios = json.loads(text)
    except json.JSONDecodeError:
        # Response was cut off mid-JSON — recover by closing at the last complete object
        last_brace = text.rfind("}")
        if last_brace != -1:
            repaired = text[: last_brace + 1] + "\n]"
            try:
                scenarios = json.loads(repaired)
                print(f"[analyze] Warning: response was truncated; recovered {len(scenarios)} scenarios")
            except json.JSONDecodeError as e:
                print("[analyze] Failed to parse JSON from model output:", e)
                print("--- raw output ---")
                print(text)
                raise
        else:
            raise

    return scenarios


def main():
    parser = argparse.ArgumentParser(description="Agent 2: Requirement Analysis")
    parser.add_argument("--ticket", required=True, help="Path to JIRA ticket JSON")
    parser.add_argument("--out", default="scenarios.json", help="Output file")
    parser.add_argument("--top-k", type=int, default=5, help="RAG context chunks")
    args = parser.parse_args()

    ticket = load_ticket(args.ticket)
    print(f"[analyze] Ticket: {ticket.get('title')}")

    scenarios = analyze(ticket, n_context=args.top_k)
    print(f"[analyze] Generated {len(scenarios)} test scenarios")

    out_path = Path(args.out)
    out_path.write_text(json.dumps(scenarios, indent=2))
    print(f"[analyze] Saved to {out_path}")

    for s in scenarios:
        print(f"  - [{s['type']}/{s['layer']}] {s['id']}: {s['title']}")


if __name__ == "__main__":
    main()
