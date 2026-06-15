# Agentic AI Testing Pipeline — Hackathon Prototype

## Pipeline
1. **agent1_understanding/ingest.py** — clones a GitHub repo, indexes code/docs/specs into ChromaDB
2. **agent2_requirements/analyze.py** — reads a JIRA ticket (mock JSON for now), queries Agent 1's index, asks Claude to generate structured test scenarios (JSON)
3. **agent3_test_writer/** (next) — converts scenarios into Playwright (UI) + REST Assured/Postman (API) test code
4. **agent4_execution/** (next) — runs generated tests, parses results, generates report

## Setup
```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...   # your own key
```

## Run Agent 1 (codebase ingestion)
```bash
cd agent1_understanding
python3 ingest.py --repo https://github.com/DILEEPKUMAREBIX/ltc_hackathon_cra --reset
python3 ingest.py --query "How does the login API work?"
```

## Run Agent 2 (requirement analysis -> test scenarios)
Edit agent2_requirements/sample_ticket.json with your real feature ticket, then:
```bash
cd agent2_requirements
python3 analyze.py --ticket sample_ticket.json --out scenarios.json
```
Output: scenarios.json — the contract consumed by Agent 3.

## Notes
- Agent 1 uses an offline hashing-based embedding (scikit-learn) to avoid
  model-download issues. For better semantic search, swap
  HashingEmbeddingFunction for OpenAI/Voyage/Anthropic embeddings once you
  have API access.
- Agent 2 requires ANTHROPIC_API_KEY to be set in your environment.
