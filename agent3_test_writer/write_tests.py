"""
Agent 3: Test Writer Agent
---------------------------------------
Reads scenarios.json from Agent 2, queries ChromaDB for app code context
and existing test patterns, captures a live UI snapshot using Playwright,
then calls Claude to generate runnable test code and pushes to the test repo.

Generated files:
  playwright/test_generated.spec.js  — Playwright UI tests
  api/test_generated.py              — pytest API tests

Usage:
    python write_tests.py --scenarios ../agent2_requirements/scenarios.json
    python write_tests.py --scenarios ../agent2_requirements/scenarios.json --dry-run
    python write_tests.py --scenarios ../agent2_requirements/scenarios.json --no-ui-snapshot
"""

import argparse
import json
import os
import shutil
import stat
import sys
from datetime import datetime
from pathlib import Path

import anthropic
from git import Repo

AGENT1_DIR = Path(__file__).parent.parent / "agent1_understanding"
sys.path.insert(0, str(AGENT1_DIR))
from ingest import query as rag_query, CODEBASE_COLLECTION, TEST_PATTERNS_COLLECTION  # noqa: E402
from ui_inspector import inspect_pages  # noqa: E402

MODEL = "claude-sonnet-4-6"
TEST_REPO_URL = "https://github.com/DILEEPKUMAREBIX/ltc_hackathon_cra_automation_tests"
REPO_CACHE = Path(__file__).parent / "repo_cache"
OUTPUT_DIR = Path(__file__).parent / "output"
CREDENTIALS_FILE = Path(__file__).parent.parent / "config" / "test_credentials.json"

FRONTEND_URL = "http://localhost:5173"
BACKEND_URL = "http://localhost:8000"


def load_credentials() -> dict:
    if CREDENTIALS_FILE.exists():
        creds = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8"))
        print(f"[write_tests] Loaded credentials from {CREDENTIALS_FILE.name}")
        return creds
    print("[write_tests] Warning: config/test_credentials.json not found — using placeholders")
    return {
        "valid": {"username": "<USERNAME>", "password": "<PASSWORD>"},
        "invalid": [{"username": "wronguser", "password": "wrong", "note": "expect 401"}],
    }


def _force_remove(func, path, _exc_info):
    os.chmod(path, stat.S_IWRITE)
    func(path)


# ---------------------------------------------------------------------------
# Context gathering
# ---------------------------------------------------------------------------
def gather_context(scenarios: list, collection_name: str, n_results: int = 5) -> str:
    """Build a combined RAG context string from all scenario titles."""
    search_text = " ".join(s.get("title", "") for s in scenarios)
    try:
        hits = rag_query(search_text, n_results=n_results, collection_name=collection_name)
    except Exception as e:
        print(f"[write_tests] Warning: could not query '{collection_name}': {e}")
        return "(no context available)"

    blocks = [f"--- {h['source']} ---\n{h['text']}" for h in hits]
    return "\n\n".join(blocks)


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------
def build_playwright_prompt(scenarios: list, app_context: str, test_patterns: str,
                            ui_snapshot: str = "", credentials: dict = None) -> str:
    ui_scenarios = [s for s in scenarios if s.get("layer") in ("ui", "both")]
    creds = credentials or {}
    creds_block = json.dumps(creds, indent=2) if creds else "(not provided)"
    return f"""You are a senior QA engineer. Generate a complete, runnable Playwright test file
(JavaScript, CommonJS) for the scenarios listed below.

APP CODEBASE CONTEXT (from RAG):
{app_context}

EXISTING TEST PATTERNS (follow these conventions exactly):
{test_patterns}

LIVE UI SNAPSHOT (captured from the running app — use these REAL selectors, labels, and field names):
{ui_snapshot if ui_snapshot else "(not available — app may not be running; infer selectors from codebase context)"}

TEST CREDENTIALS (use these exact values — do not invent credentials):
{creds_block}

FRONTEND URL: {FRONTEND_URL}
BACKEND URL:  {BACKEND_URL}

SCENARIOS TO IMPLEMENT ({len(ui_scenarios)} total):
{json.dumps(ui_scenarios, indent=2)}

REQUIREMENTS:
- Use @playwright/test (import {{ test, expect }} from '@playwright/test')
- One test() block per scenario, named with the scenario id and title
- Use REAL field ids/names/placeholders from the UI snapshot above — do not guess selectors
- Use the valid credentials for positive login tests and each invalid entry for negative tests
- Use page.goto(), page.fill(), page.click(), page.waitForURL(), expect(page) etc.
- Add a beforeEach that navigates to the login page if needed
- Handle both positive and negative assertions
- Return ONLY the complete .spec.js file content, no markdown fences
"""


def build_pytest_prompt(scenarios: list, app_context: str, test_patterns: str,
                        ui_snapshot: str = "", credentials: dict = None) -> str:
    api_scenarios = [s for s in scenarios if s.get("layer") in ("api", "both")]
    creds = credentials or {}
    creds_block = json.dumps(creds, indent=2) if creds else "(not provided)"
    return f"""You are a senior QA engineer. Generate a complete, runnable pytest test file
(Python) for the API scenarios listed below.

APP CODEBASE CONTEXT (from RAG):
{app_context}

EXISTING TEST PATTERNS (follow these conventions exactly):
{test_patterns}

LIVE UI SNAPSHOT (for reference — shows real field names and page structure):
{ui_snapshot if ui_snapshot else "(not available)"}

TEST CREDENTIALS (use these exact values — do not invent credentials):
{creds_block}

BACKEND BASE URL: {BACKEND_URL}

SCENARIOS TO IMPLEMENT ({len(api_scenarios)} total):
{json.dumps(api_scenarios, indent=2)}

REQUIREMENTS:
- Use the requests library (import requests)
- Use pytest fixtures for the base URL and any shared setup
- One def test_<id>() function per scenario
- Use the valid credentials for positive auth tests and each invalid entry for negative tests
- Use real API field names from the codebase context above
- Assert HTTP status codes, response body fields, and headers as appropriate
- Include a conftest.py section at the top as a comment if fixtures are needed
- Return ONLY the complete test_generated.py file content, no markdown fences
"""


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def call_claude(prompt: str) -> str:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(
        block.text for block in response.content if block.type == "text"
    ).strip()

    # Strip markdown code fences if model wraps output anyway
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        for prefix in ("javascript", "js", "python", "py"):
            if text.startswith(prefix):
                text = text[len(prefix):]
        # strip trailing fence
        if "```" in text:
            text = text[: text.rfind("```")]
        text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Git: clone, commit, push
# ---------------------------------------------------------------------------
def push_to_repo(playwright_code: str, pytest_code: str, dry_run: bool = False,
                  branch: str = ""):
    token = os.environ.get("GITHUB_TOKEN")
    if not token and not dry_run:
        raise EnvironmentError(
            "GITHUB_TOKEN is not set. Export it before running:\n"
            "  set GITHUB_TOKEN=ghp_YOUR_TOKEN_HERE"
        )

    # Use a timestamped branch so each pipeline run gets its own review branch
    if not branch:
        branch = "ai-tests/" + datetime.now().strftime("%Y%m%d-%H%M%S")

    repo_name = TEST_REPO_URL.rstrip("/").split("/")[-1].replace(".git", "")
    dest = REPO_CACHE / repo_name

    if dest.exists():
        shutil.rmtree(dest, onexc=_force_remove)

    REPO_CACHE.mkdir(parents=True, exist_ok=True)

    # Build authenticated clone URL
    if token:
        auth_url = TEST_REPO_URL.replace("https://", f"https://{token}@")
    else:
        auth_url = TEST_REPO_URL

    print(f"[write_tests] Cloning test repo -> {dest}")
    repo = Repo.clone_from(auth_url, dest, depth=1)

    # Create and checkout new review branch
    new_branch = repo.create_head(branch)
    new_branch.checkout()
    print(f"[write_tests] Created branch: {branch}")

    # Write generated files
    pw_path = dest / "playwright" / "test_generated.spec.js"
    py_path = dest / "api" / "test_generated.py"

    pw_path.parent.mkdir(parents=True, exist_ok=True)
    py_path.parent.mkdir(parents=True, exist_ok=True)

    pw_path.write_text(playwright_code, encoding="utf-8")
    py_path.write_text(pytest_code, encoding="utf-8")

    print(f"[write_tests] Written: {pw_path.relative_to(dest)}")
    print(f"[write_tests] Written: {py_path.relative_to(dest)}")

    if dry_run:
        print("[write_tests] --dry-run: skipping git commit and push")
        return

    repo.index.add([str(pw_path), str(py_path)])
    repo.index.commit("chore: add AI-generated test cases via agentic pipeline [Agent 3]")
    origin = repo.remote(name="origin")
    origin.push(refspec=f"{branch}:{branch}", set_upstream=True)
    print(f"[write_tests] Pushed to origin/{branch}")

    # Print a direct PR creation link for the engineer
    repo_path = TEST_REPO_URL.rstrip("/").replace("https://github.com/", "")
    pr_url = f"https://github.com/{repo_path}/compare/{branch}?expand=1"
    print(f"[write_tests] Open PR: {pr_url}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Agent 3: Test Writer")
    parser.add_argument(
        "--scenarios",
        default="../agent2_requirements/scenarios.json",
        help="Path to scenarios.json from Agent 2",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate tests locally but skip git push",
    )
    parser.add_argument(
        "--app-url",
        default=FRONTEND_URL,
        help="Base URL of the running frontend app for UI inspection",
    )
    parser.add_argument(
        "--no-ui-snapshot",
        action="store_true",
        help="Skip live UI inspection (use if the app is not running)",
    )
    parser.add_argument("--top-k", type=int, default=6, help="RAG context chunks")
    parser.add_argument(
        "--branch",
        default="",
        help="Branch name to push generated tests to (default: ai-tests/<timestamp>)",
    )
    args = parser.parse_args()

    scenarios_path = Path(args.scenarios)
    if not scenarios_path.exists():
        print(f"[write_tests] ERROR: scenarios file not found: {scenarios_path}")
        sys.exit(1)

    scenarios = json.loads(scenarios_path.read_text(encoding="utf-8"))
    print(f"[write_tests] Loaded {len(scenarios)} scenarios from {scenarios_path}")

    credentials = load_credentials()

    ui_scenarios = [s for s in scenarios if s.get("layer") in ("ui", "both")]
    api_scenarios = [s for s in scenarios if s.get("layer") in ("api", "both")]
    print(f"[write_tests] UI: {len(ui_scenarios)}  API: {len(api_scenarios)}")

    # Gather context from both ChromaDB collections
    print("[write_tests] Querying codebase context...")
    app_context = gather_context(scenarios, CODEBASE_COLLECTION, n_results=args.top_k)

    print("[write_tests] Querying test pattern context...")
    test_patterns = gather_context(scenarios, TEST_PATTERNS_COLLECTION, n_results=args.top_k)

    # Capture live UI snapshot so Claude knows real selectors / field names
    ui_snapshot = ""
    if not args.no_ui_snapshot and ui_scenarios:
        print(f"[write_tests] Inspecting live UI at {args.app_url} ...")
        pages_to_inspect = [
            args.app_url,
            args.app_url.rstrip("/") + "/login",
        ]
        ui_snapshot = inspect_pages(pages_to_inspect)
        snapshot_lines = ui_snapshot.count("\n")
        print(f"[write_tests] UI snapshot captured ({snapshot_lines} lines)")
    elif args.no_ui_snapshot:
        print("[write_tests] UI snapshot skipped (--no-ui-snapshot)")
    else:
        print("[write_tests] UI snapshot skipped (no UI scenarios)")

    # Generate Playwright tests
    playwright_code = ""
    if ui_scenarios:
        print(f"[write_tests] Generating Playwright tests for {len(ui_scenarios)} UI scenarios...")
        pw_prompt = build_playwright_prompt(scenarios, app_context, test_patterns, ui_snapshot, credentials)
        playwright_code = call_claude(pw_prompt)
        print(f"[write_tests] Playwright code: {len(playwright_code)} chars")
    else:
        print("[write_tests] No UI scenarios — skipping Playwright generation")
        playwright_code = "// No UI scenarios generated"

    # Generate pytest tests
    pytest_code = ""
    if api_scenarios:
        print(f"[write_tests] Generating pytest tests for {len(api_scenarios)} API scenarios...")
        py_prompt = build_pytest_prompt(scenarios, app_context, test_patterns, ui_snapshot, credentials)
        pytest_code = call_claude(py_prompt)
        print(f"[write_tests] pytest code: {len(pytest_code)} chars")
    else:
        print("[write_tests] No API scenarios — skipping pytest generation")
        pytest_code = "# No API scenarios generated"

    # Save local copies
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "test_generated.spec.js").write_text(playwright_code, encoding="utf-8")
    (OUTPUT_DIR / "test_generated.py").write_text(pytest_code, encoding="utf-8")
    print(f"[write_tests] Local copies saved to {OUTPUT_DIR}")

    # Push to test repo
    push_to_repo(playwright_code, pytest_code, dry_run=args.dry_run, branch=args.branch)


if __name__ == "__main__":
    main()
