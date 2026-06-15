"""
Pipeline UI Server
---------------------------------------
FastAPI server that orchestrates all 4 agents via Server-Sent Events (SSE),
letting the browser stream real-time logs from each agent subprocess.

Usage:
    set ANTHROPIC_API_KEY=sk-ant-...
    set GITHUB_TOKEN=ghp_...
    python pipeline_ui/server.py
    # Open http://localhost:8080
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

app = FastAPI(title="Agentic Test Pipeline")

BASE_DIR   = Path(__file__).parent.parent
AGENT1_DIR = BASE_DIR / "agent1_understanding"
AGENT2_DIR = BASE_DIR / "agent2_requirements"
AGENT3_DIR = BASE_DIR / "agent3_test_writer"
AGENT4_DIR = BASE_DIR / "agent4_execution"
UI_DIR     = Path(__file__).parent

DEFAULT_APP_REPO  = "https://github.com/DILEEPKUMAREBIX/ltc_hackathon_cra"
DEFAULT_TEST_REPO = "https://github.com/DILEEPKUMAREBIX/ltc_hackathon_cra_automation_tests"


# ---------------------------------------------------------------------------
# SSE streaming helper
# ---------------------------------------------------------------------------
async def stream_subprocess(cmd: list, cwd: Path):
    env = os.environ.copy()
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(cwd),
        env=env,
    )
    async for raw in process.stdout:
        line = raw.decode("utf-8", errors="replace").rstrip()
        if line:
            yield f"data: {json.dumps({'log': line})}\n\n"
    rc = await process.wait()
    yield f"data: {json.dumps({'exit': rc})}\n\n"


def sse_response(generator):
    return StreamingResponse(
        generator,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# Static
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def index():
    return HTMLResponse((UI_DIR / "index.html").read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Agent endpoints
# ---------------------------------------------------------------------------
@app.get("/run/agent1")
async def run_agent1(
    repo: str = DEFAULT_APP_REPO,
    test_repo: str = DEFAULT_TEST_REPO,
):
    cmd = [
        sys.executable, str(AGENT1_DIR / "ingest.py"),
        "--repo", repo,
        "--test-repo", test_repo,
    ]
    return sse_response(stream_subprocess(cmd, BASE_DIR))


@app.get("/run/agent2")
async def run_agent2():
    cmd = [
        sys.executable, str(AGENT2_DIR / "analyze.py"),
        "--ticket", str(AGENT2_DIR / "sample_ticket.json"),
        "--out",    str(AGENT2_DIR / "scenarios.json"),
    ]
    return sse_response(stream_subprocess(cmd, BASE_DIR))


@app.get("/run/agent3")
async def run_agent3(
    dry_run: bool = False,
    no_ui_snapshot: bool = False,
    app_url: str = "http://localhost:5173",
):
    cmd = [
        sys.executable, str(AGENT3_DIR / "write_tests.py"),
        "--scenarios", str(AGENT2_DIR / "scenarios.json"),
        "--app-url", app_url,
    ]
    if dry_run:
        cmd.append("--dry-run")
    if no_ui_snapshot:
        cmd.append("--no-ui-snapshot")
    return sse_response(stream_subprocess(cmd, BASE_DIR))


@app.get("/run/agent4")
async def run_agent4():
    cmd = [sys.executable, str(AGENT4_DIR / "run_tests.py")]
    return sse_response(stream_subprocess(cmd, BASE_DIR))


# ---------------------------------------------------------------------------
# Clear output files
# ---------------------------------------------------------------------------
@app.delete("/api/clear")
async def clear_outputs():
    removed = []
    targets = [
        AGENT2_DIR / "scenarios.json",
        AGENT3_DIR / "output" / "test_generated.spec.js",
        AGENT3_DIR / "output" / "test_generated.py",
        AGENT4_DIR / "report.json",
    ]
    for f in targets:
        if f.exists():
            f.unlink()
            removed.append(f.name)
    return JSONResponse({"cleared": removed})


# ---------------------------------------------------------------------------
# Data endpoints
# ---------------------------------------------------------------------------
@app.get("/api/ticket")
async def get_ticket():
    f = AGENT2_DIR / "sample_ticket.json"
    if not f.exists():
        return JSONResponse({"error": "not found"}, status_code=404)
    return JSONResponse(json.loads(f.read_text(encoding="utf-8")))


@app.post("/api/ticket")
async def save_ticket(request: Request):
    body = await request.json()
    (AGENT2_DIR / "sample_ticket.json").write_text(
        json.dumps(body, indent=2), encoding="utf-8"
    )
    return JSONResponse({"status": "saved"})


@app.get("/api/scenarios")
async def get_scenarios():
    f = AGENT2_DIR / "scenarios.json"
    if not f.exists():
        return JSONResponse({"error": "Run Agent 2 first"}, status_code=404)
    return JSONResponse(json.loads(f.read_text(encoding="utf-8")))


@app.get("/api/report")
async def get_report():
    f = AGENT4_DIR / "report.json"
    if not f.exists():
        return JSONResponse({"error": "Run Agent 4 first"}, status_code=404)
    return JSONResponse(json.loads(f.read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("🚀  Pipeline UI → http://localhost:8080")
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="warning")
