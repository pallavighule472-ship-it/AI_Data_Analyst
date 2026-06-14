from fastapi import FastAPI, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import asyncio
import uuid, io, sqlite3, json, base64
from datetime import datetime, timedelta
from fpdf import FPDF

from Data_Analyst_Backend import (
    workflow, schema_node, query_classifier_node,
    analysis_node, chart_node, llm, build_response_prompt,
)

app = FastAPI(title="AI Data Analyst API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

MAX_FILE_SIZE = 50 * 1024 * 1024
MAX_SESSIONS  = 50
RATE_LIMIT    = 10
DB_PATH       = "sessions.db"

# ── SQLite session store ───────────────────────────────────

def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            csv_data   BLOB NOT NULL,
            created_at REAL DEFAULT (julianday('now'))
        )
    """)
    conn.commit()
    conn.close()

_init_db()


def _db_save(session_id: str, data: bytes):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO sessions (session_id, csv_data) VALUES (?, ?)",
            (session_id, data),
        )


def _db_load(session_id: str) -> pd.DataFrame | None:
    with sqlite3.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT csv_data FROM sessions WHERE session_id = ?", (session_id,)
        ).fetchone()
    return pd.read_csv(io.BytesIO(row[0])) if row else None


def _db_evict():
    with sqlite3.connect(DB_PATH) as conn:
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        if count >= MAX_SESSIONS:
            conn.execute("""
                DELETE FROM sessions WHERE session_id IN (
                    SELECT session_id FROM sessions ORDER BY created_at ASC LIMIT ?
                )
            """, (count - MAX_SESSIONS + 1,))


# ── In-memory cache (populated on demand, not at startup) ──

sessions: dict[str, pd.DataFrame] = {}

# ── Rate limiter ──────────────────────────────────────────

request_log: dict[str, list] = {}
_last_log_cleanup = datetime.min


def is_rate_limited(client_ip: str) -> bool:
    global _last_log_cleanup
    now    = datetime.now()
    cutoff = now - timedelta(minutes=1)

    if (now - _last_log_cleanup).total_seconds() > 300:
        stale = [ip for ip, ts in request_log.items() if not ts or max(ts) < cutoff]
        for ip in stale:
            del request_log[ip]
        _last_log_cleanup = now

    recent = [t for t in request_log.get(client_ip, []) if t > cutoff]
    if len(recent) >= RATE_LIMIT:
        request_log[client_ip] = recent
        return True
    recent.append(now)
    request_log[client_ip] = recent
    return False


# ── Routes ────────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "message": "AI Data Analyst API is running."}


@app.post("/upload")
async def upload_csv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload a CSV file. Returns a session_id to use in /analyze."""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are supported.")

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 50MB.")

    try:
        df = await asyncio.to_thread(pd.read_csv, io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {e}")

    session_id = str(uuid.uuid4())
    sessions[session_id] = df
    background_tasks.add_task(_db_evict)
    background_tasks.add_task(_db_save, session_id, contents)

    return {
        "session_id": session_id,
        "columns":    list(df.columns),
        "shape":      {"rows": df.shape[0], "cols": df.shape[1]},
        "preview":    df.head(5).to_dict(orient="records"),
    }


class QueryRequest(BaseModel):
    session_id: str
    query:      str
    history:    list[dict] = []


@app.post("/analyze")
def analyze(request: Request, body: QueryRequest):
    """Run the AI analyst on a previously uploaded CSV (non-streaming)."""
    if is_rate_limited(request.client.host):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 10 requests per minute.")

    df = sessions.get(body.session_id)
    if df is None:
        df = _db_load(body.session_id)
    if df is None:
        raise HTTPException(
            status_code=404,
            detail="Session not found. Please upload your CSV via /upload first.",
        )

    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        result = workflow.invoke({
            "user_query":   body.query,
            "dataframe":    df,
            "chat_history": body.history,
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Workflow error: {e}")

    analysis = result.get("analysis_result") or {}
    return {
        "response": result.get("final_response", ""),
        "chart":    result.get("chart"),
        "code":     analysis.get("code"),
        "error":    analysis.get("error"),
    }


# ── Streaming endpoint ─────────────────────────────────────

@app.post("/analyze_stream")
def analyze_stream(request: Request, body: QueryRequest):
    """Run analysis and stream the LLM response token by token.

    Protocol: first line is JSON metadata (chart, code, error),
    followed by the LLM response text as raw chunks.
    """
    if is_rate_limited(request.client.host):
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Max 10 requests per minute.")

    df = sessions.get(body.session_id)
    if df is None:
        df = _db_load(body.session_id)
    if df is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    if not body.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # Run analysis pipeline (schema → classify → pandas → chart)
    state: dict = {
        "user_query":      body.query,
        "dataframe":       df,
        "chat_history":    body.history,
        "schema":          {},
        "analysis_result": {},
        "needs_chart":     False,
        "intent":          "open",
        "chart":           None,
        "final_response":  "",
    }
    try:
        state.update(schema_node(state))
        state.update(query_classifier_node(state))
        state.update(analysis_node(state))
        if state.get("needs_chart"):
            state.update(chart_node(state))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis pipeline error: {e}")

    analysis = state.get("analysis_result") or {}
    chart    = state.get("chart")
    prompt   = build_response_prompt(body.query, analysis, chart, body.history)

    def generate():
        # Line 1: JSON metadata so the client can render chart + store code/error
        meta = {
            "chart": chart,
            "code":  analysis.get("code"),
            "error": analysis.get("error"),
        }
        yield json.dumps(meta) + "\n"

        # Remaining chunks: streamed LLM response text
        try:
            for chunk in llm.stream(prompt):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            yield f"\n[Error generating response: {e}]"

    return StreamingResponse(generate(), media_type="text/plain")


# ── PDF export endpoint ────────────────────────────────────

class ReportRequest(BaseModel):
    query:    str
    response: str
    chart:    str | None = None


@app.post("/export_pdf")
def export_pdf(body: ReportRequest):
    """Generate a branded PDF report with question, analysis text, and optional chart."""
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(108, 99, 255)
    pdf.cell(0, 12, "AnalytIQ Analysis Report", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, f"Generated: {datetime.now().strftime('%Y-%m-%d  %H:%M')}", ln=True, align="C")
    pdf.ln(8)

    pdf.set_fill_color(240, 240, 255)

    # Question
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "  Question", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, body.query)
    pdf.ln(5)

    # Analysis
    pdf.set_font("Helvetica", "B", 13)
    pdf.cell(0, 9, "  Analysis", ln=True, fill=True)
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 7, body.response)

    # Chart
    if body.chart:
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 13)
        pdf.cell(0, 9, "  Chart", ln=True, fill=True)
        pdf.ln(3)
        try:
            img_bytes = base64.b64decode(body.chart)
            pdf.image(io.BytesIO(img_bytes), w=170)
        except Exception:
            pdf.set_font("Helvetica", "I", 10)
            pdf.cell(0, 7, "(Chart could not be embedded.)", ln=True)

    pdf_bytes = bytes(pdf.output())
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=analytiq_report.pdf"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("run:app", host="127.0.0.1", port=8000, reload=True)
