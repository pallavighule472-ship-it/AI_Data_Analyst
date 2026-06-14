import os
import streamlit as st
import requests
import base64
from PIL import Image
import io
import html
import time
import json as _json

def post_with_retry(url: str, max_retries: int = 3, **kwargs) -> requests.Response:
    for attempt in range(max_retries):
        try:
            return requests.post(url, **kwargs)
        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)

API_URL  = os.getenv("API_URL",  "http://127.0.0.1:8000")
PASSWORD = os.getenv("ANALYTIQ_PASSWORD", "analytiq123")

st.set_page_config(page_title="AnalytIQ", page_icon="⚡", layout="wide")

if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False
if "login_attempts" not in st.session_state:
    st.session_state["login_attempts"] = 0

if not st.session_state["authenticated"]:
    st.title("⚡ AnalytIQ")
    if st.session_state["login_attempts"] >= 5:
        st.error("Too many failed attempts. Please restart the app.")
        st.stop()
    pwd = st.text_input("Enter password to continue", type="password")
    if st.button("Login"):
        if pwd == PASSWORD:
            st.session_state["authenticated"] = True
            st.session_state["login_attempts"] = 0
            st.rerun()
        else:
            st.session_state["login_attempts"] += 1
            remaining = 5 - st.session_state["login_attempts"]
            st.error(f"Incorrect password. {remaining} attempt(s) remaining.")
    st.stop()

st.markdown("""
<style>
[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #eeeeee; }
@media (max-width: 768px) {
    .hero-title { font-size: 1.4rem !important; }
    .hero-subtitle { font-size: 0.85rem !important; }
    .hero-card { padding: 1.2rem !important; }
    .hero-icon { display: none !important; }
}
.brand-title    { font-size: 1.5rem; font-weight: 700; color: #1a1a1a; margin-bottom: 0; }
.brand-subtitle { font-size: 0.8rem; color: #999; margin-top: 2px; }
.upload-label   { font-size: 0.72rem; font-weight: 700; color: #6c63ff; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 8px; }
.hero-card      { background: linear-gradient(135deg, #efefff 0%, #e4e4f8 100%); border-radius: 16px; padding: 2.5rem; margin-bottom: 2rem; position: relative; overflow: hidden; }
.ai-badge       { background-color: #6c63ff; color: white; padding: 5px 14px; border-radius: 20px; font-size: 0.72rem; font-weight: 700; letter-spacing: 0.08em; display: inline-block; margin-bottom: 1rem; }
.hero-title     { font-size: 2rem; font-weight: 700; color: #1a1a1a; margin-bottom: 0.6rem; }
.hero-subtitle  { font-size: 0.95rem; color: #555; max-width: 520px; line-height: 1.6; }
.hero-icon      { position: absolute; right: 2.5rem; top: 50%; transform: translateY(-50%); font-size: 5rem; opacity: 0.10; }
.empty-state    { text-align: center; padding: 4rem 2rem; }
.empty-title    { font-size: 1.4rem; font-weight: 700; color: #2a2a2a; margin-bottom: 0.4rem; }
.empty-subtitle { font-size: 0.92rem; color: #999; line-height: 1.6; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="brand-title">⚡ AnalytIQ</p>', unsafe_allow_html=True)
    st.markdown('<p class="brand-subtitle">Intelligent CSV Analysis</p>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<p class="upload-label">Upload Data</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", type=["csv"], label_visibility="collapsed", accept_multiple_files=True)
    st.caption("Max 50 MB per file")

    if "files" not in st.session_state:
        st.session_state["files"] = {}

    for uf in uploaded_files:
        if uf.name not in st.session_state["files"]:
            with st.spinner(f"Uploading {uf.name}..."):
                try:
                    resp = post_with_retry(
                        f"{API_URL}/upload",
                        files={"file": (uf.name, uf.getvalue(), "text/csv")},
                        timeout=30
                    )
                except requests.exceptions.ConnectionError:
                    st.error("Cannot connect to backend. Make sure `python run.py` is running.")
                    st.stop()
                except requests.exceptions.Timeout:
                    st.error("Upload timed out after 3 attempts. Please try again.")
                    st.stop()
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["files"][uf.name] = {
                    "session_id": data["session_id"],
                    "columns":    data["columns"],
                    "preview":    data["preview"],
                    "shape":      data["shape"],
                }
                st.success(f"✅ {uf.name} — {data['shape']['rows']} rows loaded")
            else:
                st.error(f"Upload failed: {resp.json().get('detail')}")

    if st.session_state["files"]:
        active_file = st.selectbox("Active dataset", list(st.session_state["files"].keys()))
        st.session_state["active_file"] = active_file
        active = st.session_state["files"][active_file]
        st.session_state["session_id"] = active["session_id"]
        st.session_state["columns"]    = active["columns"]
        st.session_state["preview"]    = active["preview"]
        st.session_state["shape"]      = active["shape"]

        st.divider()
        shape = active["shape"]
        c1, c2 = st.columns(2)
        c1.metric("Rows", shape["rows"])
        c2.metric("Cols", shape["cols"])

        st.markdown("**Columns**")
        for col in active["columns"]:
            st.markdown(f"- `{col}`")

        st.divider()
        if st.button("🗑️ Clear chat"):
            st.session_state["history"] = []
            st.rerun()

# ── Hero ──────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <div class="ai-badge">AI POWERED</div>
    <div class="hero-title">Your Intelligent Data Analyst</div>
    <div class="hero-subtitle">Ask questions to your CSV data— get instant analysis, charts, and insights from your CSV data.</div>
    <div class="hero-icon">⚡</div>
</div>
""", unsafe_allow_html=True)

if not st.session_state.get("files"):
    st.markdown("""
    <div class="empty-state">
        <div style="font-size:4rem">🗄️</div>
        <div class="empty-title">No Data Loaded Yet</div>
        <div class="empty-subtitle">Upload a CSV file from the sidebar to get started.<br>Then ask anything in plain English.</div>
    </div>
    """, unsafe_allow_html=True)

else:
    with st.expander("🔍 Dataset Preview", expanded=False):
        st.dataframe(st.session_state["preview"], use_container_width=True)

    if "history" not in st.session_state:
        st.session_state["history"] = []

    # Show suggested questions only before first question is asked
    if not st.session_state["history"]:
        cols = st.session_state["columns"]
        preview = st.session_state["preview"]
        num_cols = [c for c in cols if isinstance((preview[0] if preview else {}).get(c), (int, float))]
        cat_cols = [c for c in cols if c not in num_cols]
        num_col  = num_cols[0] if num_cols else cols[0]
        cat_col  = cat_cols[0] if cat_cols else cols[-1]

        safe_num = html.escape(num_col)
        safe_cat = html.escape(cat_col)
        st.markdown(f"""
        <div style="background:#f7f7ff; border-radius:12px; padding:1.2rem 1.5rem; margin-bottom:1rem;">
            <p style="font-weight:600; color:#6c63ff; margin-bottom:0.6rem;">💡 You can ask things like:</p>
            <ul style="color:#444; line-height:2; margin:0; padding-left:1.2rem;">
                <li>What is the average <b>{safe_num}</b>?</li>
                <li>Show me the top 5 rows by <b>{safe_num}</b></li>
                <li>Compare <b>{safe_num}</b> by <b>{safe_cat}</b></li>
                <li>Give me a summary of the dataset</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    for item in st.session_state["history"]:
        with st.chat_message("user"):
            st.write(item["query"])
        with st.chat_message("assistant"):
            st.write(item["response"])
            if item.get("chart"):
                st.image(Image.open(io.BytesIO(base64.b64decode(item["chart"]))), use_container_width=True)
            

    query = st.chat_input("Ask something about your data...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        with st.chat_message("assistant"):
            stream_meta: dict = {}

            def _stream():
                """Generator: yields text chunks from /analyze_stream.
                Side-effect: populates stream_meta with chart/code/error
                from the first JSON line before streaming begins.
                """
                try:
                    resp = requests.post(
                        f"{API_URL}/analyze_stream",
                        json={
                            "session_id": st.session_state["session_id"],
                            "query":      query,
                            "history":    [
                                {"query": h["query"], "response": h["response"]}
                                for h in st.session_state["history"][-3:]
                            ],
                        },
                        stream=True,
                        timeout=90,
                    )
                except requests.exceptions.ConnectionError:
                    yield "Cannot connect to backend. Make sure `python run.py` is running."
                    return
                except requests.exceptions.Timeout:
                    yield "Request timed out. Please try again."
                    return

                if resp.status_code != 200:
                    try:
                        detail = resp.json().get("detail", "Unknown error")
                    except Exception:
                        detail = resp.text
                    yield f"Error {resp.status_code}: {detail}"
                    return

                buf       = b""
                meta_done = False
                for chunk in resp.iter_content(chunk_size=64):
                    if not chunk:
                        continue
                    if not meta_done:
                        buf += chunk
                        if b"\n" in buf:
                            meta_line, rest = buf.split(b"\n", 1)
                            try:
                                stream_meta.update(_json.loads(meta_line.decode()))
                            except Exception:
                                pass
                            meta_done = True
                            if rest:
                                yield rest.decode("utf-8", errors="replace")
                    else:
                        yield chunk.decode("utf-8", errors="replace")

            response_text = st.write_stream(_stream())

            # Show chart (available after streaming completes)
            chart_b64 = stream_meta.get("chart")
            if chart_b64:
                chart_bytes = base64.b64decode(chart_b64)
                st.image(Image.open(io.BytesIO(chart_bytes)), use_container_width=True)
                st.download_button(
                    "⬇️ Download Chart",
                    data=chart_bytes,
                    file_name="chart.png",
                    mime="image/png",
                    key=f"chart_{len(st.session_state['history'])}",
                )

            # PDF report download
            try:
                pdf_resp = post_with_retry(
                    f"{API_URL}/export_pdf",
                    json={
                        "query":    query,
                        "response": response_text or "",
                        "chart":    chart_b64,
                    },
                    timeout=20,
                )
                if pdf_resp.status_code == 200:
                    st.download_button(
                        "📄 Download PDF Report",
                        data=pdf_resp.content,
                        file_name="analytiq_report.pdf",
                        mime="application/pdf",
                        key=f"pdf_{len(st.session_state['history'])}",
                    )
            except Exception:
                pass  # PDF is non-critical; silently skip on failure

            st.session_state["history"].append({
                "query":    query,
                "response": response_text or "",
                "chart":    chart_b64,
                "code":     stream_meta.get("code"),
            })
