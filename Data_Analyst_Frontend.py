import streamlit as st
import pandas as pd
import base64
import io
from PIL import Image

@st.cache_data(show_spinner=False)
def load_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))

@st.cache_data(show_spinner=False)
def get_df_stats(df: pd.DataFrame) -> tuple:
    null_pct = round(df.isnull().sum().sum() / df.size * 100, 1)
    numeric_cols = len(df.select_dtypes("number").columns)
    return null_pct, numeric_cols

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

* { font-family: 'Inter', sans-serif; }

/* ── Background ── */
.stApp {
    background: #fafafa;
    background-image:
        radial-gradient(ellipse at 10% 0%, rgba(139, 92, 246, 0.06) 0%, transparent 40%),
        radial-gradient(ellipse at 90% 100%, rgba(16, 185, 129, 0.05) 0%, transparent 40%);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #f0f0f0;
    box-shadow: 2px 0 20px rgba(0,0,0,0.04);
}

#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar Brand ── */
.sidebar-brand {
    font-size: 19px;
    font-weight: 700;
    color: #0f172a;
}
.sidebar-sub {
    font-size: 11px;
    color: #94a3b8;
    margin-top: 2px;
    letter-spacing: 0.4px;
}

/* ── Section Label ── */
.section-label {
    font-size: 11px;
    font-weight: 700;
    color: #8b5cf6;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    margin-bottom: 10px;
}

/* ── Hero Header ── */
.hero-header {
    background: linear-gradient(135deg, #f5f3ff 0%, #ede9fe 40%, #e0f2fe 100%);
    border: 1px solid #e5e7eb;
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-header::after {
    content: '⚡';
    position: absolute;
    right: 40px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 72px;
    opacity: 0.12;
}
.hero-title {
    font-size: 30px;
    font-weight: 700;
    color: #0f172a;
    margin: 0 0 8px 0;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(90deg, #8b5cf6, #6366f1);
    color: white;
    font-size: 11px;
    font-weight: 600;
    padding: 3px 12px;
    border-radius: 20px;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
    text-transform: uppercase;
}
.hero-sub {
    font-size: 15px;
    color: #64748b;
    margin: 0;
    font-weight: 400;
    max-width: 500px;
}

/* ── Metric Cards ── */
.metric-card {
    background: #ffffff;
    border: 1px solid #f1f5f9;
    border-radius: 16px;
    padding: 20px 22px;
    text-align: left;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 12px rgba(0,0,0,0.03);
    transition: box-shadow 0.2s, transform 0.2s;
}
.metric-card:hover {
    box-shadow: 0 4px 20px rgba(139, 92, 246, 0.12);
    transform: translateY(-2px);
}
.metric-icon {
    font-size: 20px;
    margin-bottom: 10px;
    display: block;
}
.metric-value {
    font-size: 24px;
    font-weight: 700;
    color: #0f172a;
    display: block;
}
.metric-label {
    font-size: 12px;
    color: #94a3b8;
    margin-top: 3px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Schema rows ── */
.schema-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 7px 0;
    border-bottom: 1px solid #f8fafc;
    font-size: 13px;
}
.schema-col { color: #1e293b; font-weight: 500; }
.schema-dtype {
    background: #f5f3ff;
    color: #7c3aed;
    border-radius: 5px;
    padding: 2px 9px;
    font-size: 11px;
    font-weight: 600;
}

/* ── File pill ── */
.file-pill {
    display: inline-block;
    background: #f5f3ff;
    color: #7c3aed;
    border: 1px solid #ede9fe;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 12px;
    font-weight: 600;
}

/* ── Query Input ── */
.stTextInput > div > div > input {
    background: #ffffff !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 12px !important;
    color: #0f172a !important;
    font-size: 15px !important;
    padding: 14px 18px !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04) !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #8b5cf6 !important;
    box-shadow: 0 0 0 3px rgba(139,92,246,0.1) !important;
}
.stTextInput > div > div > input::placeholder { color: #9ca3af !important; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 10px;
    font-size: 13px;
    font-weight: 500;
    border: 1.5px solid #e5e7eb;
    background: #ffffff;
    color: #374151;
    transition: all 0.18s;
    box-shadow: 0 1px 2px rgba(0,0,0,0.04);
}
.stButton > button:hover {
    border-color: #8b5cf6;
    color: #7c3aed;
    background: #f5f3ff;
    box-shadow: 0 2px 8px rgba(139,92,246,0.12);
}

/* ── Primary Button ── */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #8b5cf6, #6366f1) !important;
    color: white !important;
    border: none !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 12px rgba(139,92,246,0.3) !important;
}
[data-testid="stButton"] button[kind="primary"]:hover {
    background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
    box-shadow: 0 6px 18px rgba(139,92,246,0.4) !important;
    transform: translateY(-1px);
}

/* ── Answer Card ── */
.response-card {
    background: #ffffff;
    border: 1px solid #ede9fe;
    border-top: 3px solid #8b5cf6;
    border-radius: 0 0 14px 14px;
    padding: 22px 26px;
    font-size: 15px;
    line-height: 1.8;
    color: #1e293b;
    box-shadow: 0 4px 16px rgba(139,92,246,0.07);
}
.response-label {
    background: linear-gradient(90deg, #8b5cf6, #6366f1);
    color: white;
    font-size: 11px;
    font-weight: 700;
    padding: 5px 16px;
    border-radius: 8px 8px 0 0;
    text-transform: uppercase;
    letter-spacing: 1px;
    display: inline-block;
    margin-bottom: 0;
}

/* ── History Item ── */
.history-item {
    background: #ffffff;
    border: 1px solid #f1f5f9;
    border-left: 3px solid #e0e7ff;
    border-radius: 0 10px 10px 0;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 13px;
    color: #64748b;
    transition: border-left-color 0.2s;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
}
.history-item:hover { border-left-color: #8b5cf6; }
.history-q { color: #0f172a; font-weight: 600; margin-bottom: 4px; }

/* ── Empty State ── */
.empty-state {
    text-align: center;
    padding: 100px 20px;
}
.empty-icon  { font-size: 64px; margin-bottom: 16px; }
.empty-title { font-size: 22px; font-weight: 700; color: #1e293b; margin-bottom: 8px; }
.empty-sub   { font-size: 14px; color: #94a3b8; }

/* ── Error Card ── */
.error-card {
    background: #fff5f5;
    border: 1px solid #fecaca;
    border-left: 3px solid #ef4444;
    border-radius: 0 10px 10px 0;
    padding: 14px 18px;
    font-size: 14px;
    color: #b91c1c;
}

/* ── Divider ── */
hr { border-color: #f1f5f9 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Backend Import ─────────────────────────────────────────────────────────
from Data_Analyst_Backend import workflow

# ─── Session State ──────────────────────────────────────────────────────────
if "dataframes"  not in st.session_state: st.session_state.dataframes  = {}
if "history"     not in st.session_state: st.session_state.history     = []
if "active_file" not in st.session_state: st.session_state.active_file = None
if "result"      not in st.session_state: st.session_state.result      = None
if "last_query"  not in st.session_state: st.session_state.last_query  = ""

# ─── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">⚡AnalytIQ</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Intelligent CSV Analysis</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown('<div class="section-label">Upload Data</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        label="",
        type=["csv"],
        accept_multiple_files=True,
        help="Upload one or more CSV files",
        label_visibility="collapsed"
    )

    if uploaded:
        for f in uploaded:
            if f.name not in st.session_state.dataframes:
                try:
                    st.session_state.dataframes[f.name] = load_csv(f.getvalue())
                except Exception as e:
                    st.error(f"Could not read {f.name}: {e}")

        uploaded_names = {f.name for f in uploaded}
        st.session_state.dataframes = {
            k: v for k, v in st.session_state.dataframes.items()
            if k in uploaded_names
        }

    if st.session_state.dataframes:
        st.markdown('<div class="section-label" style="margin-top:16px">Active File</div>', unsafe_allow_html=True)
        selected = st.selectbox(
            label="",
            options=list(st.session_state.dataframes.keys()),
            label_visibility="collapsed"
        )
        st.session_state.active_file = selected
        df_active = st.session_state.dataframes[selected]

        st.divider()

        st.markdown('<div class="section-label">Preview</div>', unsafe_allow_html=True)
        st.dataframe(df_active.head(5), use_container_width=True, hide_index=True)

        st.divider()

        st.markdown('<div class="section-label">Schema</div>', unsafe_allow_html=True)
        schema_html = ""
        for col, dtype in df_active.dtypes.items():
            schema_html += f"""
            <div class="schema-row">
                <span class="schema-col">{col}</span>
                <span class="schema-dtype">{dtype}</span>
            </div>"""
        st.markdown(schema_html, unsafe_allow_html=True)

        st.divider()

        st.markdown('<div class="section-label">Stats</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Rows",    f"{len(df_active):,}")
        c2.metric("Columns", len(df_active.columns))

        null_pct, numeric_cols = get_df_stats(df_active)
        c3, c4 = st.columns(2)
        c3.metric("Numeric", numeric_cols)
        c4.metric("Nulls %", f"{null_pct}%")

        st.divider()

        if st.button("Clear History", use_container_width=True):
            st.session_state.history = []
            st.session_state.result  = None
            st.rerun()

# ─── Main Area ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <div class="hero-badge">AI Powered</div>
    <div class="hero-title">Your Intelligent Data Analyst</div>
    <p class="hero-sub">Ask questions to your CSV data— get instant analysis, charts, and insights from your CSV data.</p>
</div>
""", unsafe_allow_html=True)

# ── No files loaded
if not st.session_state.dataframes:
    st.markdown("""
    <div class="empty-state">
        <div class="empty-icon">🗄️</div>
        <div class="empty-title">No Data Loaded Yet</div>
        <p class="empty-sub">Upload a CSV file from the sidebar to get started.<br>Then ask anything in plain English.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Metric Bar
df_active = st.session_state.dataframes[st.session_state.active_file]
m1, m2, m3, m4 = st.columns(4)

for col, icon, label, value in [
    (m1, "🗂️", "Files Loaded",  str(len(st.session_state.dataframes))),
    (m2, "📋", "Total Rows",    f"{len(df_active):,}"),
    (m3, "🔢", "Columns",       str(len(df_active.columns))),
    (m4, "🎯", "Active File",   st.session_state.active_file.replace(".csv", "")),
]:
    with col:
        st.markdown(f"""
        <div class="metric-card">
            <span class="metric-icon">{icon}</span>
            <span class="metric-value">{value}</span>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Query Input
st.markdown('<div class="section-label">Ask a Question</div>', unsafe_allow_html=True)
st.markdown(
    f"Querying: <span class='file-pill'>{st.session_state.active_file}</span>",
    unsafe_allow_html=True
)

col_q, col_btn = st.columns([5, 1])
with col_q:
    query = st.text_input(
        label="",
        placeholder="e.g. What is the average salary by department?",
        value=st.session_state.last_query,
        label_visibility="collapsed"
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("Analyze →", type="primary", use_container_width=True)

# ── Sample Questions
st.markdown('<div class="section-label" style="margin-top:16px">Quick Questions</div>', unsafe_allow_html=True)
sq1, sq2, sq3 = st.columns(3)
samples = [
    "What is the average salary by department?",
    "Show salary distribution by department",
    "Which department has the most employees?"
]
for col, sample in zip([sq1, sq2, sq3], samples):
    with col:
        if st.button(sample, use_container_width=True, key=f"sq_{sample}"):
            query = sample
            run   = True

st.divider()

# ── Run Analysis
if run and query.strip():
    st.session_state.last_query = query
    with st.spinner("Analyzing your data..."):
        try:
            result = workflow.invoke({
                "user_query": query.strip(),
                "dataframe":  df_active
            })
            st.session_state.result = result
            st.session_state.history.insert(0, {
                "query":    query.strip(),
                "response": result.get("final_response", ""),
                "file":     st.session_state.active_file
            })
        except Exception as e:
            st.markdown(
                f'<div class="error-card">❌ Analysis failed: {e}</div>',
                unsafe_allow_html=True
            )
            st.session_state.result = None

elif run and not query.strip():
    st.warning("Please enter a question before clicking Analyze.")

# ── Results
if st.session_state.result:
    result = st.session_state.result

    st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)

    if result.get("chart"):
        res_col, chart_col = st.columns([1, 1])
    else:
        res_col   = st.container()
        chart_col = None

    with res_col:
        st.markdown('<span class="response-label">Answer</span>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="response-card">{result.get("final_response", "No response generated.")}</div>',
            unsafe_allow_html=True
        )

        analysis = result.get("analysis_result", {})
        if analysis.get("code"):
            with st.expander("View Generated Code", expanded=False):
                st.code(analysis["code"], language="python")

        if analysis.get("error"):
            st.markdown(
                f'<div class="error-card">⚠️ {analysis["error"]}</div>',
                unsafe_allow_html=True
            )

    if chart_col and result.get("chart"):
        with chart_col:
            st.markdown('<div class="section-label">Chart</div>', unsafe_allow_html=True)
            img_bytes = base64.b64decode(result["chart"])
            img       = Image.open(io.BytesIO(img_bytes))
            st.image(img, use_container_width=True)

# ── Query History
if st.session_state.history:
    st.divider()
    st.markdown('<div class="section-label">History</div>', unsafe_allow_html=True)

    for item in st.session_state.history[:8]:
        st.markdown(f"""
        <div class="history-item">
            <div class="history-q">↳ {item['query']}</div>
            <span class="file-pill">{item['file']}</span>
            <span style="font-size:12px; color:#94a3b8; margin-left:8px;">
                {item['response'][:120]}{'...' if len(item['response']) > 120 else ''}
            </span>
        </div>
        """, unsafe_allow_html=True)
