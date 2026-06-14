from langchain_openai import ChatOpenAI
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import io
import base64
import ast

from typing import TypedDict,Any
import pandas as pd
import numpy as np
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
load_dotenv()

llm=ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2
)

class AnalystState(TypedDict):
    user_query:      str
    dataframe:       pd.DataFrame
    schema:          dict
    analysis_result: Any
    needs_chart:     bool
    intent:          str
    chart:           Any
    final_response:  str
    chat_history:    list

#define graph
graph=StateGraph(AnalystState)

#define schema node
def schema_node(state: AnalystState) -> dict:

    df = state["dataframe"]

    schema = {
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "shape": df.shape,
        "sample": df.head(3).to_dict()
    }

    return {
        "schema": schema
    }

_CHART_KEYWORDS = (
    "chart", "plot", "graph", "visualize", "visualise", "visualization",
    "pie", "donut", "bar", "line", "trend", "histogram", "scatter",
    "distribution", "breakdown", "proportion", "frequency", "spread",
    "compare", "comparison", "over time", "monthly", "yearly", "growth",
)

def query_classifier_node(state) -> dict:
    query = state["user_query"].lower()
    needs_chart = any(kw in query for kw in _CHART_KEYWORDS)
    intent = "open"
    for category in ("stats", "trend", "comparison", "filter"):
        if category in query:
            intent = category
            break
    return {"needs_chart": needs_chart, "intent": intent}


BLOCKED_NAMES = {"eval", "exec", "open", "compile", "__import__", "globals", "locals", "vars", "dir", "getattr", "setattr", "delattr", "breakpoint", "input"}

def is_safe_code(code: str) -> bool:
    """Returns True if code contains only safe pandas/numpy operations."""
    try:
        tree = ast.parse(code, mode="eval")
    except (SyntaxError, ValueError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in BLOCKED_NAMES:
                return False
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False
    return True


def dataframe_tool(question: str, df: pd.DataFrame, schema: dict = None) -> dict:

    # build schema context for the LLM
    schema_str = f"""
Columns : {list(df.columns)}
Dtypes  : {df.dtypes.to_dict()}
Shape   : {df.shape}
Sample  :
{df.head(3).to_string()}
"""

    prompt = f"""You are a strict Python/pandas expert. Your ONLY job is to write pandas expressions for questions about a specific DataFrame.

DataFrame Info:
{schema_str}

User Question: {question}

STRICT RULES:
1. The question MUST be answerable using ONLY the columns and data in the DataFrame above.
2. If the question is unrelated to this data (general knowledge, opinions, weather, coding help, etc.), respond with ONLY the word: IRRELEVANT
3. If the question references a column that does NOT exist in the DataFrame, respond with ONLY the word: IRRELEVANT
4. If the question is answerable, write a SINGLE Python expression using the variable `df`.
5. Return ONLY the expression — no explanation, no markdown, no variable assignment.

Examples of VALID responses:
- df["salary"].mean()
- df.groupby("department")["salary"].mean()
- df["age"].value_counts()

Examples that must return IRRELEVANT:
- "What is the capital of France?"
- "Write me a poem"
- "What is machine learning?"
- Any question about a column not listed above
"""

    try:
        response = llm.invoke(prompt)
    except Exception as e:
        return {"data": None, "code": None, "error": f"LLM error: {e}"}

    # clean up LLM response (sometimes wraps in ```python ... ```)
    code = response.content.strip()
    code = code.replace("```python", "").replace("```", "").strip()

    if code.strip().upper() == "IRRELEVANT":
        return {
            "data": None,
            "code": None,
            "error": "IRRELEVANT"
        }

    if not is_safe_code(code):
        return {"data": None, "code": code, "error": "Unsafe code detected — execution blocked."}

    try:
        result = eval(code, {"__builtins__": {}, "df": df, "pd": pd, "np": np})
        return {
            "data": result,
            "code": code,
            "error": None
        }
    except Exception as e:
        return {
            "data": None,
            "code": code,
            "error": str(e)
        }

#Function for analysis_node
def analysis_node(state) -> AnalystState:

    result = dataframe_tool(
        question=state["user_query"],
        df=state["dataframe"],
        schema=state.get("schema")    
    )
    return {
        "analysis_result": result
    }


# AnalytIQ brand palette — purple-led, 8 distinct colours
_PALETTE = ["#6c63ff", "#ff6584", "#43d6b5", "#ffa94d",
            "#4dabf7", "#a9e34b", "#f06595", "#74c0fc"]

def _palette(n: int) -> list[str]:
    """Return n colours, cycling through the palette."""
    return (_PALETTE * ((n // len(_PALETTE)) + 1))[:n]

def _style_ax(ax, grid_axis: str = "y") -> None:
    """Apply clean, modern styling to an axes object."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e0e0e0")
    ax.spines["bottom"].set_color("#e0e0e0")
    ax.tick_params(colors="#555555", labelsize=9)
    if grid_axis:
        ax.grid(axis=grid_axis, color="#e8e8e8", linewidth=0.8, linestyle="--")
    ax.set_axisbelow(True)


def chart_tool(analysis_result: dict, user_query: str) -> str | None:
    """Generate an attractive chart and return a base64-encoded PNG, or None."""
    query = user_query.lower()
    data  = analysis_result.get("data")

    if data is None:
        return None

    BG = "#f8f8ff"
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)

    try:
        # ── pandas Series ────────────────────────────────────
        if isinstance(data, pd.Series):
            n   = len(data)
            col = _palette(n)

            want_donut   = any(w in query for w in ["pie", "donut", "proportion", "share", "percentage"])
            want_trend   = any(w in query for w in ["trend", "over time", "monthly", "yearly", "growth"])
            want_hist    = any(w in query for w in ["histogram", "spread"])
            is_cat_index = n > 0 and (
                data.index.dtype == object
                or isinstance(data.index[0], str)
            )

            # Auto-donut: categorical index with ≤ 8 slices + distribution/breakdown keyword
            if not want_donut and is_cat_index and n <= 8:
                if any(w in query for w in ["distribution", "breakdown", "count", "frequency"]):
                    want_donut = True

            if want_donut and n >= 2:
                # ── Donut chart ──────────────────────────────
                wedges, texts, autotexts = ax.pie(
                    data.values,
                    labels=data.index.astype(str),
                    autopct="%1.1f%%",
                    colors=col,
                    startangle=90,
                    pctdistance=0.80,
                    wedgeprops=dict(width=0.52, edgecolor="white", linewidth=2),
                )
                for t in texts:
                    t.set_fontsize(9); t.set_color("#333")
                for at in autotexts:
                    at.set_fontsize(8); at.set_color("white"); at.set_fontweight("bold")
                ax.spines[:].set_visible(False)
                ax.grid(False)

            elif want_trend:
                # ── Filled line chart ────────────────────────
                xs = list(range(n))
                ax.plot(xs, data.values, color=_PALETTE[0],
                        linewidth=2.5, marker="o", markersize=5, zorder=3)
                ax.fill_between(xs, data.values, alpha=0.12, color=_PALETTE[0])
                ax.set_xticks(xs)
                ax.set_xticklabels(data.index.astype(str), rotation=45, ha="right", fontsize=8)
                _style_ax(ax, "y")

            elif want_hist:
                # ── Histogram ────────────────────────────────
                ax.hist(data.values, bins=min(20, n), color=_PALETTE[0],
                        edgecolor="white", linewidth=0.8, alpha=0.85)
                _style_ax(ax, "y")

            elif n > 8:
                # ── Horizontal bar (many categories) ─────────
                top = data.nlargest(12)
                bars = ax.barh(top.index.astype(str)[::-1], top.values[::-1],
                               color=_PALETTE[0], alpha=0.88)
                max_v = top.values.max() if len(top) else 1
                for bar, val in zip(bars, top.values[::-1]):
                    ax.text(bar.get_width() + max_v * 0.01,
                            bar.get_y() + bar.get_height() / 2,
                            f"{val:,.0f}", va="center", fontsize=8, color="#444")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color("#e0e0e0")
                ax.grid(axis="x", color="#e8e8e8", linewidth=0.8, linestyle="--")
                ax.set_axisbelow(True)

            else:
                # ── Vertical bar chart with value labels ──────
                xs = list(range(n))
                bars = ax.bar(xs, data.values, color=col,
                              edgecolor="white", linewidth=1.5, alpha=0.9)
                ax.set_xticks(xs)
                ax.set_xticklabels(data.index.astype(str),
                                   rotation=30 if n > 4 else 0, ha="right", fontsize=9)
                max_v = data.values.max() if n else 1
                for bar, val in zip(bars, data.values):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + max_v * 0.012,
                            f"{val:,.0f}" if isinstance(val, (int, float)) else str(val),
                            ha="center", va="bottom", fontsize=8, fontweight="600", color="#333")
                _style_ax(ax, "y")

        # ── pandas DataFrame ──────────────────────────────────
        elif isinstance(data, pd.DataFrame):
            cols = _PALETTE[:len(data.columns)]
            if any(w in query for w in ["trend", "over time", "monthly", "yearly"]):
                for i, col_name in enumerate(data.columns):
                    ax.plot(data.index.astype(str), data[col_name],
                            label=col_name, color=cols[i % len(cols)],
                            linewidth=2, marker="o", markersize=4)
                ax.legend(fontsize=9)
                ax.tick_params(axis="x", rotation=45)
                _style_ax(ax, "y")
            elif any(w in query for w in ["scatter", "correlation", "relationship"]):
                if len(data.columns) >= 2:
                    ax.scatter(data.iloc[:, 0], data.iloc[:, 1],
                               c=_PALETTE[0], alpha=0.7,
                               edgecolors="white", linewidth=0.5, s=55)
                    ax.set_xlabel(data.columns[0], fontsize=9)
                    ax.set_ylabel(data.columns[1], fontsize=9)
                    _style_ax(ax, "both")
                else:
                    data.plot(kind="bar", ax=ax, color=cols)
                    _style_ax(ax, "y")
            else:
                data.plot(kind="bar", ax=ax, color=cols, edgecolor="white")
                ax.tick_params(axis="x", rotation=30)
                ax.legend(fontsize=9)
                _style_ax(ax, "y")

        # ── Scalar value ──────────────────────────────────────
        elif isinstance(data, (int, float)):
            ax.bar(["Result"], [data], color=_PALETTE[0],
                         edgecolor="white", linewidth=1.5, width=0.35, alpha=0.9)
            ax.text(0, data + abs(data) * 0.03,
                    f"{data:,.2f}", ha="center", fontsize=14,
                    fontweight="bold", color=_PALETTE[0])
            ax.set_ylim(0, data * 1.25 if data > 0 else data * 0.75)
            _style_ax(ax, "y")

        else:
            plt.close(fig)
            return None

        ax.set_title(user_query[:72], fontsize=11, fontweight="bold",
                     pad=14, color="#1a1a2e")
        plt.tight_layout(pad=1.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=BG)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    except Exception as e:
        print(f"[chart_tool] Failed: {e}")
        return None

    finally:
        plt.close(fig)


def chart_node(state) -> AnalystState:

    chart = chart_tool(
        analysis_result=state["analysis_result"],
        user_query=state["user_query"]
    )

    return {
        "chart": chart
    }


def _format_analysis_data(data) -> str:
    if isinstance(data, pd.DataFrame):
        return data.to_string(index=True, max_rows=20)
    if isinstance(data, pd.Series):
        return data.to_string(max_rows=20)
    if isinstance(data, (int, float, np.integer, np.floating)):
        return str(round(float(data), 4))
    if data is None:
        return "No data returned from analysis."
    return str(data)


def build_response_prompt(user_query: str, analysis_result: dict, chart: str | None, chat_history: list) -> str:
    data_str      = _format_analysis_data(analysis_result.get("data"))
    chart_context = "A chart has been generated to visualize this." if chart else "No chart was generated."

    history_str = ""
    if chat_history:
        history_str = "\nPrevious conversation (for context only):\n"
        for h in chat_history[-3:]:
            history_str += f"Q: {h.get('query', '')}\nA: {h.get('response', '')}\n"
        history_str += "\n"

    return f"""You are a strict data analyst assistant. You ONLY answer questions based on the data provided below. You have NO access to outside knowledge.

{history_str}User Question:
{user_query}

Analysis Result:
{data_str}

{chart_context}

STRICT RULES — follow these before anything else:
- If the Analysis Result is "No data returned from analysis." OR the question is clearly unrelated to the dataset, respond ONLY with:
  "I can only answer questions about the uploaded dataset. Please ask something related to the data."
- Do NOT use any general knowledge, assumptions, or facts from outside the Analysis Result.
- Do NOT fabricate numbers, names, or trends that are not present in the Analysis Result.
- If the data is ambiguous or incomplete, say so — never guess.

If the question IS answered by the Analysis Result above, respond with:
1. Direct Answer   — answer in one sentence using only the data
2. Key Insight     — one non-obvious observation from the data
3. Recommendation  — one actionable suggestion based on the insight

Be concise, under 120 words total. Use plain language.
- Round numbers to sensible precision based on context (e.g. age → whole number, salary → 2 decimal places, percentage → 1 decimal place).
"""


def response_tool(user_query: str, analysis_result: dict, chart: str | None, llm, chat_history: list | None = None) -> str:
    prompt = build_response_prompt(user_query, analysis_result, chart, chat_history or [])
    try:
        response = llm.invoke(prompt)
    except Exception as e:
        return f"Unable to generate response due to an LLM error: {e}"
    return response.content


#Function for response_node
def response_node(state) -> AnalystState:

    response = response_tool(
        user_query=state["user_query"],
        analysis_result=state["analysis_result"],
        chart=state.get("chart"),
        llm=llm,
        chat_history=state.get("chat_history", [])
    )

    return {
        "final_response": response
    }

#Add nodes
graph.add_node("schema_node",           schema_node)
graph.add_node("query_classifier_node", query_classifier_node)
graph.add_node("analysis_node",         analysis_node)
graph.add_node("chart_node",            chart_node)
graph.add_node("response_node",         response_node)

#Add edges
graph.add_edge(START,                   "schema_node")
graph.add_edge("schema_node",           "query_classifier_node")
graph.add_edge("query_classifier_node", "analysis_node")
graph.add_conditional_edges(
    "analysis_node",
    lambda state: "chart_node" if state["needs_chart"] else "response_node"
)
graph.add_edge("chart_node",    "response_node")
graph.add_edge("response_node", END)


#graph compilation
workflow=graph.compile()

if __name__ == "__main__":
    print(workflow.invoke({
        "user_query":"What is the average salary of employees?",
        "dataframe":pd.read_csv("data.csv")
    }))
