import pandas as pd
import pytest
from Data_Analyst_Backend import is_safe_code, dataframe_tool, chart_tool


# ── is_safe_code tests ────────────────────────────────────
def test_safe_code_valid():
    assert is_safe_code('df["salary"].mean()') is True

def test_safe_code_blocks_import():
    assert is_safe_code('__import__("os").system("rm -rf /")') is False

def test_safe_code_blocks_exec():
    assert is_safe_code('exec("import os")') is False

def test_safe_code_blocks_open():
    assert is_safe_code('open(".env").read()') is False

def test_safe_code_blocks_dunder():
    assert is_safe_code('df.__class__.__bases__') is False

def test_safe_code_invalid_syntax():
    assert is_safe_code('df[[[') is False


# ── dataframe_tool tests ──────────────────────────────────
@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "name":       ["Alice", "Bob", "Charlie"],
        "salary":     [50000, 60000, 70000],
        "department": ["HR", "IT", "IT"],
    })

def test_dataframe_tool_irrelevant(sample_df):
    result = dataframe_tool("What is the capital of France?", sample_df)
    assert result["error"] == "IRRELEVANT"
    assert result["data"] is None

def test_dataframe_tool_unsafe_code(sample_df, monkeypatch):
    # force LLM to return dangerous code
    class FakeResponse:
        content = '__import__("os")'
    monkeypatch.setattr("Data_Analyst_Backend.llm.invoke", lambda _: FakeResponse())
    result = dataframe_tool("anything", sample_df)
    assert result["error"] == "Unsafe code detected — execution blocked."


# ── chart_tool tests ──────────────────────────────────────
def test_chart_tool_returns_none_on_no_data():
    result = chart_tool({"data": None}, "show chart")
    assert result is None

def test_chart_tool_returns_base64_for_series(sample_df):
    series_result = {"data": sample_df["salary"]}
    chart = chart_tool(series_result, "show salary distribution")
    assert chart is not None
    assert isinstance(chart, str)  # base64 string

def test_chart_tool_returns_base64_for_scalar():
    scalar_result = {"data": 55000.0}
    chart = chart_tool(scalar_result, "average salary")
    assert chart is not None
