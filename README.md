# ⚡ AnalytIQ — AI-Powered Data Analyst

AnalytIQ is an intelligent CSV analysis tool that lets you ask questions about your data in plain English and get instant answers, charts, and insights — no SQL or coding required.

Built with **LangGraph**, **FastAPI**, and **Streamlit**.

---

## Features

- **Natural Language Queries** — Ask anything about your data in plain English
- **AI-Generated Charts** — Bar, line, donut, scatter, histogram — auto-selected based on your question
- **Streaming Responses** — Word-by-word output like ChatGPT
- **Multi-File Support** — Upload and switch between multiple CSV files
- **PDF Report Export** — Download a branded PDF with your question, analysis, and chart
- **Conversation Memory** — Remembers the last 3 exchanges for follow-up questions
- **Session Persistence** — Data stored in SQLite; survives server restarts
- **Anti-Hallucination Guardrails** — Strict prompts + AST-based code validation prevent fabricated answers
- **Rate Limiting** — Max 10 requests per minute per IP
- **Password Protected UI** — Simple authentication gate

---

## Tech Stack

| Layer | Technology |
|---|---|
| AI Orchestration | LangGraph (StateGraph) |
| LLM | OpenAI GPT-4o-mini via LangChain |
| Backend API | FastAPI + Uvicorn |
| Frontend | Streamlit |
| Data Processing | Pandas, NumPy |
| Charts | Matplotlib |
| Session Storage | SQLite |
| PDF Export | fpdf2 |
| Tracing | LangSmith |
| Containerization | Docker |

---

## Project Structure

```
AI_Data_Analyst/
├── Data_Analyst_Backend.py   # LangGraph workflow, AI nodes, chart logic
├── run.py                    # FastAPI backend (upload, analyze, stream, PDF)
├── app.py                    # Streamlit frontend
├── test_backend.py           # Pytest test suite
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker config
├── .env                      # API keys (not committed)
└── sessions.db               # SQLite session store (auto-created)
```

---

## How It Works

```
User Question
      |
      v
 schema_node           reads column names, types, shape
      |
      v
 query_classifier_node  decides: needs chart? what intent?
      |
      v
 analysis_node          LLM writes pandas expression, safe eval()
      |
      |-- needs chart? --> chart_node --> matplotlib --> base64 PNG
      |
      v
 response_node          LLM writes plain-English answer
      |
      v
 Streamed to UI         text + chart + PDF download
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AI_Data_Analyst.git
cd AI_Data_Analyst
```

### 2. Create a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY="your-openai-api-key"
LANGCHAIN_API_KEY="your-langsmith-api-key"
LANGCHAIN_PROJECT="AI_Data_Analyst"
LANGCHAIN_TRACING_V2=true
```

---

## Running the App

**Terminal 1 — Start the backend:**

```bash
python run.py
```

Wait for: `Application startup complete.`

**Terminal 2 — Start the frontend:**

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

Default password: `analytiq123`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/upload` | Upload a CSV file, returns session_id |
| POST | `/analyze` | Run analysis (non-streaming) |
| POST | `/analyze_stream` | Run analysis with streaming response |
| POST | `/export_pdf` | Generate PDF report |

---

## Running with Docker

```bash
docker build -t analytiq .
docker run -p 8000:8000 -p 8501:8501 --env-file .env analytiq
```

---

## Running Tests

```bash
pytest test_backend.py -v
```

---

## Security

- `eval()` is sandboxed — no built-ins, only `df`, `pd`, `np` allowed
- AST validation blocks imports, exec, open, dunder attributes before eval runs
- Column names are HTML-escaped to prevent XSS
- CORS restricted to `localhost:8501` only

---

## Author

**Pallavi Ghule**
pallavighule472@gmail.com

---

## License

This project is for educational purposes.
