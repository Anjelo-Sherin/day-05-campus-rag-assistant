# Installation and Setup Guide — Campus RAG Assistant

Follow these instructions to set up, configure, run, and test Campus RAG Assistant locally or with Docker.

---

## 1. Prerequisites

- **Operating System:** macOS, Linux, or Windows (WSL recommended)
- **Python:** Version 3.9, 3.10, or 3.11
- **Memory:** Minimum 512 MB RAM (the lightweight CPU vector engine has a tiny memory footprint)
- **Git:** Version 2.20+
- **Docker** *(Optional)*: For containerized deployment

---

## 2. Quickstart (Local Virtual Environment)

### Step 1: Clone the Repository
```bash
git clone https://github.com/Anjelo-Sherin/day-05-campus-rag-assistant.git
cd day-05-campus-rag-assistant
```

### Step 2: Create and Activate Virtual Environment
```bash
# On macOS / Linux:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run Automated Tests & SQuAD Evaluation Benchmark
```bash
python -m pytest -v
```

### Step 5: Start the Application Server
```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8005
```

Open your browser to:
- **Web Dashboard:** [http://localhost:8005](http://localhost:8005)
- **Interactive OpenAPI Docs:** [http://localhost:8005/docs](http://localhost:8005/docs)

---

## 3. Configuration & Environment Variables

Copy the example configuration file:
```bash
cp .env.example .env
```

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PORT` | `8005` | HTTP server port |
| `GEMINI_API_KEY` | *(None)* | Optional Google AI Studio API key. When provided, the app uses `gemini-3.8-flash` via the official Interactions API. When omitted, it runs 100% offline using its Grounded Extractive CPU Synthesizer. |

---

## 4. Docker Deployment

Build and run using Docker:
```bash
docker build -t campus-rag-assistant:latest .
docker run -p 8005:8005 campus-rag-assistant:latest
```

Navigate to `http://localhost:8005` in your browser.
