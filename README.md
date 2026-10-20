# Day 05 — Campus RAG Assistant: Chat with Your Documents

> **Ask questions over syllabus, academic regulations, and campus circulars — with clickable page-level citations and strict out-of-scope hallucination refusal.**
>
> **Domain:** Generative AI / LLM / Retrieval-Augmented Generation | **Level:** Intermediate (30 Projects in 30 Days)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg)](https://fastapi.tiangolo.com)
[![Pytest](https://img.shields.io/badge/Pytest-Passing-brightgreen.svg)](tests/)

---

## 1. Application Preview

### 💬 Chat with Documents & Page-Level Citations
Ask complex academic and administrative questions; receive grounded answers with clickable citation chips (`[file_name, p. X]`) that immediately reveal the exact supporting chunk, dense similarity, BM25 lexical score, and cross-feature reranker weights in the right-hand Inspector.

![Chat & Citation Inspector](screenshots/02_interactive_citations_query.png)

---

### 📂 Document Admin & Knowledge Base Management
Upload multi-page PDFs, PowerPoint slides (`.pptx`), Word documents (`.docx`), HTML web notices, or plain text. Adjust chunk size and overlap on the fly, inspect chunk boundaries, and manage the campus knowledge base with zero database overhead.

![Admin & Corpus Management Panel](screenshots/02_admin_corpus_panel.png)

---

### 📊 SQuAD-Style RAG Evaluation & Hallucination Benchmark
Run real-time automated evaluations over a 14-question ground-truth test set (10 in-scope syllabus/regulations queries + 4 adversarial out-of-scope traps) measuring Hit-Rate@1, Hit-Rate@K, Mean Reciprocal Rank (MRR), Faithfulness, and Out-of-Scope Refusal Accuracy.

![RAG Evaluation Harness Studio](screenshots/03_rag_eval_harness.png)

---

## 2. Problem Statement

University students and faculty spend hours sifting through sprawling PDFs, course syllabi, residence hall guidelines, library manuals, and circulars to answer crucial questions (e.g., *"What is the late penalty for CS-301 assignments?"*, *"What attendance percentage is mandatory before debarment?"*, *"What are the hostel night pass rules?"*).

Standard LLMs suffer from two critical flaws:
1. **Hallucination:** When asked about specific campus policies not in their training data, they invent believable but incorrect rules.
2. **Lack of Provenance:** Basic RAG demos flatten documents into text, stripping physical page numbers so users cannot verify claims.

**Campus RAG Assistant** provides:
- **Physical Page-Level Citations:** Every claim is tied directly to `[file_name, p. X]`.
- **Hybrid Dense + Lexical Retrieval:** Combines dense semantic representations with Okapi BM25 and a Stage-2 Cross-Feature Reranker to capture both semantic paraphrases and exact codes (`CS-301`, `75%`, `Rs. 500`).
- **Strict Out-of-Scope Guardrails:** If retrieval confidence falls below threshold (`0.28`), the assistant explicitly refuses to answer rather than hallucinating.
- **Built-in SQuAD Benchmark:** Automatically measures retrieval hit-rate, MRR, and factual faithfulness.

---

## 3. Architecture & Data Flow

```mermaid
flowchart LR
    A["1. Upload Docs\nPDF, PPTX, DOCX, HTML"] --> B["2. Chunk + Embed\nOverlapping splits + CPU Vectors"]
    B --> C["3. Vector DB\nFAISS / Local Vector Index\n+ Okapi BM25 Index"]
    C --> D["4. Retriever + Rerank\nHybrid Dense+BM25\n+ Cross-Feature Reranker"]
    D --> E["5. LLM Answer & Guardrail\nGemini 3.8 Flash / CPU Synth\n+ [file, p. X] Citations"]
    E --> F["6. Eval Harness\nHit-Rate@1/K, MRR,\nFaithfulness & Refusal"]
```

For detailed architectural specifications, data models, and reranking formulas, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 4. Key Features

- **Multi-Format Ingestion:** Native extraction for PDF literal streams, XML PowerPoint slides (`.pptx`), Word documents (`.docx`), and HTML pages with structured heading detection.
- **Overlapping Sentence Chunker:** 550-character chunks with 120-character overlap snapped to sentence boundaries so rules and exceptions remain intact.
- **Hybrid Retrieval:** Dense Cosine ($\alpha = 0.55$) + query-saturated Okapi BM25 ($0.45$) to guarantee both broad conceptual matching and exact number/acronym hits.
- **Cross-Feature Reranking:** 5-factor reranker scoring exact token coverage ($30\%$), bigram phrases ($15\%$), section title alignment ($15\%$), and numeric/code matches ($12\%$) with single-token false-positive penalties.
- **Dual Synthesis Backends:**
  - **Google Gemini (`gemini-3.8-flash`):** Official `google-genai` Interactions API with structured JSON output when `GEMINI_API_KEY` is provided.
  - **Grounded Extractive CPU Synthesizer:** High-precision deterministic local engine that runs in $< 2\text{ ms}$ on CPU with zero network calls and 100% offline support.
- **Interactive Citation Inspector:** Clickable chips in the UI display exact chunk text, page boundaries, dense similarity, and BM25 scores.
- **Comprehensive SQuAD-Style Evaluation Harness:** Measures Hit-Rate@1, Hit-Rate@K, MRR, and answer faithfulness with one click.

---

## 5. Technology Stack

| Component | Technology | Rationale |
| :--- | :--- | :--- |
| **Backend & Web Server** | FastAPI, Uvicorn, Pydantic | High-performance asynchronous REST endpoints, automatic OpenAPI schema (`/docs`). |
| **Frontend UI** | HTML5, Tailwind CSS, Vanilla JS | Ultra-fast single-file dashboard with no external JavaScript build step required. |
| **Vector Storage** | Local Vector Store (FAISS / NumPy L2) | Zero paid infrastructure; fast persistence in `data/vector_store/`. |
| **Retrieval Engine** | Hybrid Dense Cosine + Okapi BM25 | Handles conceptual queries and exact alphanumeric terms (`CS-301`, `75%`). |
| **Embeddings** | 1024-dim CPU Neural-Hash / MiniLM | Extremely fast vectorization on CPU without requiring GPU acceleration. |
| **LLM Generation** | Google Gemini (`gemini-3.8-flash`) / Local Synth | Official Interactions API when API key provided; local extractive synthesizer offline. |
| **Testing & Evaluation** | Pytest, TestClient | Automated 14-query ground-truth benchmark and full endpoint test suite. |

---

## 6. Directory Structure

```text
day-05-campus-rag-assistant/
├── data/
│   ├── sample_docs/              # Pre-loaded campus PDFs & HTML documents
│   │   ├── Academic_Regulations_Handbook.pdf
│   │   ├── CS301_AI_Syllabus_2026.pdf
│   │   ├── Campus_Hostel_Library_Circular.pdf
│   │   └── Scholarship_Internship_Guidelines.html
│   └── vector_store/             # Persistent chunks & L2-normalized embeddings
├── docs/                         # Extended project documentation
│   ├── API.md                    # REST API endpoints & request/response specs
│   ├── ARCHITECTURE.md           # System architecture, data flow & formulas
│   └── SETUP.md                  # Comprehensive installation & setup guide
├── screenshots/                  # Verified application screenshots
│   ├── 01_chat_citations_dashboard.png
│   ├── 02_admin_corpus_panel.png
│   ├── 02_interactive_citations_query.png
│   └── 03_rag_eval_harness.png
├── src/
│   ├── __init__.py
│   ├── api.py                    # FastAPI server & interactive web dashboard
│   ├── campus_corpus.py          # Campus document generators & 14-question benchmark
│   ├── document_parser.py        # PDF, PPTX, DOCX, HTML parser & chunker
│   ├── evaluator.py              # SQuAD evaluation harness (Hit-Rate, MRR, Faithfulness)
│   ├── rag_engine.py             # Grounded answer synthesis & refusal guardrail
│   └── vector_store.py           # Hybrid Dense+BM25 index & cross-feature reranker
├── tests/
│   └── test_rag_assistant.py     # Automated pytest test suite
├── .env.example                  # Environment configuration template
├── .gitignore                    # Excludes caches, venvs, and secrets
├── Dockerfile                    # Container definition
├── LICENSE                       # MIT License
├── README.md                     # Main documentation
└── requirements.txt              # Project dependencies
```

---

## 7. How to Run in 3 Commands

```bash
# 1. Clone repository and navigate to folder
git clone https://github.com/Anjelo-Sherin/day-05-campus-rag-assistant.git
cd day-05-campus-rag-assistant

# 2. Set up virtual environment and install dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Run tests and launch server
python -m pytest -v
python -m uvicorn src.api:app --host 0.0.0.0 --port 8005
```

Open your browser to:
- **Interactive Web App:** [http://localhost:8005](http://localhost:8005)
- **API Swagger Documentation:** [http://localhost:8005/docs](http://localhost:8005/docs)

*(Optional)* To enable live **Google Gemini (`gemini-3.8-flash`)** generation:
```bash
export GEMINI_API_KEY="your-gemini-api-key"
```

---

## 8. Evaluation Benchmark Results

Tested across the **14-question SQuAD-style Campus Benchmark** (10 in-scope syllabus and academic policy queries with ground-truth citations + 4 out-of-scope adversarial questions):

| Metric | Measured Score | Description |
| :--- | :--- | :--- |
| **Retrieval Hit-Rate @ 1** | **`100.0%` (`1.0000`)** | Target document & page ranked `#1` for every in-scope query |
| **Retrieval Hit-Rate @ 4** | **`100.0%` (`1.0000`)** | Target document & page present in top-4 retrieved chunks |
| **Mean Reciprocal Rank (MRR)** | **`1.0000`** | Average reciprocal rank of the ground-truth page |
| **Mean Answer Faithfulness** | **`87.15%` (`0.8715`)** | Fraction of substantive answer claims grounded directly in cited chunks |
| **Out-of-Scope Refusal Accuracy** | **`100.0%` (`4 / 4`)** | Adversarial questions (pizza recipe, FIFA World Cup, stock price, visa) refused without hallucinating |
| **Overall Pass Rate** | **`100.0%` (`14 / 14`)** | All queries passed retrieval, citation, and refusal checks |
| **Average Query Latency (CPU)** | **`1.61 ms`** | End-to-end hybrid retrieval, reranking, and grounded synthesis |

---

## 9. Interview Talking Points & Design Decisions

1. **Chunk Size vs. Overlap Trade-offs:**
   - Small chunks (`< 200` chars) isolate rules from exceptions (e.g. separating the `75%` attendance rule from the `65%` medical condonation clause).
   - Large chunks (`> 1500` chars) introduce noise and weaken page-level citation precision.
   - We selected **`550` characters** with **`120` characters overlap** snapped to sentence boundaries, yielding `1.00` MRR across the corpus.
2. **Why Hybrid Search (Dense + BM25) Is Essential:**
   - Pure vector embeddings struggle with exact alphanumeric tokens such as course numbers (`CS-301`), room numbers (`LHC-101`), and percentages (`75%`, `15%`). Blending dense vectors with Okapi BM25 ensures both conceptual paraphrases and exact codes rank `#1`.
3. **Preventing Single-Token Spikes in Hallucination Guardrails:**
   - Standard max-normalized BM25 can give a `1.0` score if an out-of-scope query contains a single common word (e.g. *"penalty"* in *"FIFA World Cup penalty shootout"* matching the late penalty section). Normalizing BM25 by query length saturation ($3.2 \times \sqrt{|Q|}$) and enforcing a 25% token coverage minimum raised refusal accuracy from $25\%$ to $100\%$.

---

## 10. Responsible Use & Disclaimers

This software is an educational decision-support tool designed to assist campus communities in navigating published documents. It is not an official university authority. For official grade appeals, formal disciplinary actions, or graduation clearance, always consult the University Academic Section.

---

## 11. License

This project is open-source under the [MIT License](LICENSE).
