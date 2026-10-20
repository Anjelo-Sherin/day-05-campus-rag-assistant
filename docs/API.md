# API Reference — Campus RAG Assistant

The Campus RAG Assistant exposes a RESTful FastAPI backend documented via OpenAPI/Swagger at `/docs`.

Base URL: `http://127.0.0.1:8005`

---

## 1. System Endpoints

### `GET /api/health`
Checks the server and vector index status.

**Response (200 OK):**
```json
{
  "status": "ok",
  "service": "Campus RAG Assistant",
  "embedding_backend": "Hybrid-MiniLM-Hash-1024d (CPU)",
  "indexed_documents": 4,
  "total_chunks": 20
}
```

---

## 2. Document Knowledge Base Endpoints

### `GET /api/documents`
Lists all documents currently indexed in the Vector Store with page counts and chunk statistics.

**Response (200 OK):**
```json
{
  "documents": [
    {
      "file_name": "Academic_Regulations_Handbook.pdf",
      "doc_type": "pdf",
      "page_count": 3,
      "chunk_count": 6,
      "total_chars": 2790
    },
    {
      "file_name": "CS301_AI_Syllabus_2026.pdf",
      "doc_type": "pdf",
      "page_count": 3,
      "chunk_count": 5,
      "total_chars": 2257
    }
  ],
  "total_documents": 4,
  "total_chunks": 20,
  "embedding_backend": "Hybrid-MiniLM-Hash-1024d (CPU)"
}
```

### `GET /api/documents/{file_name}/chunks`
Inspects all indexed chunks belonging to a specific file.

### `DELETE /api/documents/{file_name}`
Removes a document from the vector store and rebuilds the BM25 index.

### `POST /api/upload`
Uploads and indexes a new document (`multipart/form-data`).

**Parameters:**
- `file`: The document file (`.pdf`, `.pptx`, `.docx`, `.html`, `.md`, `.txt`).
- `chunk_size` *(int, default: 550)*: Target character count per chunk.
- `chunk_overlap` *(int, default: 120)*: Overlap character count.

### `POST /api/ingest-url`
Ingests a campus web notice or raw HTML string.

**Request Body:**
```json
{
  "url": "Exam_Notice_Nov2026.html",
  "html_content": "<h1>End-Semester Examination Circular</h1><p>...</p>",
  "chunk_size": 550,
  "chunk_overlap": 120
}
```

### `POST /api/reset-corpus`
Rebuilds the vector store using the default campus documents with custom chunk parameters.

---

## 3. RAG Query & Chat Endpoints

### `POST /api/chat`
Performs hybrid search, cross-feature reranking, out-of-scope guardrail checks, and answer synthesis.

**Request Body:**
```json
{
  "query": "What is the minimum attendance percentage required and medical condonation?",
  "top_k": 4,
  "alpha": 0.55,
  "use_reranker": true,
  "refusal_threshold": 0.28,
  "file_filter": null,
  "api_key": null
}
```

**Response (200 OK — Grounded Answer):**
```json
{
  "query": "What is the minimum attendance percentage required and medical condonation?",
  "answer": "Article 4.2: Medical Leave and Attendance Condonation. [Academic_Regulations_Handbook.pdf, p. 1]\n\n**Supporting Details from Campus Documents:**\n- Every enrolled undergraduate and postgraduate student must maintain a minimum attendance of 75% separately in every registered theory and laboratory course. [Academic_Regulations_Handbook.pdf, p. 1]",
  "is_refusal": false,
  "refusal_reason": null,
  "confidence_score": 0.605,
  "faithfulness_score": 0.88,
  "latency_ms": 2.4,
  "generator_backend": "Grounded Extractive Synthesizer (CPU)",
  "citations": [
    {
      "citation_id": 1,
      "file_name": "Academic_Regulations_Handbook.pdf",
      "page_number": 1,
      "section_title": "Article 4.1: Mandatory Class Attendance Requirement",
      "doc_type": "pdf",
      "snippet": "University Academic Regulations Handbook (2026-2027 Edition)...",
      "rerank_score": 0.605,
      "dense_score": 0.627,
      "bm25_score": 1.0
    }
  ],
  "retrieved_chunks": [...]
}
```

**Response (200 OK — Out-of-Scope Refusal):**
```json
{
  "query": "What is the recipe for baking a classic sourdough pizza crust in a wood-fired oven?",
  "answer": "**Out-of-Scope Refusal (No Hallucination Policy):**\nI could not find reliable evidence to answer this question in the uploaded campus documents...",
  "is_refusal": true,
  "refusal_reason": "Retrieval confidence (0.03) below refusal threshold (0.28)",
  "confidence_score": 0.0308,
  "faithfulness_score": 1.0,
  "latency_ms": 1.45,
  "generator_backend": "Strict Grounding Guardrail",
  "citations": [],
  "retrieved_chunks": [...]
}
```

---

## 4. Evaluation Benchmark Endpoint

### `POST /api/evaluate`
Executes the automated 14-question SQuAD-style benchmark.

**Request Body:**
```json
{
  "top_k": 4,
  "alpha": 0.55,
  "use_reranker": true,
  "refusal_threshold": 0.28
}
```

**Response (200 OK):**
```json
{
  "summary": {
    "total_questions": 14,
    "in_scope_questions": 10,
    "out_of_scope_questions": 4,
    "hit_rate_at_1": 1.0,
    "hit_rate_at_k": 1.0,
    "mrr": 1.0,
    "mean_faithfulness": 0.8715,
    "out_of_scope_refusal_rate": 1.0,
    "overall_pass_rate": 1.0,
    "avg_query_latency_ms": 1.6
  },
  "results": [...]
}
```
