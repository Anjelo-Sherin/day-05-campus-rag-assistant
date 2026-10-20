"""
Day 05: Campus RAG Assistant - Chat with Your Documents.
Interactive Web UI + REST API Server with:
1. Chat with Campus Documents (Interactive Citation Chips, Source Page Inspector, Hallucination Refusal Badge)
2. Document Admin Panel (Upload PDFs, PPTX slides, DOCX, HTML/Text, or Ingest URLs, Adjustable Chunk Size & Overlap)
3. RAG Evaluation & Ablation Studio (SQuAD-style 14-question benchmark measuring Hit-Rate@1, Hit-Rate@K, MRR, Faithfulness, and Refusal Rate)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from src.campus_corpus import EVALUATION_BENCHMARK_SET, seed_default_corpus
from src.document_parser import chunk_pages, parse_document, parse_url_or_html
from src.evaluator import run_evaluation_suite
from src.rag_engine import CampusRAGEngine
from src.vector_store import CampusVectorStore

BASE_DIR = Path(__file__).resolve().parent.parent
SAMPLE_DOCS_DIR = BASE_DIR / "data" / "sample_docs"
VECTOR_STORE_DIR = BASE_DIR / "data" / "vector_store"

vector_store = CampusVectorStore(storage_dir=VECTOR_STORE_DIR)
seed_default_corpus(vector_store, SAMPLE_DOCS_DIR, chunk_size=550, chunk_overlap=120)
rag_engine = CampusRAGEngine(vector_store=vector_store, refusal_threshold=0.28)

app = FastAPI(
    title="Campus RAG Assistant API",
    description="Day 05: Retrieval-Augmented Generation over Campus Syllabus, Regulations & Circulars with Page-Level Citations & Evaluation Suite.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Question to ask over campus documents")
    top_k: int = Field(4, ge=1, le=10)
    alpha: float = Field(0.55, ge=0.0, le=1.0, description="Dense vs BM25 weight (1.0=Dense, 0.0=BM25)")
    use_reranker: bool = Field(True)
    refusal_threshold: float = Field(0.28, ge=0.05, le=0.85)
    file_filter: Optional[List[str]] = Field(None)
    api_key: Optional[str] = Field(None, description="Optional Gemini API key for live gemini-3.8-flash generation")


class UrlIngestRequest(BaseModel):
    url: str
    html_content: Optional[str] = None
    chunk_size: int = Field(550, ge=150, le=2000)
    chunk_overlap: int = Field(120, ge=0, le=500)


class RebuildConfigRequest(BaseModel):
    chunk_size: int = Field(550, ge=150, le=2000)
    chunk_overlap: int = Field(120, ge=0, le=500)


class EvalRunRequest(BaseModel):
    top_k: int = Field(4, ge=1, le=10)
    alpha: float = Field(0.55, ge=0.0, le=1.0)
    use_reranker: bool = Field(True)
    refusal_threshold: float = Field(0.28, ge=0.05, le=0.85)


@app.get("/api/health")
def health_check() -> Dict[str, Any]:
    docs = vector_store.get_document_stats()
    return {
        "status": "ok",
        "service": "Campus RAG Assistant",
        "embedding_backend": vector_store.embedder.backend_name,
        "indexed_documents": len(docs),
        "total_chunks": len(vector_store.chunks),
    }


@app.get("/api/documents")
def list_documents() -> Dict[str, Any]:
    docs = vector_store.get_document_stats()
    return {
        "documents": docs,
        "total_documents": len(docs),
        "total_chunks": len(vector_store.chunks),
        "embedding_backend": vector_store.embedder.backend_name,
    }


@app.get("/api/documents/{file_name}/chunks")
def get_document_chunks(file_name: str) -> Dict[str, Any]:
    matching = [c.to_dict() for c in vector_store.chunks if c.file_name == file_name]
    if not matching:
        raise HTTPException(status_code=404, detail=f"Document '{file_name}' not found in vector index")
    return {"file_name": file_name, "chunks": matching}


@app.delete("/api/documents/{file_name}")
def delete_document(file_name: str) -> Dict[str, Any]:
    removed = vector_store.remove_document(file_name)
    if removed == 0:
        raise HTTPException(status_code=404, detail=f"Document '{file_name}' not found")
    return {
        "removed_chunks": removed,
        "file_name": file_name,
        "remaining_chunks": len(vector_store.chunks),
    }


@app.post("/api/upload")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(550),
    chunk_overlap: int = Form(120),
) -> Dict[str, Any]:
    raw_bytes = await file.read()
    fname = file.filename or "uploaded_document.txt"
    pages = parse_document(fname, raw_bytes)
    if not pages:
        raise HTTPException(status_code=400, detail=f"Could not extract readable text from '{fname}'.")

    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 4)

    new_chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    added = vector_store.add_chunks(new_chunks, replace_existing_file=True)

    # Save uploaded file to sample_docs for persistence
    SAMPLE_DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (SAMPLE_DOCS_DIR / fname).write_bytes(raw_bytes)

    return {
        "status": "indexed",
        "file_name": fname,
        "pages_extracted": len(pages),
        "chunks_indexed": added,
        "total_chunks_in_store": len(vector_store.chunks),
    }


@app.post("/api/ingest-url")
def ingest_url_endpoint(req: UrlIngestRequest) -> Dict[str, Any]:
    try:
        pages = parse_url_or_html(req.url, html_override=req.html_content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to fetch or parse URL: {exc}")

    if not pages:
        raise HTTPException(status_code=400, detail="No text content extracted from URL/HTML")

    overlap = min(req.chunk_overlap, req.chunk_size - 20)
    new_chunks = chunk_pages(pages, chunk_size=req.chunk_size, chunk_overlap=overlap)
    added = vector_store.add_chunks(new_chunks, replace_existing_file=True)

    return {
        "status": "indexed",
        "file_name": pages[0].file_name,
        "pages_extracted": len(pages),
        "chunks_indexed": added,
        "total_chunks_in_store": len(vector_store.chunks),
    }


@app.post("/api/reset-corpus")
def reset_default_corpus(req: RebuildConfigRequest) -> Dict[str, Any]:
    overlap = min(req.chunk_overlap, req.chunk_size - 20)
    total = seed_default_corpus(
        vector_store,
        SAMPLE_DOCS_DIR,
        chunk_size=req.chunk_size,
        chunk_overlap=overlap,
        force_rebuild=True,
    )
    return {
        "status": "rebuilt",
        "chunk_size": req.chunk_size,
        "chunk_overlap": overlap,
        "total_chunks": total,
        "documents": vector_store.get_document_stats(),
    }


@app.post("/api/chat")
def chat_endpoint(req: ChatRequest) -> Dict[str, Any]:
    ans = rag_engine.answer_question(
        query=req.query,
        top_k=req.top_k,
        alpha=req.alpha,
        use_reranker=req.use_reranker,
        refusal_threshold=req.refusal_threshold,
        file_filter=req.file_filter,
        api_key=req.api_key,
    )
    return ans.to_dict()


@app.post("/api/evaluate")
def evaluate_endpoint(req: EvalRunRequest) -> Dict[str, Any]:
    return run_evaluation_suite(
        rag_engine=rag_engine,
        top_k=req.top_k,
        alpha=req.alpha,
        use_reranker=req.use_reranker,
        refusal_threshold=req.refusal_threshold,
    )


@app.get("/api/benchmark-questions")
def get_benchmark_questions() -> Dict[str, Any]:
    return {"questions": EVALUATION_BENCHMARK_SET}


UI_HTML = r"""<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Campus RAG Assistant — Day 05 | Chat with Your Documents</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Inter', sans-serif; background-color: #0b0f19; color: #e2e8f0; }
    .mono { font-family: 'JetBrains Mono', monospace; }
    .glass { background: rgba(15, 23, 42, 0.78); backdrop-filter: blur(12px); border: 1px solid rgba(51, 65, 85, 0.55); }
    .citation-chip {
      display: inline-flex; align-items: center; gap: 4px;
      padding: 2px 8px; border-radius: 9999px; font-size: 0.75rem; font-weight: 600;
      background: rgba(99, 102, 241, 0.18); color: #a5b4fc; border: 1px solid rgba(99, 102, 241, 0.45);
      cursor: pointer; transition: all 0.15s ease;
    }
    .citation-chip:hover { background: rgba(99, 102, 241, 0.35); color: #ffffff; transform: translateY(-1px); }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: #0f172a; }
    ::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
  </style>
</head>
<body class="min-h-screen flex flex-col">
  <!-- Top Header -->
  <header class="border-b border-slate-800 bg-slate-950/90 sticky top-0 z-30">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 via-purple-500 to-pink-500 flex items-center justify-center shadow-lg shadow-indigo-500/20 font-bold text-white text-lg">
          05
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h1 class="font-bold text-lg text-white tracking-tight">Campus RAG Assistant</h1>
            <span class="text-xs px-2 py-0.5 rounded-full bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 font-medium">Day 05 • GenAI / LLM</span>
          </div>
          <p class="text-xs text-slate-400">Ask questions over syllabus, regulations & circulars — with page-level citations & hallucination guardrails</p>
        </div>
      </div>

      <!-- Navigation Tabs -->
      <div class="flex items-center gap-1.5 bg-slate-900 p-1 rounded-xl border border-slate-800">
        <button onclick="switchTab('chat')" id="tab-btn-chat" class="px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all bg-indigo-600 text-white shadow">
          💬 Chat & Citations
        </button>
        <button onclick="switchTab('admin')" id="tab-btn-admin" class="px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all text-slate-400 hover:text-white">
          📂 Admin & Corpus (<span id="nav-doc-count">4</span>)
        </button>
        <button onclick="switchTab('eval')" id="tab-btn-eval" class="px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all text-slate-400 hover:text-white">
          📊 RAG Eval Harness
        </button>
      </div>
    </div>
  </header>

  <!-- Main Content -->
  <main class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 py-6">

    <!-- ==================== TAB 1: CHAT & CITATIONS ==================== -->
    <section id="tab-chat" class="grid grid-cols-1 lg:grid-cols-12 gap-6">
      <!-- Left Column: Chat Window (7 cols) -->
      <div class="lg:col-span-7 flex flex-col glass rounded-2xl overflow-hidden h-[78vh]">
        <!-- Sample Quick Prompts Bar -->
        <div class="p-3.5 border-b border-slate-800/80 bg-slate-900/60">
          <div class="flex items-center justify-between mb-2">
            <span class="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Quick Campus Queries (Click to Test)</span>
            <span id="engine-badge" class="text-[11px] mono text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">Hybrid Dense+BM25 + Reranker</span>
          </div>
          <div class="flex flex-wrap gap-1.5">
            <button onclick="askPreset('What is the minimum attendance required in every course and what is the medical relaxation?')" class="text-xs px-2.5 py-1 rounded-lg bg-slate-800/90 hover:bg-indigo-600/30 hover:border-indigo-500/50 border border-slate-700 text-slate-200 transition">
              📋 Attendance & Medical Rule
            </button>
            <button onclick="askPreset('How is the final grade weighted in CS-301 Artificial Intelligence and what is the late submission penalty?')" class="text-xs px-2.5 py-1 rounded-lg bg-slate-800/90 hover:bg-indigo-600/30 hover:border-indigo-500/50 border border-slate-700 text-slate-200 transition">
              🎓 CS-301 Grading & Slip Days
            </button>
            <button onclick="askPreset('What are the hostel gate curfew timings and library overdue book fines?')" class="text-xs px-2.5 py-1 rounded-lg bg-slate-800/90 hover:bg-indigo-600/30 hover:border-indigo-500/50 border border-slate-700 text-slate-200 transition">
              🏛️ Hostel Curfew & Library Fine
            </button>
            <button onclick="askPreset('What are the eligibility criteria and stipend for the Merit-cum-Means Scholarship?')" class="text-xs px-2.5 py-1 rounded-lg bg-slate-800/90 hover:bg-indigo-600/30 hover:border-indigo-500/50 border border-slate-700 text-slate-200 transition">
              💡 MCM Scholarship
            </button>
            <button onclick="askPreset('What is the recipe for baking a classic sourdough pizza crust in a wood-fired oven?')" class="text-xs px-2.5 py-1 rounded-lg bg-rose-950/60 hover:bg-rose-600/30 border border-rose-700/50 text-rose-200 transition">
              🛡️ Test Out-of-Scope Refusal
            </button>
          </div>
        </div>

        <!-- Chat Messages Stream -->
        <div id="chat-messages" class="flex-1 overflow-y-auto p-4 space-y-4">
          <div class="p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-sm text-slate-300 leading-relaxed">
            <div class="font-semibold text-white mb-1 flex items-center gap-2">
              <span>🎓 Welcome to Campus RAG Assistant</span>
            </div>
            Ask any question about <span class="text-indigo-400 font-medium">CS-301 AI Syllabus</span>,
            <span class="text-indigo-400 font-medium">Academic Regulations Handbook</span>,
            <span class="text-indigo-400 font-medium">Hostel & Library Circulars</span>, or
            <span class="text-indigo-400 font-medium">Scholarships & Internships</span>.
            Every answer includes clickable <span class="citation-chip">file.pdf • p. X</span> chips that inspect the exact source page on the right, and out-of-scope questions are strictly refused instead of hallucinated.
          </div>
        </div>

        <!-- Chat Input Bar -->
        <form onsubmit="submitChat(event)" class="p-3.5 border-t border-slate-800 bg-slate-950/80 flex gap-2">
          <input
            id="chat-input"
            type="text"
            placeholder="Ask a question about syllabus, attendance, grading, hostel rules, scholarships..."
            class="flex-1 bg-slate-900 border border-slate-700/80 rounded-xl px-4 py-2.5 text-sm text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500"
            required
          />
          <button
            id="send-btn"
            type="submit"
            class="px-5 py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-sm transition shadow-lg shadow-indigo-600/25"
          >
            Ask RAG →
          </button>
        </form>
      </div>

      <!-- Right Column: Source Citation Inspector & Retrieval Controls (5 cols) -->
      <div class="lg:col-span-5 flex flex-col gap-5">
        <!-- Source Page & Chunk Inspector -->
        <div class="glass rounded-2xl p-4 flex-1 flex flex-col">
          <div class="flex items-center justify-between border-b border-slate-800 pb-3 mb-3">
            <div>
              <h2 class="text-sm font-bold text-white flex items-center gap-2">
                <span>🔍 Page-Level Citation & Evidence Inspector</span>
              </h2>
              <p class="text-xs text-slate-400">Click any citation chip in chat to inspect the exact retrieved chunk & scores</p>
            </div>
            <span id="active-faithfulness-badge" class="text-xs mono px-2.5 py-1 rounded-full bg-slate-800 text-slate-300 border border-slate-700">
              Ready
            </span>
          </div>

          <div id="citation-inspector" class="flex-1 overflow-y-auto space-y-3 max-h-[42vh]">
            <div class="text-xs text-slate-500 italic p-6 text-center border border-dashed border-slate-800 rounded-xl">
              Submit a question on the left to inspect retrieved top-k chunks, dense cosine similarity, BM25 lexical scores, and cross-feature reranker weights.
            </div>
          </div>
        </div>

        <!-- RAG Hyperparameters & Gemini Key Box -->
        <div class="glass rounded-2xl p-4 space-y-3">
          <div class="flex items-center justify-between">
            <h3 class="text-xs font-bold uppercase tracking-wider text-slate-300">⚙️ Retrieval & Guardrail Controls</h3>
            <span class="text-[11px] mono text-indigo-400" id="param-readout">k=4 | α=0.55 | thresh=0.28</span>
          </div>

          <div class="grid grid-cols-3 gap-3 text-xs">
            <div>
              <label class="block text-slate-400 mb-1">Top-K Chunks</label>
              <input id="ctrl-topk" type="number" min="1" max="8" value="4" onchange="updateParamReadout()" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white mono" />
            </div>
            <div>
              <label class="block text-slate-400 mb-1">Hybrid α (Dense)</label>
              <input id="ctrl-alpha" type="number" step="0.05" min="0" max="1" value="0.55" onchange="updateParamReadout()" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white mono" />
            </div>
            <div>
              <label class="block text-slate-400 mb-1">Refusal Thresh</label>
              <input id="ctrl-thresh" type="number" step="0.02" min="0.10" max="0.70" value="0.28" onchange="updateParamReadout()" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-2.5 py-1.5 text-white mono" />
            </div>
          </div>

          <div class="flex items-center justify-between pt-1">
            <label class="flex items-center gap-2 text-xs text-slate-300 cursor-pointer">
              <input id="ctrl-rerank" type="checkbox" checked class="rounded bg-slate-900 border-slate-700 text-indigo-600" />
              <span>Enable Stage-2 Cross-Feature Reranker</span>
            </label>
          </div>

          <div class="pt-1 border-t border-slate-800/80">
            <label class="block text-[11px] text-slate-400 mb-1">Optional Gemini API Key (uses <code class="text-indigo-300">gemini-3.8-flash</code> Interactions API; works offline if blank)</label>
            <input id="ctrl-apikey" type="password" placeholder="AIzaSy... (Leave empty for built-in Grounded CPU Synthesizer)" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-1.5 text-xs text-white mono" />
          </div>
        </div>
      </div>
    </section>

    <!-- ==================== TAB 2: ADMIN & CORPUS PANEL ==================== -->
    <section id="tab-admin" class="hidden space-y-6">
      <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <!-- Upload & Ingest Panel (5 cols) -->
        <div class="lg:col-span-5 space-y-5">
          <div class="glass rounded-2xl p-5 space-y-4">
            <h2 class="text-base font-bold text-white flex items-center gap-2">
              <span>📤 Upload Campus Document (PDF, PPTX, DOCX, HTML, MD)</span>
            </h2>
            <p class="text-xs text-slate-400">
              Documents are parsed with page/slide numbers, split into overlapping chunks, embedded on CPU, and indexed in the local Vector Store.
            </p>
            <form onsubmit="uploadFile(event)" class="space-y-3">
              <input id="upload-file-input" type="file" accept=".pdf,.pptx,.docx,.html,.htm,.md,.txt" required class="block w-full text-xs text-slate-300 file:mr-3 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-xs file:font-semibold file:bg-indigo-600 file:text-white hover:file:bg-indigo-500 bg-slate-900/90 border border-slate-800 rounded-xl p-2" />
              <div class="grid grid-cols-2 gap-3 text-xs">
                <div>
                  <label class="block text-slate-400 mb-1">Chunk Size (chars)</label>
                  <input id="upload-chunk-size" type="number" value="550" min="200" max="1800" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-white mono" />
                </div>
                <div>
                  <label class="block text-slate-400 mb-1">Chunk Overlap (chars)</label>
                  <input id="upload-chunk-overlap" type="number" value="120" min="0" max="450" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-white mono" />
                </div>
              </div>
              <button type="submit" class="w-full py-2.5 rounded-xl bg-indigo-600 hover:bg-indigo-500 text-white font-semibold text-xs transition">
                Parse, Chunk & Index Document
              </button>
            </form>
            <div id="upload-status" class="text-xs hidden p-3 rounded-xl"></div>
          </div>

          <!-- Paste Circular / Webpage HTML Ingest -->
          <div class="glass rounded-2xl p-5 space-y-3">
            <h3 class="text-sm font-bold text-white">🌐 Ingest Campus Circular / Web Notice</h3>
            <input id="ingest-url-name" type="text" placeholder="Circular Name or URL (e.g. Exam_Notice_Nov2026.html)" class="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-white" />
            <textarea id="ingest-html-body" rows="4" placeholder="Paste circular text or HTML notice content here..." class="w-full bg-slate-900 border border-slate-700 rounded-lg p-3 text-xs text-white"></textarea>
            <button onclick="ingestCustomNotice()" class="w-full py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition">
              Index Notice into Vector Store
            </button>
          </div>
        </div>

        <!-- Indexed Documents Table & Chunk Explorer (7 cols) -->
        <div class="lg:col-span-7 space-y-5">
          <div class="glass rounded-2xl p-5">
            <div class="flex flex-wrap items-center justify-between gap-2 mb-4">
              <div>
                <h2 class="text-base font-bold text-white">📚 Indexed Campus Knowledge Base</h2>
                <p class="text-xs text-slate-400" id="store-summary-line">Loading vector index statistics...</p>
              </div>
              <button onclick="rebuildDefaultCorpus()" class="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 border border-slate-700 transition">
                🔄 Reset / Re-Chunk Default Corpus
              </button>
            </div>

            <div class="overflow-x-auto">
              <table class="w-full text-left border-collapse text-xs">
                <thead>
                  <tr class="border-b border-slate-800 text-slate-400 uppercase">
                    <th class="py-2.5 px-3">Document File</th>
                    <th class="py-2.5 px-3">Type</th>
                    <th class="py-2.5 px-3">Pages</th>
                    <th class="py-2.5 px-3">Chunks</th>
                    <th class="py-2.5 px-3">Characters</th>
                    <th class="py-2.5 px-3 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody id="docs-table-body" class="divide-y divide-slate-800/60"></tbody>
              </table>
            </div>
          </div>

          <!-- Document Chunk Modal / Drawer -->
          <div class="glass rounded-2xl p-5">
            <h3 class="text-sm font-bold text-white mb-2" id="chunk-preview-title">📄 Document Chunk Inspector (Select a file above)</h3>
            <div id="chunk-preview-list" class="space-y-2.5 max-h-[34vh] overflow-y-auto text-xs text-slate-400">
              Click "Inspect Chunks" on any indexed document above to verify overlapping chunk boundaries and page numbers.
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ==================== TAB 3: EVALUATION HARNESS ==================== -->
    <section id="tab-eval" class="hidden space-y-6">
      <div class="glass rounded-2xl p-6">
        <div class="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 pb-4 mb-5">
          <div>
            <h2 class="text-lg font-bold text-white flex items-center gap-2">
              <span>📊 SQuAD-Style RAG Evaluation & Hallucination Benchmark</span>
            </h2>
            <p class="text-xs text-slate-400">
              Evaluates 14 ground-truth test queries (10 in-scope campus policy/syllabus questions with exact file+page targets + 4 out-of-scope adversarial traps)
            </p>
          </div>
          <button onclick="runEvaluation()" id="run-eval-btn" class="px-5 py-2.5 rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 hover:from-indigo-500 hover:to-purple-500 text-white font-semibold text-xs shadow-lg shadow-indigo-600/30 transition">
            ▶ Run Full Evaluation Benchmark
          </button>
        </div>

        <!-- KPI Metric Cards -->
        <div class="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5 mb-6">
          <div class="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800">
            <div class="text-[11px] text-slate-400 uppercase font-semibold">Hit-Rate @ 1</div>
            <div id="eval-hr1" class="text-2xl font-bold text-emerald-400 mono mt-1">—</div>
            <div class="text-[11px] text-slate-500">Exact Page Top-1</div>
          </div>
          <div class="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800">
            <div class="text-[11px] text-slate-400 uppercase font-semibold">Hit-Rate @ K</div>
            <div id="eval-hrk" class="text-2xl font-bold text-indigo-400 mono mt-1">—</div>
            <div class="text-[11px] text-slate-500">Exact Page in Top-4</div>
          </div>
          <div class="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800">
            <div class="text-[11px] text-slate-400 uppercase font-semibold">MRR Score</div>
            <div id="eval-mrr" class="text-2xl font-bold text-purple-400 mono mt-1">—</div>
            <div class="text-[11px] text-slate-500">Mean Reciprocal Rank</div>
          </div>
          <div class="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800">
            <div class="text-[11px] text-slate-400 uppercase font-semibold">Faithfulness</div>
            <div id="eval-faith" class="text-2xl font-bold text-cyan-400 mono mt-1">—</div>
            <div class="text-[11px] text-slate-500">Context Grounding</div>
          </div>
          <div class="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800">
            <div class="text-[11px] text-slate-400 uppercase font-semibold">Refusal Accuracy</div>
            <div id="eval-refusal" class="text-2xl font-bold text-amber-400 mono mt-1">—</div>
            <div class="text-[11px] text-slate-500">Out-of-Scope Guardrail</div>
          </div>
          <div class="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800">
            <div class="text-[11px] text-slate-400 uppercase font-semibold">Avg Latency</div>
            <div id="eval-latency" class="text-2xl font-bold text-slate-200 mono mt-1">—</div>
            <div class="text-[11px] text-slate-500">Per RAG Query</div>
          </div>
        </div>

        <!-- Per-Question Diagnostic Table -->
        <div class="overflow-x-auto">
          <table class="w-full text-left border-collapse text-xs">
            <thead>
              <tr class="border-b border-slate-800 text-slate-400 uppercase">
                <th class="py-2.5 px-2.5">ID</th>
                <th class="py-2.5 px-2.5">Question</th>
                <th class="py-2.5 px-2.5">Expected Citation</th>
                <th class="py-2.5 px-2.5">Top-1 Retrieved</th>
                <th class="py-2.5 px-2.5">Rank</th>
                <th class="py-2.5 px-2.5">Faithfulness</th>
                <th class="py-2.5 px-2.5">Status</th>
              </tr>
            </thead>
            <tbody id="eval-table-body" class="divide-y divide-slate-800/60">
              <tr>
                <td colspan="7" class="py-8 text-center text-slate-500 italic">
                  Click "Run Full Evaluation Benchmark" above to compute live retrieval Hit-Rate, MRR, and Faithfulness metrics.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>
    </section>
  </main>

  <script>
    let lastRetrievedChunks = [];

    function switchTab(tab) {
      ['chat', 'admin', 'eval'].forEach(t => {
        document.getElementById('tab-' + t).classList.toggle('hidden', t !== tab);
        const btn = document.getElementById('tab-btn-' + t);
        if (t === tab) {
          btn.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all bg-indigo-600 text-white shadow';
        } else {
          btn.className = 'px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all text-slate-400 hover:text-white';
        }
      });
      if (tab === 'admin') loadDocuments();
      if (tab === 'eval' && document.getElementById('eval-hr1').textContent === '—') {
        runEvaluation();
      }
    }

    function updateParamReadout() {
      const k = document.getElementById('ctrl-topk').value;
      const a = document.getElementById('ctrl-alpha').value;
      const th = document.getElementById('ctrl-thresh').value;
      document.getElementById('param-readout').textContent = `k=${k} | α=${a} | thresh=${th}`;
    }

    function escapeHtml(str) {
      return (str || '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    function formatAnswerWithChips(text) {
      let safe = escapeHtml(text);
      safe = safe.replace(/\*\*(.*?)\*\*/g, '<strong class="text-white font-semibold">$1</strong>');
      safe = safe.replace(/`([^`]+)`/g, '<code class="mono text-indigo-300 bg-slate-800 px-1.5 py-0.5 rounded text-xs">$1</code>');
      safe = safe.replace(/\[([^\],]+),\s*p\.\s*(\d+)\]/g, (match, fname, page) => {
        const cleanFile = fname.trim();
        return `<span onclick="highlightChunk('${cleanFile}', ${page})" class="citation-chip" title="Click to inspect ${cleanFile} Page ${page}">📄 ${cleanFile} • p.${page}</span>`;
      });
      return safe.replace(/\n/g, '<br/>');
    }

    function askPreset(q) {
      document.getElementById('chat-input').value = q;
      submitChat(new Event('submit'));
    }

    async function submitChat(e) {
      if (e) e.preventDefault();
      const inputEl = document.getElementById('chat-input');
      const query = inputEl.value.trim();
      if (!query) return;

      const chatBox = document.getElementById('chat-messages');
      chatBox.insertAdjacentHTML('beforeend', `
        <div class="flex justify-end">
          <div class="max-w-[82%] rounded-2xl rounded-br-sm bg-indigo-600 text-white px-4 py-2.5 text-sm shadow">
            ${escapeHtml(query)}
          </div>
        </div>
      `);
      inputEl.value = '';
      chatBox.scrollTop = chatBox.scrollHeight;

      const top_k = parseInt(document.getElementById('ctrl-topk').value || '4', 10);
      const alpha = parseFloat(document.getElementById('ctrl-alpha').value || '0.55');
      const refusal_threshold = parseFloat(document.getElementById('ctrl-thresh').value || '0.28');
      const use_reranker = document.getElementById('ctrl-rerank').checked;
      const api_key = document.getElementById('ctrl-apikey').value.trim() || null;

      const sendBtn = document.getElementById('send-btn');
      sendBtn.disabled = true;
      sendBtn.textContent = 'Searching...';

      try {
        const res = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ query, top_k, alpha, use_reranker, refusal_threshold, api_key })
        });
        const data = await res.json();
        lastRetrievedChunks = data.retrieved_chunks || [];

        const badgeColor = data.is_refusal
          ? 'bg-amber-500/15 text-amber-300 border-amber-500/40'
          : 'bg-emerald-500/15 text-emerald-300 border-emerald-500/40';
        const statusLabel = data.is_refusal
          ? '🛡️ Out-of-Scope Refusal (Hallucination Prevented)'
          : `✅ Grounded Answer • Faithfulness ${(data.faithfulness_score * 100).toFixed(0)}%`;

        let citationsRow = '';
        if (data.citations && data.citations.length > 0) {
          citationsRow = `<div class="mt-3 pt-2.5 border-t border-slate-800 flex flex-wrap items-center gap-1.5">
            <span class="text-[11px] text-slate-400 mr-1">Cited Sources:</span>
            ${data.citations.map(c => `<span onclick="highlightChunk('${c.file_name}', ${c.page_number})" class="citation-chip">📄 ${c.file_name} • p.${c.page_number} (${(c.rerank_score*100).toFixed(0)}%)</span>`).join('')}
          </div>`;
        }

        chatBox.insertAdjacentHTML('beforeend', `
          <div class="p-4 rounded-2xl bg-slate-900/95 border ${data.is_refusal ? 'border-amber-500/40' : 'border-slate-800'} text-sm space-y-2 shadow">
            <div class="flex flex-wrap items-center justify-between gap-2 text-xs">
              <span class="px-2.5 py-0.5 rounded-full border font-medium ${badgeColor}">${statusLabel}</span>
              <span class="mono text-[11px] text-slate-400">${data.generator_backend} • ${data.latency_ms} ms</span>
            </div>
            <div class="text-slate-200 leading-relaxed pt-1">${formatAnswerWithChips(data.answer)}</div>
            ${citationsRow}
          </div>
        `);
        chatBox.scrollTop = chatBox.scrollHeight;
        renderCitationInspector(data);
      } catch (err) {
        chatBox.insertAdjacentHTML('beforeend', `<div class="p-3 rounded-xl bg-rose-950 border border-rose-700 text-xs text-rose-200">Error: ${err}</div>`);
      } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = 'Ask RAG →';
      }
    }

    function renderCitationInspector(data) {
      const box = document.getElementById('citation-inspector');
      const badge = document.getElementById('active-faithfulness-badge');
      badge.textContent = `Conf: ${(data.confidence_score * 100).toFixed(0)}% | Faithfulness: ${(data.faithfulness_score * 100).toFixed(0)}%`;

      if (!data.retrieved_chunks || data.retrieved_chunks.length === 0) {
        box.innerHTML = `<div class="text-xs text-slate-500 p-4">No chunks retrieved.</div>`;
        return;
      }

      box.innerHTML = data.retrieved_chunks.map((rc) => `
        <div id="chunk-card-${rc.file_name}-${rc.page_number}" class="p-3.5 rounded-xl bg-slate-900/90 border border-slate-800 space-y-2 transition-all">
          <div class="flex items-center justify-between gap-2">
            <div class="flex items-center gap-2">
              <span class="px-2 py-0.5 rounded bg-indigo-500/20 text-indigo-300 font-bold mono text-xs">#${rc.rank}</span>
              <span class="font-semibold text-xs text-white">${escapeHtml(rc.file_name)}</span>
              <span class="text-xs px-2 py-0.5 rounded-full bg-slate-800 text-indigo-300 mono">Page ${rc.page_number}</span>
            </div>
            <span class="text-xs mono font-semibold text-emerald-400">Score: ${rc.rerank_score.toFixed(3)}</span>
          </div>
          <div class="text-[11px] text-slate-400 font-medium">${escapeHtml(rc.section_title)}</div>
          <p class="text-xs text-slate-300 leading-relaxed bg-slate-950/70 p-2.5 rounded-lg border border-slate-800/80">${escapeHtml(rc.text)}</p>
          <div class="flex flex-wrap items-center justify-between text-[11px] mono text-slate-400 pt-1">
            <span>Dense: ${rc.dense_score.toFixed(3)} | BM25: ${rc.bm25_score.toFixed(3)} | Hybrid: ${rc.hybrid_score.toFixed(3)}</span>
            <span>Matched: ${(rc.matched_terms || []).join(', ') || 'none'}</span>
          </div>
        </div>
      `).join('');
    }

    function highlightChunk(fileName, pageNum) {
      const target = document.getElementById(`chunk-card-${fileName}-${pageNum}`);
      if (target) {
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        target.classList.add('ring-2', 'ring-indigo-500');
        setTimeout(() => target.classList.remove('ring-2', 'ring-indigo-500'), 1800);
      }
    }

    async function loadDocuments() {
      const res = await fetch('/api/documents');
      const data = await res.json();
      document.getElementById('nav-doc-count').textContent = data.total_documents;
      document.getElementById('store-summary-line').textContent =
        `${data.total_documents} documents (${data.total_chunks} overlapping chunks) • Embedding: ${data.embedding_backend}`;

      const tbody = document.getElementById('docs-table-body');
      tbody.innerHTML = data.documents.map(d => `
        <tr class="hover:bg-slate-900/50">
          <td class="py-2.5 px-3 font-medium text-white">${escapeHtml(d.file_name)}</td>
          <td class="py-2.5 px-3"><span class="px-2 py-0.5 rounded bg-slate-800 text-slate-300 mono uppercase text-[10px]">${d.doc_type}</span></td>
          <td class="py-2.5 px-3 mono">${d.page_count}</td>
          <td class="py-2.5 px-3 mono text-indigo-400 font-semibold">${d.chunk_count}</td>
          <td class="py-2.5 px-3 mono text-slate-400">${d.total_chars.toLocaleString()}</td>
          <td class="py-2.5 px-3 text-right space-x-2">
            <button onclick="inspectDocChunks('${d.file_name}')" class="px-2.5 py-1 rounded bg-indigo-600/20 hover:bg-indigo-600/40 text-indigo-300 border border-indigo-500/30">Inspect Chunks</button>
            <button onclick="deleteDoc('${d.file_name}')" class="px-2 py-1 rounded bg-rose-600/20 hover:bg-rose-600/40 text-rose-300 border border-rose-500/30">Delete</button>
          </td>
        </tr>
      `).join('');
    }

    async function inspectDocChunks(fileName) {
      const res = await fetch(`/api/documents/${encodeURIComponent(fileName)}/chunks`);
      const data = await res.json();
      document.getElementById('chunk-preview-title').textContent = `📄 Chunks in ${fileName} (${data.chunks.length} chunks)`;
      document.getElementById('chunk-preview-list').innerHTML = data.chunks.map(c => `
        <div class="p-3 rounded-xl bg-slate-900 border border-slate-800 space-y-1">
          <div class="flex justify-between mono text-[11px] text-indigo-300">
            <span>${escapeHtml(c.chunk_id)}</span>
            <span>Page ${c.page_number} • chars ${c.char_start}..${c.char_end}</span>
          </div>
          <div class="text-slate-200 leading-relaxed">${escapeHtml(c.text)}</div>
        </div>
      `).join('');
    }

    async function deleteDoc(fileName) {
      await fetch(`/api/documents/${encodeURIComponent(fileName)}`, { method: 'DELETE' });
      loadDocuments();
    }

    async function uploadFile(e) {
      e.preventDefault();
      const fileInput = document.getElementById('upload-file-input');
      if (!fileInput.files.length) return;
      const formData = new FormData();
      formData.append('file', fileInput.files[0]);
      formData.append('chunk_size', document.getElementById('upload-chunk-size').value);
      formData.append('chunk_overlap', document.getElementById('upload-chunk-overlap').value);

      const statusEl = document.getElementById('upload-status');
      statusEl.className = 'text-xs p-3 rounded-xl bg-slate-900 border border-slate-700 text-slate-300 block';
      statusEl.textContent = 'Parsing pages and generating embeddings...';

      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (res.ok) {
        statusEl.className = 'text-xs p-3 rounded-xl bg-emerald-950/80 border border-emerald-600/40 text-emerald-200 block';
        statusEl.textContent = `✅ Indexed ${data.file_name}: ${data.pages_extracted} pages → ${data.chunks_indexed} overlapping chunks.`;
        fileInput.value = '';
        loadDocuments();
      } else {
        statusEl.className = 'text-xs p-3 rounded-xl bg-rose-950 border border-rose-700 text-rose-200 block';
        statusEl.textContent = `❌ ${data.detail || 'Upload failed'}`;
      }
    }

    async function ingestCustomNotice() {
      const name = document.getElementById('ingest-url-name').value.trim() || 'Campus_Notice.html';
      const body = document.getElementById('ingest-html-body').value.trim();
      if (!body) return;
      const res = await fetch('/api/ingest-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ url: name, html_content: body, chunk_size: 550, chunk_overlap: 120 })
      });
      if (res.ok) {
        document.getElementById('ingest-html-body').value = '';
        loadDocuments();
      }
    }

    async function rebuildDefaultCorpus() {
      const chunk_size = parseInt(document.getElementById('upload-chunk-size').value || '550', 10);
      const chunk_overlap = parseInt(document.getElementById('upload-chunk-overlap').value || '120', 10);
      await fetch('/api/reset-corpus', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ chunk_size, chunk_overlap })
      });
      loadDocuments();
    }

    async function runEvaluation() {
      const btn = document.getElementById('run-eval-btn');
      btn.disabled = true;
      btn.textContent = 'Running 14-Query Benchmark...';
      try {
        const top_k = parseInt(document.getElementById('ctrl-topk').value || '4', 10);
        const alpha = parseFloat(document.getElementById('ctrl-alpha').value || '0.55');
        const refusal_threshold = parseFloat(document.getElementById('ctrl-thresh').value || '0.28');
        const use_reranker = document.getElementById('ctrl-rerank').checked;

        const res = await fetch('/api/evaluate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ top_k, alpha, use_reranker, refusal_threshold })
        });
        const data = await res.json();
        const s = data.summary;

        document.getElementById('eval-hr1').textContent = (s.hit_rate_at_1 * 100).toFixed(0) + '%';
        document.getElementById('eval-hrk').textContent = (s.hit_rate_at_k * 100).toFixed(0) + '%';
        document.getElementById('eval-mrr').textContent = s.mrr.toFixed(3);
        document.getElementById('eval-faith').textContent = (s.mean_faithfulness * 100).toFixed(0) + '%';
        document.getElementById('eval-refusal').textContent = (s.out_of_scope_refusal_rate * 100).toFixed(0) + '%';
        document.getElementById('eval-latency').textContent = s.avg_query_latency_ms.toFixed(1) + ' ms';

        const tbody = document.getElementById('eval-table-body');
        tbody.innerHTML = data.results.map(r => `
          <tr class="hover:bg-slate-900/60">
            <td class="py-2.5 px-2.5 mono font-semibold text-indigo-300">${r.id}</td>
            <td class="py-2.5 px-2.5 text-slate-200 max-w-md">${escapeHtml(r.question)}</td>
            <td class="py-2.5 px-2.5 mono text-[11px] text-slate-300">${escapeHtml(r.expected_citation)}</td>
            <td class="py-2.5 px-2.5 mono text-[11px] text-indigo-300">${escapeHtml(r.top_retrieved_citation)}</td>
            <td class="py-2.5 px-2.5 mono">${r.exact_page_rank !== null ? '#' + r.exact_page_rank : (r.is_refusal ? 'Refused' : 'Miss')}</td>
            <td class="py-2.5 px-2.5 mono text-cyan-300">${(r.faithfulness * 100).toFixed(0)}%</td>
            <td class="py-2.5 px-2.5">
              ${r.passed
                ? '<span class="px-2 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-semibold">PASS</span>'
                : '<span class="px-2 py-0.5 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30 font-semibold">FAIL</span>'}
            </td>
          </tr>
        `).join('');
      } finally {
        btn.disabled = false;
        btn.textContent = '▶ Run Full Evaluation Benchmark';
      }
    }

    window.addEventListener('DOMContentLoaded', () => {
      const h = (window.location.hash || '').replace('#', '');
      if (h === 'admin' || h === 'eval') {
        switchTab(h);
      } else if (h === 'demo' || h === 'query') {
        askPreset('What is the minimum attendance percentage required in every course and what is the medical relaxation?');
      }
    });
  </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def serve_ui() -> HTMLResponse:
    return HTMLResponse(content=UI_HTML)
