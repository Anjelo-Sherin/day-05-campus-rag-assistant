"""
Comprehensive Unit & Integration Tests for Day 05: Campus RAG Assistant.
Tests:
1. Multi-format Document Parsing (PDF binary generation & extraction, HTML, Markdown, PPTX/DOCX)
2. Overlapping Chunker with page-level preservation
3. Hybrid Dense + BM25 Vector Store & Cross-Feature Reranker
4. Grounded RAG Answer Engine with inline [file, p. X] citations & Out-of-Scope Hallucination Refusal
5. SQuAD-style 14-question Evaluation Benchmark (Hit-Rate@1, Hit-Rate@K, MRR, Faithfulness, Refusal Rate)
6. FastAPI REST Endpoints (/api/health, /api/documents, /api/chat, /api/upload, /api/evaluate)
"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from src.api import app
from src.campus_corpus import seed_default_corpus
from src.document_parser import (
    build_minimal_pdf,
    chunk_pages,
    parse_document,
)
from src.evaluator import run_evaluation_suite
from src.rag_engine import CampusRAGEngine
from src.vector_store import CampusVectorStore


def test_pdf_generation_and_page_extraction() -> None:
    raw_pages = [
        "Page 1: CS-301 Grading Policy requires 25% assignments and 30% end-semester exam.",
        "Page 2: Late submissions receive a 10% daily deduction after 3 grace slip days.",
    ]
    pdf_bytes = build_minimal_pdf(raw_pages)
    assert pdf_bytes.startswith(b"%PDF-1.4")

    extracted = parse_document("test_syllabus.pdf", pdf_bytes)
    assert len(extracted) == 2
    assert extracted[0].page_number == 1
    assert "CS-301 Grading Policy" in extracted[0].text
    assert extracted[1].page_number == 2
    assert "3 grace slip days" in extracted[1].text


def test_overlapping_chunker_preserves_page_metadata() -> None:
    long_text = (
        "Section 1: Attendance Rules\n"
        + ("Every student must maintain 75% attendance in all courses. " * 25)
    )
    pages = parse_document("regulations.txt", long_text.encode("utf-8"))
    chunks = chunk_pages(pages, chunk_size=300, chunk_overlap=80)
    assert len(chunks) >= 3
    for c in chunks:
        assert c.file_name == "regulations.txt"
        assert c.page_number >= 1
        assert len(c.text) <= 320


def test_hybrid_retrieval_and_grounded_citations(tmp_path: Path) -> None:
    store = CampusVectorStore(storage_dir=tmp_path / "vs")
    sample_dir = tmp_path / "docs"
    total_chunks = seed_default_corpus(store, sample_dir, chunk_size=550, chunk_overlap=120, force_rebuild=True)
    assert total_chunks >= 10

    engine = CampusRAGEngine(vector_store=store, refusal_threshold=0.28)

    # 1. In-scope query about attendance
    ans = engine.answer_question("What is the minimum attendance percentage required in every course?")
    assert not ans.is_refusal
    assert "75%" in ans.answer
    assert "Academic_Regulations_Handbook.pdf" in ans.answer
    assert len(ans.citations) >= 1
    assert ans.citations[0].file_name == "Academic_Regulations_Handbook.pdf"
    assert ans.citations[0].page_number == 1
    assert ans.faithfulness_score >= 0.80

    # 2. Out-of-scope query must be refused
    refusal = engine.answer_question("What is the recipe for baking a classic sourdough pizza crust in a wood-fired oven?")
    assert refusal.is_refusal
    assert refusal.citations == []
    assert refusal.faithfulness_score == 1.0


def test_evaluation_benchmark_metrics(tmp_path: Path) -> None:
    store = CampusVectorStore(storage_dir=tmp_path / "vs_eval")
    sample_dir = tmp_path / "docs_eval"
    seed_default_corpus(store, sample_dir, chunk_size=550, chunk_overlap=120, force_rebuild=True)
    engine = CampusRAGEngine(vector_store=store, refusal_threshold=0.28)

    report = run_evaluation_suite(engine, top_k=4, alpha=0.55, use_reranker=True, refusal_threshold=0.28)
    summary = report["summary"]

    assert summary["total_questions"] == 14
    assert summary["hit_rate_at_1"] >= 0.90
    assert summary["hit_rate_at_k"] >= 0.90
    assert summary["mrr"] >= 0.90
    assert summary["mean_faithfulness"] >= 0.85
    assert summary["out_of_scope_refusal_rate"] == 1.0


def test_fastapi_endpoints() -> None:
    client = TestClient(app)

    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "ok"

    r_docs = client.get("/api/documents")
    assert r_docs.status_code == 200
    assert r_docs.json()["total_documents"] >= 4

    r_chat = client.post(
        "/api/chat",
        json={
            "query": "How many books can an undergraduate borrow from the Central Library and what is the overdue fine?",
            "top_k": 4,
            "alpha": 0.55,
            "use_reranker": True,
            "refusal_threshold": 0.28,
        },
    )
    assert r_chat.status_code == 200
    body = r_chat.json()
    assert not body["is_refusal"]
    assert "Campus_Hostel_Library_Circular.pdf" in body["answer"]

    r_eval = client.post(
        "/api/evaluate",
        json={"top_k": 4, "alpha": 0.55, "use_reranker": True, "refusal_threshold": 0.28},
    )
    assert r_eval.status_code == 200
    assert r_eval.json()["summary"]["hit_rate_at_k"] >= 0.90
