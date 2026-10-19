"""
RAG Evaluation Harness (Workflow Step 6: Eval Page).

Measures retrieval and generation quality across the SQuAD-style Campus Ground-Truth Benchmark:
- Hit-Rate@1 (Top-1 Exact Document + Page Match)
- Hit-Rate@K (Top-K Exact Document + Page Match)
- Mean Reciprocal Rank (MRR)
- Answer Faithfulness Score (lexical & factual grounding against retrieved citations)
- Keyword Completeness Score (fraction of gold-standard facts present in generated answer)
- Out-of-Scope Refusal Accuracy (hallucination prevention rate on unanswerable queries)
- Chunk Size & Hybrid Search Alpha Ablation Experiment
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional

from src.campus_corpus import EVALUATION_BENCHMARK_SET
from src.rag_engine import CampusRAGEngine


def run_evaluation_suite(
    rag_engine: CampusRAGEngine,
    top_k: int = 4,
    alpha: float = 0.55,
    use_reranker: bool = True,
    refusal_threshold: float = 0.28,
    benchmark_items: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Execute the complete RAG Evaluation Harness over the 14-question benchmark
    and return both aggregate metrics and per-question diagnostic breakdowns.
    """
    items = benchmark_items or EVALUATION_BENCHMARK_SET
    t_start = time.perf_counter()

    in_scope_total = 0
    hit_at_1_count = 0
    hit_at_k_count = 0
    file_hit_at_k_count = 0
    mrr_sum = 0.0
    faithfulness_sum = 0.0
    keyword_recall_sum = 0.0

    out_of_scope_total = 0
    correct_refusal_count = 0

    per_question_results: List[Dict[str, Any]] = []

    for item in items:
        qid = item["id"]
        question = item["question"]
        is_answerable = item["is_answerable"]
        expected_file = item.get("expected_file")
        expected_page = item.get("expected_page")
        expected_keywords = [k.lower() for k in item.get("expected_keywords", [])]

        rag_resp = rag_engine.answer_question(
            query=question,
            top_k=top_k,
            alpha=alpha,
            use_reranker=use_reranker,
            refusal_threshold=refusal_threshold,
        )

        if is_answerable:
            in_scope_total += 1
            # Check exact (file_name + page_number) rank in retrieved_chunks
            exact_rank: Optional[int] = None
            file_matched = False
            for r_idx, rc in enumerate(rag_resp.retrieved_chunks, start=1):
                if rc.chunk.file_name == expected_file:
                    file_matched = True
                    if rc.chunk.page_number == expected_page and exact_rank is None:
                        exact_rank = r_idx

            hit_1 = exact_rank == 1
            hit_k = exact_rank is not None and exact_rank <= top_k
            rr = (1.0 / exact_rank) if exact_rank is not None else 0.0

            if hit_1:
                hit_at_1_count += 1
            if hit_k:
                hit_at_k_count += 1
            if file_matched:
                file_hit_at_k_count += 1
            mrr_sum += rr
            faithfulness_sum += rag_resp.faithfulness_score

            ans_lower = rag_resp.answer.lower()
            if expected_keywords:
                kw_hits = sum(1 for kw in expected_keywords if kw in ans_lower)
                kw_recall = kw_hits / len(expected_keywords)
            else:
                kw_recall = 1.0
            keyword_recall_sum += kw_recall

            passed = hit_k and (not rag_resp.is_refusal) and rag_resp.faithfulness_score >= 0.75

            per_question_results.append(
                {
                    "id": qid,
                    "category": item.get("category", "General"),
                    "question": question,
                    "is_answerable": True,
                    "expected_citation": f"{expected_file} (p. {expected_page})",
                    "top_retrieved_citation": (
                        f"{rag_resp.retrieved_chunks[0].chunk.file_name} (p. {rag_resp.retrieved_chunks[0].chunk.page_number})"
                        if rag_resp.retrieved_chunks
                        else "None"
                    ),
                    "exact_page_rank": exact_rank,
                    "hit_at_1": hit_1,
                    "hit_at_k": hit_k,
                    "reciprocal_rank": round(rr, 4),
                    "faithfulness": round(rag_resp.faithfulness_score, 4),
                    "keyword_coverage": round(kw_recall, 4),
                    "confidence": round(rag_resp.confidence_score, 4),
                    "is_refusal": rag_resp.is_refusal,
                    "passed": passed,
                    "answer_preview": rag_resp.answer[:220] + ("..." if len(rag_resp.answer) > 220 else ""),
                    "latency_ms": round(rag_resp.latency_ms, 2),
                }
            )
        else:
            out_of_scope_total += 1
            correctly_refused = rag_resp.is_refusal
            if correctly_refused:
                correct_refusal_count += 1

            per_question_results.append(
                {
                    "id": qid,
                    "category": item.get("category", "Out-of-Scope"),
                    "question": question,
                    "is_answerable": False,
                    "expected_citation": "REFUSAL (Out of Scope)",
                    "top_retrieved_citation": (
                        f"{rag_resp.retrieved_chunks[0].chunk.file_name} (p. {rag_resp.retrieved_chunks[0].chunk.page_number})"
                        if rag_resp.retrieved_chunks
                        else "None"
                    ),
                    "exact_page_rank": None,
                    "hit_at_1": correctly_refused,
                    "hit_at_k": correctly_refused,
                    "reciprocal_rank": 1.0 if correctly_refused else 0.0,
                    "faithfulness": round(rag_resp.faithfulness_score, 4),
                    "keyword_coverage": 1.0 if correctly_refused else 0.0,
                    "confidence": round(rag_resp.confidence_score, 4),
                    "is_refusal": rag_resp.is_refusal,
                    "passed": correctly_refused,
                    "answer_preview": rag_resp.answer[:220] + ("..." if len(rag_resp.answer) > 220 else ""),
                    "latency_ms": round(rag_resp.latency_ms, 2),
                }
            )

    total_time_ms = (time.perf_counter() - t_start) * 1000.0
    in_denom = max(1, in_scope_total)
    oos_denom = max(1, out_of_scope_total)

    summary = {
        "total_questions": len(items),
        "in_scope_questions": in_scope_total,
        "out_of_scope_questions": out_of_scope_total,
        "top_k": top_k,
        "hybrid_alpha": alpha,
        "use_reranker": use_reranker,
        "refusal_threshold": refusal_threshold,
        "hit_rate_at_1": round(hit_at_1_count / in_denom, 4),
        "hit_rate_at_k": round(hit_at_k_count / in_denom, 4),
        "document_hit_rate_at_k": round(file_hit_at_k_count / in_denom, 4),
        "mrr": round(mrr_sum / in_denom, 4),
        "mean_faithfulness": round(faithfulness_sum / in_denom, 4),
        "mean_keyword_coverage": round(keyword_recall_sum / in_denom, 4),
        "out_of_scope_refusal_rate": round(correct_refusal_count / oos_denom, 4),
        "overall_pass_rate": round(
            sum(1 for r in per_question_results if r["passed"]) / max(1, len(per_question_results)),
            4,
        ),
        "total_eval_time_ms": round(total_time_ms, 2),
        "avg_query_latency_ms": round(total_time_ms / max(1, len(items)), 2),
    }

    return {
        "summary": summary,
        "results": per_question_results,
    }
