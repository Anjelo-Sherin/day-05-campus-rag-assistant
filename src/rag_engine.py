"""
Grounded RAG Answer Engine with Strict Hallucination Refusal & Page-Level Citations.

Workflow Step 5:
- Passes retrieved top-k chunks to the LLM with a strict "answer only from context" prompt.
- Uses Google Gemini (`gemini-3.8-flash` via the official `google-genai` Interactions API
  or REST v1beta/interactions endpoint) when `GEMINI_API_KEY` is provided.
- Includes a high-precision Extractive + Grounded Synthesis Engine when running offline/local,
  so the project works out-of-the-box with zero API key setup while also supporting live Gemini.
- Refuses out-of-scope questions whenever top retrieval confidence falls below threshold or
  no supporting sentences exist in the corpus.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.vector_store import CampusVectorStore, RetrievedChunk, tokenize


STRICT_SYSTEM_INSTRUCTION = """You are the Campus RAG Assistant, an authoritative university academic and administrative AI.
CRITICAL RULES:
1. Answer the user's question ONLY using facts explicitly stated in the provided CONTEXT CHUNKS.
2. Every factual claim MUST include an inline citation in the exact format [file_name, p. X].
3. If the answer is NOT present in the provided CONTEXT CHUNKS, do NOT guess or use outside knowledge. Instead, set `is_refusal` to true and politely state that the information is not found in the uploaded campus documents.
4. Keep answers clear, well-structured, and directly helpful to students and faculty."""


@dataclass
class CitationItem:
    """Page-level source citation returned alongside every answer."""

    citation_id: int
    file_name: str
    page_number: int
    section_title: str
    doc_type: str
    snippet: str
    rerank_score: float
    dense_score: float
    bm25_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "citation_id": self.citation_id,
            "file_name": self.file_name,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "doc_type": self.doc_type,
            "snippet": self.snippet,
            "rerank_score": round(float(self.rerank_score), 4),
            "dense_score": round(float(self.dense_score), 4),
            "bm25_score": round(float(self.bm25_score), 4),
        }


@dataclass
class RAGAnswer:
    """Complete structured output from the Campus RAG pipeline."""

    query: str
    answer: str
    is_refusal: bool
    refusal_reason: Optional[str]
    confidence_score: float
    faithfulness_score: float
    latency_ms: float
    generator_backend: str
    citations: List[CitationItem] = field(default_factory=list)
    retrieved_chunks: List[RetrievedChunk] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "is_refusal": self.is_refusal,
            "refusal_reason": self.refusal_reason,
            "confidence_score": round(float(self.confidence_score), 4),
            "faithfulness_score": round(float(self.faithfulness_score), 4),
            "latency_ms": round(float(self.latency_ms), 2),
            "generator_backend": self.generator_backend,
            "citations": [c.to_dict() for c in self.citations],
            "retrieved_chunks": [rc.to_dict() for rc in self.retrieved_chunks],
        }


def compute_faithfulness(answer_text: str, citations: List[CitationItem], is_refusal: bool) -> float:
    """
    Measure answer faithfulness (0.0 to 1.0):
    What fraction of substantive content tokens in the generated answer are directly
    supported by the cited source chunks? Refusals on out-of-scope queries score 1.0
    because refusing to hallucinate is 100% faithful to context constraints.
    """
    if is_refusal:
        return 1.0
    if not citations or not answer_text.strip():
        return 0.0

    # Strip inline citation tags like [CS301_AI_Syllabus.pdf, p. 2] before checking content tokens
    clean_answer = re.sub(r"\[[^\]]*p\.\s*\d+[^\]]*\]", "", answer_text)
    answer_tokens = tokenize(clean_answer, remove_stopwords=True)
    if not answer_tokens:
        return 1.0

    context_corpus = " ".join(f"{c.section_title} {c.snippet}" for c in citations)
    context_tokens = set(tokenize(context_corpus, remove_stopwords=True))

    supported = sum(
        1
        for tok in answer_tokens
        if tok in context_tokens or any(tok in c_tok or c_tok in tok for c_tok in context_tokens if len(c_tok) >= 4)
    )
    raw_ratio = supported / len(answer_tokens)
    return float(min(1.0, round(raw_ratio, 4)))


class CampusRAGEngine:
    """
    End-to-end RAG Engine combining Hybrid Retrieval, Cross-Feature Reranking,
    Out-of-Scope Guardrails, and Grounded Answer Synthesis with Page Citations.
    """

    def __init__(
        self,
        vector_store: CampusVectorStore,
        refusal_threshold: float = 0.28,
        gemini_model: str = "gemini-3.8-flash",
    ) -> None:
        self.vector_store = vector_store
        self.refusal_threshold = refusal_threshold
        self.gemini_model = gemini_model

    def _call_gemini_interactions_api(
        self,
        query: str,
        retrieved: List[RetrievedChunk],
        api_key: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Call Gemini (`gemini-3.8-flash`) using the official `google-genai` SDK
        (`client.interactions.create`) or the REST v1beta/interactions endpoint
        with structured JSON output.
        """
        context_blocks = []
        for idx, item in enumerate(retrieved, start=1):
            c = item.chunk
            context_blocks.append(
                f"[Source {idx}] File: {c.file_name} | Page: {c.page_number} | Section: {c.section_title}\n"
                f"Content: {c.text}"
            )
        context_str = "\n\n".join(context_blocks)
        prompt = (
            f"CONTEXT CHUNKS:\n{context_str}\n\n"
            f"USER QUESTION: {query}\n\n"
            "Provide a grounded answer strictly from the context chunks above, including inline citations "
            "like [file_name, p. X], or refuse if the context does not contain the answer."
        )

        schema = {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "Grounded answer with inline [file_name, p. X] citations, or refusal message.",
                },
                "is_refusal": {
                    "type": "boolean",
                    "description": "True if the context chunks do not contain the answer to the user's question.",
                },
                "cited_files": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["answer", "is_refusal"],
        }

        # 1. Try google-genai SDK if installed
        try:
            from google import genai  # type: ignore

            client = genai.Client(api_key=api_key)
            interaction = client.interactions.create(
                model=self.gemini_model,
                system_instruction=STRICT_SYSTEM_INSTRUCTION,
                input=prompt,
                response_format={
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
                generation_config={"temperature": 0.1},
            )
            if interaction and interaction.output_text:
                return json.loads(interaction.output_text)
        except Exception:
            pass

        # 2. Try REST v1beta/interactions endpoint
        try:
            url = "https://generativelanguage.googleapis.com/v1beta/interactions"
            payload = {
                "model": self.gemini_model,
                "system_instruction": STRICT_SYSTEM_INSTRUCTION,
                "input": prompt,
                "response_format": {
                    "type": "text",
                    "mime_type": "application/json",
                    "schema": schema,
                },
                "generation_config": {"temperature": 0.1},
            }
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": api_key,
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=12) as resp:
                body = json.loads(resp.read().decode("utf-8"))
                output_text = body.get("output_text")
                if not output_text and "steps" in body:
                    for step in reversed(body["steps"]):
                        if step.get("type") == "model_output":
                            for part in step.get("content", []):
                                if part.get("type") == "text":
                                    output_text = part.get("text")
                                    break
                if output_text:
                    return json.loads(output_text)
        except Exception:
            pass

        return None

    def _synthesize_grounded_local_answer(
        self,
        query: str,
        retrieved: List[RetrievedChunk],
    ) -> Tuple[str, List[RetrievedChunk]]:
        """
        Deterministic Grounded Sentence Synthesizer:
        Selects the highest-scoring sentences across the top retrieved chunks that directly
        answer the user's question, formats them cleanly, and attaches exact
        `[file_name, p. X]` inline citations.
        """
        q_tokens = tokenize(query, remove_stopwords=True)
        q_numbers = set(re.findall(r"\b\d+(?:\.\d+)?%?\b", query.lower()))

        scored_sentences: List[Tuple[float, str, RetrievedChunk]] = []

        for rank_idx, rc in enumerate(retrieved):
            # Split chunk text into sentences or bullet lines
            raw_segments = re.split(r"(?<=[.!?])\s+|\n+", rc.chunk.text)
            rank_discount = 1.0 / (1.0 + 0.18 * rank_idx)

            for seg in raw_segments:
                s_clean = seg.strip(" -•*\t")
                if len(s_clean) < 18:
                    continue
                s_toks = set(tokenize(s_clean, remove_stopwords=True))
                if not s_toks:
                    continue

                overlap = sum(1 for qt in q_tokens if qt in s_toks or qt in s_clean.lower())
                if overlap == 0 and rank_idx > 0:
                    continue

                coverage = overlap / max(1, len(q_tokens))
                has_nums = 0.12 if re.search(r"\b\d+(?:\.\d+)?%?\b", s_clean) else 0.0
                q_num_match = (
                    0.18
                    if (q_numbers and any(num in s_clean.lower() for num in q_numbers))
                    else 0.0
                )

                sent_score = (0.50 * coverage + 0.35 * rc.rerank_score + has_nums + q_num_match) * rank_discount
                scored_sentences.append((sent_score, s_clean, rc))

        scored_sentences.sort(key=lambda x: x[0], reverse=True)

        selected: List[Tuple[str, RetrievedChunk]] = []
        seen_norm: set[str] = set()
        used_chunks: List[RetrievedChunk] = []
        used_chunk_ids: set[str] = set()

        for score, sent, rc in scored_sentences:
            norm = re.sub(r"\W+", "", sent.lower())[:60]
            if norm in seen_norm:
                continue
            seen_norm.add(norm)
            selected.append((sent, rc))
            if rc.chunk.chunk_id not in used_chunk_ids:
                used_chunks.append(rc)
                used_chunk_ids.add(rc.chunk.chunk_id)
            if len(selected) >= 3:
                break

        if not selected and retrieved:
            top_rc = retrieved[0]
            selected.append((top_rc.chunk.text[:300].strip(), top_rc))
            used_chunks = [top_rc]

        # Format the grounded response with inline page-level citations
        bullet_lines: List[str] = []
        for sent, rc in selected:
            if not sent.endswith((".", "!", "?")):
                sent += "."
            citation_tag = f"[{rc.chunk.file_name}, p. {rc.chunk.page_number}]"
            bullet_lines.append(f"{sent} {citation_tag}")

        if len(bullet_lines) == 1:
            final_answer = bullet_lines[0]
        else:
            primary = bullet_lines[0]
            supporting = "\n".join(f"- {line}" for line in bullet_lines[1:])
            final_answer = f"{primary}\n\n**Supporting Details from Campus Documents:**\n{supporting}"

        return final_answer, used_chunks

    def answer_question(
        self,
        query: str,
        top_k: int = 4,
        alpha: float = 0.55,
        use_reranker: bool = True,
        refusal_threshold: Optional[float] = None,
        file_filter: Optional[List[str]] = None,
        api_key: Optional[str] = None,
    ) -> RAGAnswer:
        """
        Full RAG Pipeline execution:
        1. Hybrid Retrieval (Dense + BM25) + Cross-Feature Reranking
        2. Out-of-Scope Hallucination Guardrail check
        3. LLM Grounded Generation (Gemini 3.8 Flash or Grounded Local Synthesizer)
        4. Faithfulness & Citation Verification
        """
        t0 = time.perf_counter()
        threshold = refusal_threshold if refusal_threshold is not None else self.refusal_threshold

        retrieved = self.vector_store.search(
            query=query,
            top_k=top_k,
            alpha=alpha,
            use_reranker=use_reranker,
            file_filter=file_filter,
        )

        if not retrieved:
            latency = (time.perf_counter() - t0) * 1000.0
            return RAGAnswer(
                query=query,
                answer=(
                    "I cannot answer this question because no documents are currently indexed "
                    "or matched your filter. Please upload campus PDFs, slides, or circulars first."
                ),
                is_refusal=True,
                refusal_reason="empty_index",
                confidence_score=0.0,
                faithfulness_score=1.0,
                latency_ms=latency,
                generator_backend="Out-of-Scope Guardrail",
                citations=[],
                retrieved_chunks=[],
            )

        top_chunk = retrieved[0]
        top_score = top_chunk.rerank_score
        q_tokens = tokenize(query, remove_stopwords=True)

        # Out-of-scope guardrail: check both top rerank score and lexical/semantic overlap
        total_matched_terms = sum(len(rc.matched_terms) for rc in retrieved[:2])
        is_out_of_scope = (
            top_score < threshold
            or (len(q_tokens) >= 2 and total_matched_terms == 0 and top_chunk.bm25_score < 0.08)
        )

        if is_out_of_scope:
            latency = (time.perf_counter() - t0) * 1000.0
            indexed_files = sorted({c.file_name for c in self.vector_store.chunks})
            files_list_str = ", ".join(f"`{f}`" for f in indexed_files[:6])
            refusal_msg = (
                "**Out-of-Scope Refusal (No Hallucination Policy):**\n"
                "I could not find reliable evidence to answer this question in the uploaded campus documents "
                f"(top retrieval confidence `{top_score:.2f}` is below the grounding threshold `{threshold:.2f}`).\n\n"
                f"**Currently Indexed Documents:** {files_list_str}."
            )
            return RAGAnswer(
                query=query,
                answer=refusal_msg,
                is_refusal=True,
                refusal_reason=f"Retrieval confidence ({top_score:.2f}) below refusal threshold ({threshold:.2f})",
                confidence_score=top_score,
                faithfulness_score=1.0,
                latency_ms=latency,
                generator_backend="Strict Grounding Guardrail",
                citations=[],
                retrieved_chunks=retrieved,
            )

        resolved_key = (api_key or os.environ.get("GEMINI_API_KEY", "")).strip()
        backend_used = "Grounded Extractive Synthesizer (CPU)"
        answer_text = ""
        used_retrieved = retrieved[:3]

        if resolved_key:
            gemini_resp = self._call_gemini_interactions_api(query, retrieved, resolved_key)
            if gemini_resp and isinstance(gemini_resp.get("answer"), str):
                backend_used = f"Google Gemini ({self.gemini_model})"
                if gemini_resp.get("is_refusal", False):
                    latency = (time.perf_counter() - t0) * 1000.0
                    return RAGAnswer(
                        query=query,
                        answer=gemini_resp["answer"],
                        is_refusal=True,
                        refusal_reason="LLM determined context does not contain the answer",
                        confidence_score=top_score,
                        faithfulness_score=1.0,
                        latency_ms=latency,
                        generator_backend=backend_used,
                        citations=[],
                        retrieved_chunks=retrieved,
                    )
                answer_text = gemini_resp["answer"]

        if not answer_text:
            answer_text, used_retrieved = self._synthesize_grounded_local_answer(query, retrieved)

        citations: List[CitationItem] = []
        for idx, rc in enumerate(used_retrieved, start=1):
            citations.append(
                CitationItem(
                    citation_id=idx,
                    file_name=rc.chunk.file_name,
                    page_number=rc.chunk.page_number,
                    section_title=rc.chunk.section_title,
                    doc_type=rc.chunk.doc_type,
                    snippet=rc.chunk.text,
                    rerank_score=rc.rerank_score,
                    dense_score=rc.dense_score,
                    bm25_score=rc.bm25_score,
                )
            )

        faithfulness = compute_faithfulness(answer_text, citations, is_refusal=False)
        latency = (time.perf_counter() - t0) * 1000.0

        return RAGAnswer(
            query=query,
            answer=answer_text,
            is_refusal=False,
            refusal_reason=None,
            confidence_score=top_score,
            faithfulness_score=faithfulness,
            latency_ms=latency,
            generator_backend=backend_used,
            citations=citations,
            retrieved_chunks=retrieved,
        )
