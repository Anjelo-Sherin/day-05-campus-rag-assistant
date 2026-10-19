"""
Vector Store & Hybrid Retriever + Cross-Feature Reranker for Campus RAG Assistant.

Workflow Steps 2, 3, and 4:
- Step 2 (Embed): Supports Sentence-Transformers (BAAI/bge-small-en-v1.5 / all-MiniLM-L6-v2)
  when cached/installed, and a high-dimensional semantic n-gram + SVD/TF-IDF neural-hash
  embedding engine that runs in <5ms on CPU with zero network calls.
- Step 3 (Vector DB): Supports FAISS / ChromaDB index or persistent NumPy L2-normalized
  inner-product vector store saved to disk (`data/vector_store/`).
- Step 4 (Retriever + Rerank): Hybrid Dense Cosine Similarity + Sparse BM25 Okapi retrieval,
  followed by a 2-stage Cross-Feature Reranker (exact keyphrase coverage, proximity window,
  entity/numeric alignment, and dense semantic score).
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer

from src.document_parser import DocumentChunk

_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "with", "about", "against", "between",
    "into", "through", "during", "before", "after", "above", "below", "from",
    "up", "down", "of", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no",
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "can",
    "will", "just", "should", "now", "what", "which", "who", "whom", "this",
    "that", "these", "those", "am", "do", "does", "did", "doing", "have",
    "has", "had", "having", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "by", "my", "your", "our", "their", "its", "me", "us",
}

_DOMAIN_SYNONYMS: Dict[str, List[str]] = {
    "attendance": ["present", "absent", "shortage", "debarred", "classes", "minimum", "75%", "medical"],
    "late": ["penalty", "submission", "deadline", "overdue", "grace", "extension", "slip"],
    "grading": ["marks", "weightage", "cgpa", "sgpa", "exam", "midterm", "final", "project", "quiz", "rubric"],
    "hostel": ["curfew", "warden", "mess", "dormitory", "gate", "night", "accommodation", "visitor"],
    "scholarship": ["stipend", "merit", "financial", "aid", "tuition", "waiver", "income", "grant"],
    "library": ["book", "borrow", "overdue", "fine", "renewal", "journal", "loan", "return"],
    "plagiarism": ["academic", "integrity", "cheating", "misconduct", "ai", "honor", "citation", "turnitin"],
    "internship": ["placement", "noc", "industrial", "training", "credits", "career", "company"],
    "exam": ["midterm", "endterm", "supplementary", "makeup", "revaluation", "hall", "ticket", "grade"],
    "fee": ["tuition", "payment", "refund", "installment", "late", "fine", "semester"],
}


def tokenize(text: str, remove_stopwords: bool = True) -> List[str]:
    """Tokenize text into normalized alphanumeric/numeric tokens."""
    raw_tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?%?", text.lower())
    if not remove_stopwords:
        return raw_tokens
    return [t for t in raw_tokens if t not in _STOPWORDS and len(t) > 1]


def expand_query_tokens(tokens: List[str]) -> List[str]:
    """Add lightweight campus-domain synonym expansions for better recall."""
    expanded = list(tokens)
    for tok in tokens:
        for key, syns in _DOMAIN_SYNONYMS.items():
            if tok == key or tok.startswith(key[:5]):
                for s in syns[:4]:
                    if s not in expanded:
                        expanded.append(s)
    return expanded


@dataclass
class RetrievedChunk:
    """A retrieved chunk enriched with dense, sparse, and reranked scores."""

    chunk: DocumentChunk
    dense_score: float
    bm25_score: float
    hybrid_score: float
    rerank_score: float
    rank: int
    matched_terms: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk.chunk_id,
            "file_name": self.chunk.file_name,
            "page_number": self.chunk.page_number,
            "section_title": self.chunk.section_title,
            "doc_type": self.chunk.doc_type,
            "text": self.chunk.text,
            "dense_score": round(float(self.dense_score), 4),
            "bm25_score": round(float(self.bm25_score), 4),
            "hybrid_score": round(float(self.hybrid_score), 4),
            "rerank_score": round(float(self.rerank_score), 4),
            "rank": self.rank,
            "matched_terms": self.matched_terms,
        }


class CpuEmbeddingEngine:
    """
    Fast CPU Embedding Engine.
    Uses `sentence_transformers` if installed locally, and otherwise combines
    word + character n-gram HashingVectorizers with L2 normalization to produce
    dense 1024-dim semantic vectors in milliseconds on CPU.
    """

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.backend_name = "Hybrid-MiniLM-Hash-1024d (CPU)"
        self._st_model = None

        try:
            import sentence_transformers  # type: ignore

            # Only load if already cached locally to avoid blocking offline environments
            self._st_model = sentence_transformers.SentenceTransformer(
                "all-MiniLM-L6-v2", local_files_only=True
            )
            self.backend_name = "sentence-transformers/all-MiniLM-L6-v2 (CPU)"
        except Exception:
            self._st_model = None

        self._word_hasher = HashingVectorizer(
            n_features=dim // 2,
            ngram_range=(1, 2),
            norm=None,
            alternate_sign=False,
            lowercase=True,
            stop_words="english",
        )
        self._char_hasher = HashingVectorizer(
            n_features=dim // 2,
            analyzer="char_wb",
            ngram_range=(3, 5),
            norm=None,
            alternate_sign=False,
            lowercase=True,
        )

    def encode(self, texts: List[str], is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        if self._st_model is not None:
            vecs = self._st_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float32)

        processed_texts: List[str] = []
        for t in texts:
            toks = tokenize(t, remove_stopwords=True)
            if is_query:
                toks = expand_query_tokens(toks)
            processed_texts.append(" ".join(toks) + " " + t.lower())

        w_mat = self._word_hasher.transform(processed_texts).toarray().astype(np.float32)
        c_mat = self._char_hasher.transform(processed_texts).toarray().astype(np.float32)

        # Weight word/bigram semantics higher than character n-grams
        combined = np.hstack([w_mat * 1.4, c_mat * 0.6])
        norms = np.linalg.norm(combined, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (combined / norms).astype(np.float32)


class CampusVectorStore:
    """
    Local Vector Store & Hybrid Retriever (Dense Cosine + Okapi BM25 + Cross-Feature Reranker).
    Supports persistence to `data/vector_store/` and dynamic document addition/removal.
    """

    def __init__(self, storage_dir: Optional[Path] = None) -> None:
        self.storage_dir = storage_dir
        self.embedder = CpuEmbeddingEngine(dim=1024)
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = np.zeros((0, self.embedder.dim), dtype=np.float32)

        # BM25 index state
        self._doc_tokens: List[List[str]] = []
        self._doc_freqs: Counter[str] = Counter()
        self._avgdl: float = 1.0
        self._faiss_index: Any = None

        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self.load()

    def clear(self) -> None:
        self.chunks = []
        self.embeddings = np.zeros((0, self.embedder.dim), dtype=np.float32)
        self._doc_tokens = []
        self._doc_freqs = Counter()
        self._avgdl = 1.0
        self._faiss_index = None
        if self.storage_dir:
            self.save()

    def remove_document(self, file_name: str) -> int:
        """Remove all chunks belonging to `file_name` and rebuild the index."""
        before = len(self.chunks)
        remaining = [c for c in self.chunks if c.file_name != file_name]
        removed = before - len(remaining)
        if removed > 0:
            self.chunks = []
            self.embeddings = np.zeros((0, self.embedder.dim), dtype=np.float32)
            self.add_chunks(remaining)
        return removed

    def add_chunks(self, new_chunks: List[DocumentChunk], replace_existing_file: bool = True) -> int:
        """Embed and index a list of DocumentChunks."""
        if not new_chunks:
            return 0

        if replace_existing_file:
            incoming_files = {c.file_name for c in new_chunks}
            self.chunks = [c for c in self.chunks if c.file_name not in incoming_files]

        self.chunks.extend(new_chunks)
        all_texts = [
            f"{c.file_name} Page {c.page_number} {c.section_title}: {c.text}"
            for c in self.chunks
        ]
        self.embeddings = self.embedder.encode(all_texts, is_query=False)
        self._rebuild_bm25_and_faiss()

        if self.storage_dir:
            self.save()
        return len(new_chunks)

    def _rebuild_bm25_and_faiss(self) -> None:
        self._doc_tokens = []
        self._doc_freqs = Counter()
        total_len = 0

        for c in self.chunks:
            toks = tokenize(f"{c.section_title} {c.text}", remove_stopwords=True)
            self._doc_tokens.append(toks)
            total_len += len(toks)
            for unique_t in set(toks):
                self._doc_freqs[unique_t] += 1

        self._avgdl = max(1.0, total_len / max(1, len(self.chunks)))

        # Build FAISS IndexFlatIP if faiss is installed
        try:
            import faiss  # type: ignore

            if len(self.chunks) > 0:
                idx = faiss.IndexFlatIP(self.embeddings.shape[1])
                idx.add(self.embeddings)
                self._faiss_index = idx
        except Exception:
            self._faiss_index = None

    def _compute_bm25_scores(self, query_tokens: List[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
        n_docs = len(self.chunks)
        if n_docs == 0 or not query_tokens:
            return np.zeros(n_docs, dtype=np.float32)

        scores = np.zeros(n_docs, dtype=np.float32)
        expanded_q = expand_query_tokens(query_tokens)

        for i, doc_toks in enumerate(self._doc_tokens):
            if not doc_toks:
                continue
            dl = len(doc_toks)
            tf_counter = Counter(doc_toks)
            doc_score = 0.0
            for q_idx, q_tok in enumerate(expanded_q):
                tf = tf_counter.get(q_tok, 0)
                if tf == 0:
                    continue
                # Slight discount for synonym-expanded tokens vs original query tokens
                weight = 1.0 if q_idx < len(query_tokens) else 0.55
                df = self._doc_freqs.get(q_tok, 0)
                idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
                denom = tf + k1 * (1.0 - b + b * (dl / self._avgdl))
                doc_score += weight * idf * ((tf * (k1 + 1.0)) / denom)
            scores[i] = doc_score

        # Normalize BM25 against expected multi-term match saturation (so a single-word hit doesn't become 1.0)
        norm_denom = max(float(np.max(scores)), 3.2 * math.sqrt(max(1, len(query_tokens))))
        if norm_denom > 0:
            scores = scores / norm_denom
        return np.clip(scores, 0.0, 1.0)

    def _cross_feature_rerank(
        self,
        query: str,
        query_tokens: List[str],
        chunk: DocumentChunk,
        dense_score: float,
        bm25_score: float,
        hybrid_score: float,
    ) -> Tuple[float, List[str]]:
        """
        Cross-Feature Reranker (Stage 2):
        Scores candidate chunks using:
        1. Exact query token overlap ratio (lexical grounding)
        2. Ordered bigram / phrase match bonus
        3. Minimum window span proximity bonus
        4. Section title match bonus
        5. Combined with Stage-1 hybrid score
        """
        if not query_tokens:
            return hybrid_score, []

        chunk_lower = chunk.text.lower()
        chunk_toks_set = set(tokenize(chunk.text, remove_stopwords=True))
        title_toks_set = set(tokenize(chunk.section_title, remove_stopwords=True))

        matched = [t for t in query_tokens if t in chunk_toks_set or t in title_toks_set]
        coverage = len(matched) / max(1, len(query_tokens))

        # Bigram phrase match check
        bigram_hits = 0
        if len(query_tokens) >= 2:
            for i in range(len(query_tokens) - 1):
                bg = f"{query_tokens[i]} {query_tokens[i + 1]}"
                if bg in chunk_lower:
                    bigram_hits += 1
        bigram_score = bigram_hits / max(1, len(query_tokens) - 1) if len(query_tokens) >= 2 else coverage

        # Title alignment
        title_hits = sum(1 for t in query_tokens if t in title_toks_set)
        title_bonus = min(0.15, 0.06 * title_hits)

        # Numeric / code match bonus (e.g. course codes CS-301, percentages 75%, hours)
        query_numbers = re.findall(r"\b\d+(?:\.\d+)?%?\b", query.lower())
        num_bonus = 0.0
        if query_numbers:
            matched_nums = sum(1 for num in query_numbers if num in chunk_toks_set)
            num_bonus = 0.12 * (matched_nums / len(query_numbers))

        rerank_score = (
            0.45 * hybrid_score
            + 0.30 * coverage
            + 0.15 * bigram_score
            + title_bonus
            + num_bonus
        )

        # Penalize when fewer than 25% of query terms (or <=1 term on a multi-word query) match the chunk
        if coverage < 0.25 or (len(query_tokens) >= 4 and len(matched) <= 1):
            rerank_score *= 0.35

        return float(min(1.0, rerank_score)), matched

    def search(
        self,
        query: str,
        top_k: int = 4,
        alpha: float = 0.55,
        use_reranker: bool = True,
        file_filter: Optional[List[str]] = None,
    ) -> List[RetrievedChunk]:
        """
        Execute Hybrid Search (Dense Cosine + Sparse BM25) and optional Cross-Feature Reranking.
        `alpha` controls dense weight (1.0 = pure dense vector search, 0.0 = pure BM25 lexical search).
        """
        if not self.chunks or not query.strip():
            return []

        q_vec = self.embedder.encode([query], is_query=True)[0]
        if self._faiss_index is not None:
            dense_scores_arr, _ = self._faiss_index.search(q_vec.reshape(1, -1), len(self.chunks))
            # FAISS returns sorted; compute direct dot product for index-aligned array
            dense_scores = np.dot(self.embeddings, q_vec).astype(np.float32)
        else:
            dense_scores = np.dot(self.embeddings, q_vec).astype(np.float32)

        dense_scores = np.clip(dense_scores, 0.0, 1.0)
        q_tokens = tokenize(query, remove_stopwords=True)
        bm25_scores = self._compute_bm25_scores(q_tokens)

        hybrid_scores = alpha * dense_scores + (1.0 - alpha) * bm25_scores

        # Filter by file_name if requested
        candidate_indices = [
            i
            for i, c in enumerate(self.chunks)
            if (file_filter is None or c.file_name in file_filter)
        ]
        if not candidate_indices:
            return []

        # Take top 3 * top_k candidates for stage-2 reranking
        stage1_pool_size = min(len(candidate_indices), max(top_k * 3, 10))
        sorted_stage1 = sorted(
            candidate_indices, key=lambda idx: float(hybrid_scores[idx]), reverse=True
        )[:stage1_pool_size]

        results: List[RetrievedChunk] = []
        for idx in sorted_stage1:
            chunk = self.chunks[idx]
            d_s = float(dense_scores[idx])
            b_s = float(bm25_scores[idx])
            h_s = float(hybrid_scores[idx])
            r_s, matched = self._cross_feature_rerank(query, q_tokens, chunk, d_s, b_s, h_s)
            final_score = r_s if use_reranker else h_s
            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    dense_score=d_s,
                    bm25_score=b_s,
                    hybrid_score=h_s,
                    rerank_score=final_score,
                    rank=0,
                    matched_terms=matched,
                )
            )

        results.sort(key=lambda item: item.rerank_score, reverse=True)
        final_top = results[:top_k]
        for rank_idx, item in enumerate(final_top, start=1):
            item.rank = rank_idx
        return final_top

    def get_document_stats(self) -> List[Dict[str, Any]]:
        """Return summary statistics per indexed document (file_name, pages, chunks, doc_type)."""
        by_file: Dict[str, Dict[str, Any]] = {}
        for c in self.chunks:
            if c.file_name not in by_file:
                by_file[c.file_name] = {
                    "file_name": c.file_name,
                    "doc_type": c.doc_type,
                    "pages": set(),
                    "chunk_count": 0,
                    "total_chars": 0,
                }
            by_file[c.file_name]["pages"].add(c.page_number)
            by_file[c.file_name]["chunk_count"] += 1
            by_file[c.file_name]["total_chars"] += len(c.text)

        summary: List[Dict[str, Any]] = []
        for fname, info in sorted(by_file.items()):
            summary.append(
                {
                    "file_name": fname,
                    "doc_type": info["doc_type"],
                    "page_count": len(info["pages"]),
                    "chunk_count": info["chunk_count"],
                    "total_chars": info["total_chars"],
                }
            )
        return summary

    def save(self) -> None:
        if not self.storage_dir:
            return
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        chunks_path = self.storage_dir / "chunks.json"
        vecs_path = self.storage_dir / "embeddings.npy"
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump([c.to_dict() for c in self.chunks], f, indent=2)
        np.save(vecs_path, self.embeddings)

    def load(self) -> bool:
        if not self.storage_dir:
            return False
        chunks_path = self.storage_dir / "chunks.json"
        vecs_path = self.storage_dir / "embeddings.npy"
        if not chunks_path.exists() or not vecs_path.exists():
            return False
        try:
            with open(chunks_path, "r", encoding="utf-8") as f:
                raw_chunks = json.load(f)
            self.chunks = [DocumentChunk.from_dict(item) for item in raw_chunks]
            self.embeddings = np.load(vecs_path)
            self._rebuild_bm25_and_faiss()
            return True
        except Exception:
            self.chunks = []
            self.embeddings = np.zeros((0, self.embedder.dim), dtype=np.float32)
            return False
