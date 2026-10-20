# Contributing to Campus RAG Assistant

Thank you for your interest in improving Campus RAG Assistant! We welcome community contributions, bug reports, and enhancements.

---

## Code of Conduct

Please maintain a supportive, inclusive, and professional environment for all contributors.

---

## Development Workflow

1. **Fork the repository** on GitHub.
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/<your-username>/day-05-campus-rag-assistant.git
   cd day-05-campus-rag-assistant
   ```
3. **Create a virtual environment & install dependencies**:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
4. **Create a feature branch**:
   ```bash
   git checkout -b feat/your-feature-name
   ```
5. **Make your changes** and add tests under `tests/`.
6. **Run the test suite** to ensure zero regressions:
   ```bash
   python -m pytest -v
   ```
7. **Commit using structured commit messages**:
   ```bash
   git commit -m "feat(rag): add semantic reranker optimization"
   ```
8. **Push to your fork** and submit a Pull Request.

---

## Areas for Contribution
- Additional document parsers (EPUB, RTF, LaTeX).
- Additional vector store connectors (Qdrant, Milvus, Weaviate).
- Multimodal PDF chart and diagram parsing.
