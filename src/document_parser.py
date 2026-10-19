"""
Document Parser & Overlapping Chunker for Campus RAG Assistant.
Supports:
- PDF files (pure-Python stream/text extraction + pypdf/PyMuPDF fallback with page numbers)
- PPTX slide decks (extracts slide-by-slide text as page numbers via XML zip parsing)
- DOCX documents (extracts paragraphs grouped into logical pages)
- HTML / Web pages & URLs (strips scripts/styles, extracts clean structured sections)
- Markdown & Plain Text (.md, .txt) with automatic page/section boundary detection
"""

from __future__ import annotations

import io
import re
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
import zlib
from dataclasses import asdict, dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DocumentPage:
    """Represents a single extracted page or slide from a source document."""

    file_name: str
    page_number: int
    text: str
    doc_type: str = "document"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentChunk:
    """Represents a single overlapping text chunk indexed in the Vector Store."""

    chunk_id: str
    file_name: str
    page_number: int
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    doc_type: str = "document"
    section_title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        return cls(**data)


class _CleanHTMLParser(HTMLParser):
    """Lightweight HTML-to-text parser that ignores script/style tags."""

    def __init__(self) -> None:
        super().__init__()
        self._ignore_tags = {"script", "style", "noscript", "svg", "head"}
        self._ignore_depth = 0
        self._Block_tags = {"p", "div", "section", "article", "li", "h1", "h2", "h3", "h4", "tr", "br"}
        self.parts: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        t = tag.lower()
        if t in self._ignore_tags:
            self._ignore_depth += 1
        elif t in self._Block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self._ignore_tags and self._ignore_depth > 0:
            self._ignore_depth -= 1
        elif t in self._Block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignore_depth == 0:
            cleaned = data.strip()
            if cleaned:
                self.parts.append(cleaned + " ")

    def get_text(self) -> str:
        raw = "".join(self.parts)
        raw = re.sub(r"[ \t]+", " ", raw)
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        return raw.strip()


def html_to_text(html_content: str) -> str:
    parser = _CleanHTMLParser()
    parser.feed(html_content)
    return parser.get_text()


def _decode_pdf_literal_string(s: str) -> str:
    """Unescape standard PDF literal strings (...) into readable text."""
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\" and i + 1 < n:
            nxt = s[i + 1]
            if nxt in ("n", "r", "t", "b", "f"):
                mapping = {"n": "\n", "r": "\r", "t": "\t", "b": "\b", "f": "\f"}
                out.append(mapping[nxt])
                i += 2
                continue
            elif nxt in ("(", ")", "\\"):
                out.append(nxt)
                i += 2
                continue
            elif nxt.isdigit():
                octal = s[i + 1 : i + 4]
                m = re.match(r"^[0-7]{1,3}", octal)
                if m:
                    oct_str = m.group(0)
                    out.append(chr(int(oct_str, 8)))
                    i += 1 + len(oct_str)
                    continue
        out.append(ch)
        i += 1
    return "".join(out)


def _extract_text_from_pdf_stream(stream_bytes: bytes) -> str:
    """Extract text operators (Tj, TJ, ') from a decompressed PDF content stream."""
    try:
        text_data = stream_bytes.decode("latin-1", errors="ignore")
    except Exception:
        return ""

    extracted_lines: List[str] = []

    # Match TJ arrays: [(text) -20 (more)] TJ
    for match in re.finditer(r"\[(.*?)\]\s*TJ", text_data, flags=re.DOTALL):
        arr_body = match.group(1)
        literals = re.findall(r"\(((?:\\.|[^\\()])*)\)", arr_body)
        if literals:
            line = "".join(_decode_pdf_literal_string(lit) for lit in literals)
            extracted_lines.append(line)

    # Match single literal strings: (text) Tj or (text) '
    for match in re.finditer(r"\(((?:\\.|[^\\()])*)\)\s*(?:Tj|')", text_data):
        lit = _decode_pdf_literal_string(match.group(1))
        if lit.strip():
            extracted_lines.append(lit)

    joined = "\n".join(extracted_lines)
    joined = re.sub(r"[ \t]+", " ", joined)
    return joined.strip()


def parse_pdf_bytes(raw_bytes: bytes, file_name: str) -> List[DocumentPage]:
    """
    Parse a PDF file into a list of DocumentPage objects with 1-indexed page numbers.
    Tries pypdf / fitz first if installed, then falls back to native PDF stream extraction.
    """
    # 1. Try pypdf if available
    try:
        import pypdf  # type: ignore

        reader = pypdf.PdfReader(io.BytesIO(raw_bytes))
        pages: List[DocumentPage] = []
        for idx, page in enumerate(reader.pages):
            txt = (page.extract_text() or "").strip()
            if txt:
                pages.append(
                    DocumentPage(
                        file_name=file_name,
                        page_number=idx + 1,
                        text=txt,
                        doc_type="pdf",
                    )
                )
        if pages:
            return pages
    except Exception:
        pass

    # 2. Pure-Python PDF stream & page object extractor
    pages = []
    # Check for explicit Page objects or stream blocks
    stream_pattern = re.compile(rb"stream[\r\n]+(.*?)[\r\n]+endstream", re.DOTALL)
    page_streams: List[str] = []

    for match in stream_pattern.finditer(raw_bytes):
        raw_stream = match.group(1)
        decompressed = raw_stream
        for wbits in (zlib.MAX_WBITS, -zlib.MAX_WBITS):
            try:
                decompressed = zlib.decompress(raw_stream, wbits)
                break
            except Exception:
                continue
        txt = _extract_text_from_pdf_stream(decompressed)
        if txt and len(txt.strip()) > 10:
            page_streams.append(txt)

    if page_streams:
        for idx, page_txt in enumerate(page_streams):
            pages.append(
                DocumentPage(
                    file_name=file_name,
                    page_number=idx + 1,
                    text=page_txt,
                    doc_type="pdf",
                )
            )
        return pages

    # 3. Fallback: plain text / literal scan if un-streamed PDF
    fallback_txt = _extract_text_from_pdf_stream(raw_bytes)
    if not fallback_txt:
        try:
            fallback_txt = raw_bytes.decode("utf-8", errors="ignore").strip()
        except Exception:
            fallback_txt = ""

    if fallback_txt:
        return _split_text_into_virtual_pages(fallback_txt, file_name, doc_type="pdf")
    return []


def parse_pptx_bytes(raw_bytes: bytes, file_name: str) -> List[DocumentPage]:
    """
    Extract slide-by-slide text from a .pptx file using Python's built-in zipfile + XML parser.
    Each slide becomes a DocumentPage with page_number = slide number.
    """
    pages: List[DocumentPage] = []
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            slide_files = [
                f for f in zf.namelist() if re.match(r"^ppt/slides/slide\d+\.xml$", f)
            ]
            slide_files.sort(
                key=lambda name: int(re.search(r"slide(\d+)\.xml$", name).group(1))  # type: ignore
            )
            for idx, slide_path in enumerate(slide_files):
                xml_bytes = zf.read(slide_path)
                root = ET.fromstring(xml_bytes)
                texts: List[str] = []
                for elem in root.iter():
                    if elem.tag.endswith("}t") and elem.text:
                        texts.append(elem.text.strip())
                slide_text = " ".join(t for t in texts if t)
                if slide_text.strip():
                    pages.append(
                        DocumentPage(
                            file_name=file_name,
                            page_number=idx + 1,
                            text=slide_text.strip(),
                            doc_type="slides",
                        )
                    )
    except Exception:
        pass
    return pages


def parse_docx_bytes(raw_bytes: bytes, file_name: str) -> List[DocumentPage]:
    """Extract text from a .docx file using zipfile + word/document.xml."""
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
            if "word/document.xml" not in zf.namelist():
                return []
            xml_bytes = zf.read("word/document.xml")
            root = ET.fromstring(xml_bytes)
            paragraphs: List[str] = []
            for p in root.iter():
                if p.tag.endswith("}p"):
                    runs = [t.text.strip() for t in p.iter() if t.tag.endswith("}t") and t.text]
                    para_txt = " ".join(r for r in runs if r)
                    if para_txt:
                        paragraphs.append(para_txt)
            full_text = "\n\n".join(paragraphs)
            return _split_text_into_virtual_pages(full_text, file_name, doc_type="docx")
    except Exception:
        return []


def _split_text_into_virtual_pages(
    full_text: str, file_name: str, doc_type: str = "document", chars_per_page: int = 1600
) -> List[DocumentPage]:
    """
    Split plain text or markdown into page-numbered sections.
    Honors explicit `--- PAGE N ---` or `# Page N` markers if present,
    otherwise splits cleanly along paragraph boundaries every ~1600 characters.
    """
    explicit_pattern = re.compile(
        r"(?:^|\n)(?:---\s*PAGE\s*(\d+)\s*---|===\s*PAGE\s*(\d+)\s*===|\f)",
        re.IGNORECASE,
    )
    if explicit_pattern.search(full_text):
        parts = re.split(r"(?:^|\n)(?:---\s*PAGE\s*\d+\s*---|===\s*PAGE\s*\d+\s*===|\f)", full_text, flags=re.IGNORECASE)
        pages: List[DocumentPage] = []
        page_num = 1
        for part in parts:
            cleaned = part.strip()
            if cleaned:
                pages.append(
                    DocumentPage(
                        file_name=file_name,
                        page_number=page_num,
                        text=cleaned,
                        doc_type=doc_type,
                    )
                )
                page_num += 1
        return pages

    # Otherwise group paragraphs into virtual pages of ~chars_per_page
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", full_text) if p.strip()]
    pages = []
    current_paras: List[str] = []
    current_len = 0
    page_num = 1

    for para in paragraphs:
        if current_len + len(para) > chars_per_page and current_paras:
            pages.append(
                DocumentPage(
                    file_name=file_name,
                    page_number=page_num,
                    text="\n\n".join(current_paras),
                    doc_type=doc_type,
                )
            )
            page_num += 1
            current_paras = [para]
            current_len = len(para)
        else:
            current_paras.append(para)
            current_len += len(para)

    if current_paras:
        pages.append(
            DocumentPage(
                file_name=file_name,
                page_number=page_num,
                text="\n\n".join(current_paras),
                doc_type=doc_type,
            )
        )

    return pages


def parse_document(
    file_name: str,
    content: bytes | str,
    doc_type_hint: Optional[str] = None,
) -> List[DocumentPage]:
    """
    Unified entrypoint to parse any uploaded file (PDF, PPTX, DOCX, HTML, Markdown, TXT)
    into a list of page-annotated `DocumentPage` objects.
    """
    ext = Path(file_name).suffix.lower()
    raw_bytes = content.encode("utf-8") if isinstance(content, str) else content

    if ext == ".pdf":
        return parse_pdf_bytes(raw_bytes, file_name)
    elif ext == ".pptx":
        return parse_pptx_bytes(raw_bytes, file_name)
    elif ext == ".docx":
        return parse_docx_bytes(raw_bytes, file_name)
    elif ext in (".html", ".htm"):
        txt = html_to_text(raw_bytes.decode("utf-8", errors="ignore"))
        return _split_text_into_virtual_pages(txt, file_name, doc_type=doc_type_hint or "webpage")
    else:
        txt = raw_bytes.decode("utf-8", errors="ignore")
        dtype = doc_type_hint or ("markdown" if ext == ".md" else "document")
        return _split_text_into_virtual_pages(txt, file_name, doc_type=dtype)


def parse_url_or_html(url: str, html_override: Optional[str] = None) -> List[DocumentPage]:
    """Fetch a campus webpage URL (or use provided HTML) and extract pages with citations."""
    clean_name = re.sub(r"^https?://", "", url).strip("/").replace("/", "_") or "campus_webpage.html"
    if not clean_name.endswith((".html", ".htm", ".pdf")):
        clean_name += ".html"

    if html_override is not None:
        txt = html_to_text(html_override)
        return _split_text_into_virtual_pages(txt, clean_name, doc_type="webpage")

    req = urllib.request.Request(
        url,
        headers={"User-Agent": "CampusRAGAssistant/1.0 (Educational RAG Bot)"},
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        raw = resp.read().decode("utf-8", errors="ignore")
    txt = html_to_text(raw)
    return _split_text_into_virtual_pages(txt, clean_name, doc_type="webpage")


def _extract_heading(text: str) -> str:
    """Detect the most prominent section heading inside a page or chunk."""
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith("#"):
            return s.lstrip("#").strip()[:90]
        if re.match(r"^(?:Section|Article|Module|Unit|Week|Chapter|Policy|Rule|Circular)\s+\d+", s, re.IGNORECASE):
            return s[:90]
        if len(s) < 75 and s.isupper():
            return s.title()
    first_line = text.strip().splitlines()[0] if text.strip() else ""
    return first_line[:80]


def chunk_pages(
    pages: List[DocumentPage],
    chunk_size: int = 550,
    chunk_overlap: int = 120,
) -> List[DocumentChunk]:
    """
    Split a list of `DocumentPage` objects into overlapping `DocumentChunk` items.
    Preserves exact `file_name` and `page_number` on every chunk for page-level citations.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    all_chunks: List[DocumentChunk] = []
    global_idx = 0

    for page in pages:
        text = re.sub(r"[ \t]+", " ", page.text).strip()
        if not text:
            continue

        heading = _extract_heading(text)
        n = len(text)

        if n <= chunk_size:
            chunk_id = f"{page.file_name}::p{page.page_number}::c{global_idx}"
            all_chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    file_name=page.file_name,
                    page_number=page.page_number,
                    text=text,
                    chunk_index=global_idx,
                    char_start=0,
                    char_end=n,
                    doc_type=page.doc_type,
                    section_title=heading,
                )
            )
            global_idx += 1
            continue

        start = 0
        step = max(1, chunk_size - chunk_overlap)
        while start < n:
            end = min(start + chunk_size, n)
            # Snap to sentence or word boundary if not at end of page
            if end < n:
                sentence_break = max(
                    text.rfind(". ", start + chunk_size // 2, end),
                    text.rfind("\n", start + chunk_size // 2, end),
                )
                if sentence_break != -1 and sentence_break > start:
                    end = sentence_break + 1
                else:
                    space_break = text.rfind(" ", start + chunk_size // 2, end)
                    if space_break != -1 and space_break > start:
                        end = space_break

            sub_text = text[start:end].strip()
            if sub_text:
                sub_heading = _extract_heading(sub_text) or heading
                chunk_id = f"{page.file_name}::p{page.page_number}::c{global_idx}"
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        file_name=page.file_name,
                        page_number=page.page_number,
                        text=sub_text,
                        chunk_index=global_idx,
                        char_start=start,
                        char_end=end,
                        doc_type=page.doc_type,
                        section_title=sub_heading,
                    )
                )
                global_idx += 1

            if end >= n:
                break
            start = max(start + 1, end - chunk_overlap)

    return all_chunks


def build_minimal_pdf(pages_text: List[str]) -> bytes:
    """
    Generate a standards-compliant multi-page PDF document in pure Python (zero external dependencies)
    so sample campus syllabi, circulars, and lecture notes exist as real .pdf files on disk.
    """
    objects: List[bytes] = []

    def add_obj(body: bytes) -> int:
        objects.append(body)
        return len(objects)

    # 1: Catalog, 2: Pages placeholder, 3: Font
    add_obj(b"<< /Type /Catalog /Pages 2 0 R >>")
    add_obj(b"")  # placeholder for Pages
    font_obj_id = add_obj(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    page_obj_ids: List[int] = []

    for page_str in pages_text:
        # Wrap lines to ~85 chars
        wrapped_lines: List[str] = []
        for raw_line in page_str.splitlines():
            line = raw_line.strip()
            if not line:
                wrapped_lines.append("")
                continue
            while len(line) > 88:
                split_idx = line.rfind(" ", 0, 88)
                if split_idx <= 0:
                    split_idx = 88
                wrapped_lines.append(line[:split_idx])
                line = line[split_idx:].strip()
            wrapped_lines.append(line)

        stream_lines = ["BT", "/F1 10 Tf", "50 750 Td", "14 TL"]
        for wl in wrapped_lines:
            escaped = (
                wl.replace("\\", "\\\\")
                .replace("(", "\\(")
                .replace(")", "\\)")
            )
            stream_lines.append(f"({escaped}) Tj T*")
        stream_lines.append("ET")
        stream_data = "\n".join(stream_lines).encode("latin-1", errors="ignore")
        compressed = zlib.compress(stream_data)

        content_id = add_obj(
            f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode("ascii")
            + compressed
            + b"\nendstream"
        )
        page_id = add_obj(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_obj_id} 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode("ascii")
        )
        page_obj_ids.append(page_id)

    kids_str = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
    objects[1] = f"<< /Type /Pages /Kids [{kids_str}] /Count {len(page_obj_ids)} >>".encode("ascii")

    out = io.BytesIO()
    out.write(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets: List[int] = []
    for idx, obj_bytes in enumerate(objects, start=1):
        offsets.append(out.tell())
        out.write(f"{idx} 0 obj\n".encode("ascii"))
        out.write(obj_bytes)
        out.write(b"\nendobj\n")

    xref_pos = out.tell()
    out.write(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    out.write(b"0000000000 65535 f \n")
    for off in offsets:
        out.write(f"{off:010d} 00000 n \n".encode("ascii"))

    out.write(
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "ascii"
        )
    )
    return out.getvalue()
