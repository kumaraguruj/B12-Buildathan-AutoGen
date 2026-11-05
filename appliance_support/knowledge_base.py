"""FAISS vector DB knowledge base built from PDF manuals in ./kb_docs.

Flow:  kb_docs/*.pdf  ->  read text  ->  chunk  ->  OpenAI embeddings  ->  FAISS

Ingestion is idempotent: each file is fingerprinted (SHA-1). Unchanged files are skipped,
changed files are re-embedded, and chunks of deleted files are removed.
"""
from __future__ import annotations

import hashlib
import os
import re
from collections import Counter
from io import BytesIO

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

SUPPORTED = (".pdf",)
SPLITTER = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)


# ---------------------------------------------------------------- reading
def _strip_repeated_lines(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop running headers/footers: lines (digits ignored) that appear on most pages of a multi-page PDF."""
    if len(pages) < 2:
        return pages
    norm = lambda ln: re.sub(r"\d+", "#", ln.strip())
    counts: Counter = Counter()
    for _, text in pages:
        counts.update({norm(ln) for ln in text.splitlines() if ln.strip()})
    repeated = {k for k, c in counts.items() if c >= max(2, 0.6 * len(pages))}
    cleaned = []
    for n, text in pages:
        kept = "\n".join(ln for ln in text.splitlines() if norm(ln) not in repeated).strip()
        cleaned.append((n, kept))
    return cleaned


def read_document(name: str, raw: bytes) -> tuple[str, list[tuple[int, str, str]]]:
    """Return the document title and each non-empty page with its bookmark title, if available."""
    if not name.lower().endswith(SUPPORTED):
        raise ValueError("Knowledge-base documents must be PDF files.")

    from pypdf import PdfReader
    reader = PdfReader(BytesIO(raw))
    doc_title = (reader.metadata.title if reader.metadata and reader.metadata.title else None) \
        or os.path.splitext(name)[0]
    sections: dict[int, str] = {}
    try:
        def walk(items):
            for item in items:
                if isinstance(item, list):
                    walk(item)
                else:
                    sections.setdefault(reader.get_destination_page_number(item) + 1, item.title)
        walk(reader.outline)
    except Exception:  # malformed outline - ignore, use document title
        sections = {}
    pages = [(i + 1, (page.extract_text() or "").strip()) for i, page in enumerate(reader.pages)]
    pages = _strip_repeated_lines(pages)
    return doc_title, [(number, text, sections.get(number, doc_title))
                       for number, text in pages if text]


QUESTION_RE = re.compile(r"^Q\d+\.\s+(.*)")      # "Q3. Refrigerator: Not cooling"  -> starts an FAQ entry
SECTION_RE = re.compile(r"^Section\s+\d+:\s")     # "Section 2: Refrigerator"        -> heading, not embedded


def split_by_question(pages: list[tuple[int, str, str]]) -> list[tuple[int, str, str]] | None:
    """Split a continuous FAQ into one entry per "Qn." heading, even when entries share a page.

    Returns [(page_of_heading, question_title, text)] or None if the document has no Qn. headings.
    Text before the first question (cover/intro) and "Section n:" headings are dropped.
    """
    entries: list[list] = []          # [page, title, [lines]]
    current = None
    found = False
    for page_no, text, _ in pages:
        for line in text.splitlines():
            m = QUESTION_RE.match(line.strip())
            if m:
                found = True
                current = [page_no, m.group(1).strip(), [line.strip()]]
                entries.append(current)
            elif SECTION_RE.match(line.strip()):
                current = None
            elif current is not None:
                current[2].append(line)
    if not found:
        return None
    return [(pg, title, "\n".join(lines).strip()) for pg, title, lines in entries]


def file_to_chunks(name: str, raw: bytes, source: str = "kb_docs") -> list[Document]:
    """Read a file and return embeddable chunks (one or more per FAQ question / page) with metadata."""
    _, pages = read_document(name, raw)
    digest = hashlib.sha1(raw).hexdigest()
    entries = split_by_question(pages)
    if entries is None:   # no "Qn." headings: fall back to page-level chunks titled by bookmark / file name
        entries = [(pg, section, text) for pg, text, section in pages]
    docs: list[Document] = []
    for page_no, title, text in entries:
        for chunk in SPLITTER.split_text(text):
            docs.append(Document(page_content=chunk,
                                 metadata={"title": title, "file": name, "page": page_no,
                                           "file_hash": digest, "source": source}))
    return docs


# ---------------------------------------------------------------- FAISS helpers
def _get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=config.EMBED_MODEL)


def _load_faiss() -> FAISS | None:
    """Load a saved FAISS index from disk, or return None if it doesn't exist."""
    if os.path.isdir(config.FAISS_DIR) and os.path.isfile(os.path.join(config.FAISS_DIR, "index.faiss")):
        return FAISS.load_local(config.FAISS_DIR, _get_embeddings(),
                                allow_dangerous_deserialization=True)
    return None


def _save_faiss(vs: FAISS) -> None:
    """Persist the FAISS index to disk."""
    vs.save_local(config.FAISS_DIR)


def _count_docs(vs: FAISS) -> int:
    """Return total number of documents in the FAISS store."""
    return len(vs.index_to_docstore_id)


def _delete_by_file(vs: FAISS, file_name: str) -> int:
    """Delete all chunks whose metadata['file'] == file_name. Returns count removed."""
    ids_to_delete = [
        doc_id for doc_id in vs.index_to_docstore_id.values()
        if isinstance((doc := vs.docstore.search(doc_id)), Document)
        and doc.metadata.get("file") == file_name
    ]
    if ids_to_delete:
        vs.delete(ids_to_delete)
    return len(ids_to_delete)


# ---------------------------------------------------------------- vector store
def get_vectorstore() -> FAISS:
    """Open the persistent FAISS index and sync it with the kb_docs folder."""
    vs = _load_faiss()
    if vs is None:
        # Create an empty FAISS store with a dummy doc, then remove it
        dummy = Document(page_content="init", metadata={"file": "__init__"})
        vs = FAISS.from_documents([dummy], _get_embeddings())
        _delete_by_file(vs, "__init__")
    sync_folder(vs)
    return vs


def indexed_files(vs: FAISS) -> dict[str, dict]:
    """{file_name: {"hash": ..., "chunks": n}} for everything currently embedded."""
    out: dict[str, dict] = {}
    for doc_id in vs.index_to_docstore_id.values():
        doc = vs.docstore.search(doc_id)
        if not isinstance(doc, Document):
            continue
        f = doc.metadata.get("file")
        if not f:
            continue
        entry = out.setdefault(f, {"hash": doc.metadata.get("file_hash"), "chunks": 0})
        entry["chunks"] += 1
    return out


def ingest_file(vs: FAISS, name: str, raw: bytes) -> int:
    """(Re)embed one file. Existing chunks of the same file name are replaced."""
    docs = file_to_chunks(name, raw)
    _delete_by_file(vs, name)
    if docs:
        vs.add_documents(docs)
    _save_faiss(vs)
    return len(docs)


def sync_folder(vs: FAISS, folder: str | None = None) -> dict[str, int]:
    """Embed new/changed files in the folder and drop chunks of removed files.

    Returns {"added": n_files, "updated": n_files, "removed": n_files, "skipped": n_files, "chunks": n_new_chunks}.
    """
    folder = folder or config.KB_DIR
    os.makedirs(folder, exist_ok=True)
    existing = indexed_files(vs)
    stats = {"added": 0, "updated": 0, "removed": 0, "skipped": 0, "chunks": 0}

    on_disk = sorted(f for f in os.listdir(folder) if f.lower().endswith(SUPPORTED))
    for name in on_disk:
        with open(os.path.join(folder, name), "rb") as fh:
            raw = fh.read()
        digest = hashlib.sha1(raw).hexdigest()
        if name in existing and existing[name]["hash"] == digest:
            stats["skipped"] += 1
            continue
        stats["updated" if name in existing else "added"] += 1
        stats["chunks"] += ingest_file(vs, name, raw)

    for name in set(existing) - set(on_disk):
        _delete_by_file(vs, name)
        stats["removed"] += 1

    _save_faiss(vs)
    return stats


def save_upload(name: str, raw: bytes, folder: str | None = None) -> str:
    """Persist an uploaded manual into the KB folder so it survives re-syncs."""
    if not name.lower().endswith(SUPPORTED):
        raise ValueError("Knowledge-base uploads must be PDF files.")
    folder = folder or config.KB_DIR
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, os.path.basename(name))
    with open(path, "wb") as fh:
        fh.write(raw)
    return path
