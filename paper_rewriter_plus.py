#!/usr/bin/env python3
"""
Paper Rewriter Plus — model/RAG-ready academic rewriter (local LLM + Chroma + streaming)
---------------------------------------------------------------------------------------

What this tool does
- Reads .docx, .pdf, .md/.txt, or .html and extracts clean text (heading-aware)
- Splits text into token-aware chunks with overlap
- Runs one or more tasks (rewrite, review, extract, outline, figure suggestions, etc.)
- Optional RAG: retrieves top-k snippets from a ChromaDB index and injects them per chunk
- Supports OpenAI cloud and OpenAI-compatible localhost servers (vLLM, Ollama, LM Studio, oobabooga WebUI)
- Optional streaming for interactive runs (default OFF)

Install
    pip install -U "openai>=1.40.0" tiktoken>=0.7.0 chromadb>=0.5.3 sentence-transformers>=3.0.1
    pip install -U python-docx>=1.1.2 pdfminer.six>=20231228 html2text>=2024.2.26 markdown-it-py>=3.0.0
    pip install -U rich>=13.7.1 pypdf>=4.2.0

Local endpoint example (Ollama / oobabooga / vLLM)
    export OPENAI_API_KEY=sk-local
    export OPENAI_BASE_URL=http://127.0.0.1:5000/v1

RAG ingest
    python paper_rewriter_plus.py ingest --inputs ./input_pdfs --chroma_path ./ragdb --collection papers --embed_model "BAAI/bge-large-en-v1.5" --chunk_chars 1000 --chunk_overlap 30 --embed_token_limit 480

Process with multiple tasks + RAG
    python paper_rewriter_plus.py process --input my_paper.pdf --outdir outputs --task rewrite_academic --task figure_suggestions --rag --k 6 --context_tokens 1200 --chroma_path ./ragdb --collection papers --model gpt-4o-mini

Notes
- The similarity percentage (--similarity) is a rough n-gram overlap proxy, not a plagiarism score.
- Streaming (--stream) improves UX but not quality; default OFF to keep batch outputs stable.
"""

from __future__ import annotations
import argparse
import dataclasses
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger('paper_rewriter_plus')

# ---------- Tokenization ----------
try:
    import tiktoken
except Exception as e:  # pragma: no cover
    logger.error("tiktoken is required. Install with: pip install tiktoken")
    raise

# ---------- LLM client with OpenAI-compatible local endpoints ----------
# If OPENAI_BASE_URL is set (e.g., http://localhost:11434/v1 or http://127.0.0.1:5000/v1),
# the official OpenAI SDK routes requests there.
_OPENAI_BASE_URL = os.getenv('OPENAI_BASE_URL')
_OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', 'sk-local')

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None
    logger.warning("OpenAI SDK not installed. pip install openai")

if OpenAI is not None:
    _openai_client = OpenAI(base_url=_OPENAI_BASE_URL, api_key=_OPENAI_API_KEY) if _OPENAI_BASE_URL else OpenAI(api_key=_OPENAI_API_KEY)
else:
    _openai_client = None

# ---------- Document readers ----------
try:
    from docx import Document as _Docx
except Exception:
    _Docx = None

try:
    from pdfminer.high_level import extract_text as pdf_extract_text
except Exception:
    pdf_extract_text = None

try:
    import html2text as _html2text
except Exception:
    _html2text = None

try:
    from pypdf import PdfReader  # fallback
except Exception:
    PdfReader = None

# ---------- RAG: ChromaDB + Sentence-Transformers ----------
try:
    import chromadb
    from chromadb.config import Settings as _ChromaSettings
except Exception:
    chromadb = None
    _ChromaSettings = None

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# ---------- Prompt Library (domain-agnostic) ----------
@dataclass
class PromptLibrary:
    SYSTEM_RESEARCH_ASSISTANT: str = (
        "You are a meticulous research assistant and scientific writing editor. "
        "You write in clear academic prose, keep claims precise, avoid hallucinations, "
        "and add TODO-style citation markers [n] where external sources are needed."
    )

    TASKS: Dict[str, str] = dataclasses.field(default_factory=lambda: {
        'rewrite_academic': (
            "Rewrite the provided section into a coherent, engaging academic text in active voice. "
            "Keep technical accuracy; avoid assumptions not supported by the input. "
            "Preserve all core facts; improve organization and readability. "
            "Insert citation markers as [1], [2] where claims need support. "
            "Use LaTeX math mode for equations if present."
        ),
        'edit_and_review': (
            "Fix grammar and style; propose a concise, informative section title. "
            "Return:\n1) Title\n2) Polished text\n3) Notes: bullet points with strengths, risks, missing citations."
        ),
        'summary_cim': (
            "Summarize the section in <=650 characters for an executive memo. Preserve key facts and numbers."
        ),
        'extract_quanti': (
            "Extract all chemicals (with quantities/units), materials, devices/instruments, and statistical outcomes "
            "(e.g., p-values, confidence intervals) into a Markdown table with columns: Category | Item | Value | Unit | Context. "
            "Then list all citations found in the text. If none, write 'None'."
        ),
        'outline': (
            "Create a structured outline (H1..H4) for a full paper section based on the text. Keep it general-purpose."
        ),
        'figure_suggestions': (
            "Identify optimal positions in the text to embed figures. For each slot: provide an anchor quote, "
            "a short rationale, and three caption variants: Poor, Better, Precise."
        ),
        'citation_scout': (
            "Insert [Ref Needed] markers and list web-search keywords per marker to locate appropriate sources."
        ),
        'peer_review': (
            "Provide a reviewer-style critique grouped into Major Issues, Minor Issues, and Typos/Edits."
        ),
    })

# ---------- Utility: token-aware chunking ----------

def get_encoding(model: str) -> tiktoken.Encoding:
    try:
        return tiktoken.encoding_for_model(model)
    except Exception:
        return tiktoken.get_encoding('cl100k_base')


def count_tokens(text: str, model: str) -> int:
    enc = get_encoding(model)
    return len(enc.encode(text or ''))


def chunk_text(text: str, model: str, max_tokens: int = 6000, overlap_tokens: int = 200) -> List[str]:
    enc = get_encoding(model)
    paras = re.split(r"(\n\s*\n)+", (text or '').strip())
    chunks: List[str] = []
    buf: List[str] = []
    buf_toks = 0

    def flush(force=False):
        nonlocal buf, buf_toks
        if buf and (force or buf_toks > 0):
            chunk = ''.join(buf).strip()
            if chunk:
                chunks.append(chunk)
            buf = []
            buf_toks = 0

    for p in paras:
        ptoks = len(enc.encode(p))
        if ptoks > max_tokens:
            tokens = enc.encode(p)
            start = 0
            while start < len(tokens):
                end = min(start + max_tokens, len(tokens))
                piece = enc.decode(tokens[start:end])
                if buf_toks:
                    flush(force=True)
                chunks.append(piece)
                start = max(0, end - overlap_tokens)
            continue
        if buf_toks + ptoks <= max_tokens:
            buf.append(p)
            buf_toks += ptoks
        else:
            flush(force=True)
            buf = [p]
            buf_toks = ptoks
    flush(force=True)
    return chunks

# ---------- Similarity (rough proxy) ----------

def similarity_ratio(a: str, b: str, n: int = 5) -> float:
    def shingles(s: str) -> set:
        toks = re.findall(r"\w+", (s or '').lower())
        return set(tuple(toks[i:i+n]) for i in range(max(0, len(toks)-n+1)))
    A, B = shingles(a), shingles(b)
    if not A or not B:
        return 0.0
    return len(A & B) / max(1, len(A | B))

# ---------- IO helpers ----------

def read_text_from_docx(path: Path) -> str:
    if _Docx is None:
        raise RuntimeError('python-docx not installed. pip install python-docx')
    doc = _Docx(str(path))
    lines = []
    for para in doc.paragraphs:
        txt = para.text.strip("\u000b")
        if txt:
            style = getattr(para.style, 'name', '') or ''
            if style.startswith('Heading'):
                lines.append('\n\n' + txt + '\n\n')
            else:
                lines.append(txt)
    return '\n'.join(lines)


def read_text_from_pdf(path: Path) -> str:
    if pdf_extract_text is not None:
        try:
            return pdf_extract_text(str(path))
        except Exception:
            pass
    if PdfReader is not None:
        try:
            r = PdfReader(str(path))
            return '\n'.join(page.extract_text() or '' for page in r.pages)
        except Exception:
            raise
    raise RuntimeError('No PDF reader available. Install pdfminer.six or pypdf')


def read_text_from_html(path: Path) -> str:
    if _html2text is None:
        raise RuntimeError('html2text not installed. pip install html2text')
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        html_content = f.read()
    return _html2text.html2text(html_content)


def read_text_generic(path: Path) -> str:
    s = path.suffix.lower()
    if s == '.docx':
        return read_text_from_docx(path)
    if s == '.pdf':
        return read_text_from_pdf(path)
    if s in {'.html', '.htm'}:
        return read_text_from_html(path)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

# ---------- LLM call (OpenAI-compatible, optional streaming ready) ----------

def call_chat_model(task_prompt: str, content: str, *, model: str, system_prompt: str, temperature: float = 0.2, max_tokens: int = 1200, stream: bool = False) -> str:
    if _openai_client is None:
        raise RuntimeError('OpenAI-compatible client not initialized. Install openai and/or set OPENAI_BASE_URL.')

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Task: {task_prompt}\n\nInput:\n" + content},
    ]

    if not stream:
        resp = _openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or '').strip()

    # Stream mode
    resp = _openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=True,
    )
    parts: List[str] = []
    for chunk in resp:
        delta = getattr(chunk.choices[0].delta, 'content', None)
        if delta:
            parts.append(delta)
            print(delta, end='', flush=True)
    print()
    return ''.join(parts).strip()

# ---------- RAG helpers ----------
@dataclass
class RAGConfig:
    enabled: bool = False
    chroma_path: Optional[Path] = None
    collection: str = 'papers'
    embed_model: str = 'BAAI/bge-large-en-v1.5'
    k: int = 6
    context_tokens: int = 1000

# Token-aware helpers for embedding-time safety

def hf_token_len(tokenizer, text: str) -> int:
    try:
        return len(tokenizer.encode(text, add_special_tokens=True))
    except Exception:
        # Fallback if tokenizer lacks encode signature
        return len((text or '').split())


def token_aware_chunks(text: str, tokenizer, max_tokens: int = 480) -> List[str]:
    """Split text so each chunk stays under the model's token limit to avoid PE mismatches.
    Greedy on sentence-ish/paragraph boundaries; hard-wraps only when necessary.
    """
    text = text or ''
    # First split on paragraph/sentence boundaries
    parts = re.split(r"(?<=[\.!?])\s+|\n\n+", text)
    chunks: List[str] = []
    buf_txt = ''

    def flush():
        nonlocal buf_txt
        if buf_txt.strip():
            chunks.append(buf_txt.strip())
            buf_txt = ''

    for p in parts:
        if not p.strip():
            continue
        cand = (buf_txt + (" " if buf_txt else "") + p).strip()
        if hf_token_len(tokenizer, cand) <= max_tokens:
            buf_txt = cand
        else:
            if buf_txt:
                flush()
            # If single part still too big, hard wrap by words
            if hf_token_len(tokenizer, p) > max_tokens:
                words = p.split()
                cur = []
                for w in words:
                    c2 = (" ".join(cur + [w])).strip()
                    if hf_token_len(tokenizer, c2) <= max_tokens:
                        cur.append(w)
                    else:
                        chunks.append(" ".join(cur).strip())
                        cur = [w]
                if cur:
                    chunks.append(" ".join(cur).strip())
            else:
                buf_txt = p.strip()
    if buf_txt:
        flush()
    return [c for c in chunks if c]

class RAG:
    def __init__(self, cfg: RAGConfig):
        if not cfg.enabled:
            self.enabled = False
            return
        if chromadb is None or SentenceTransformer is None:
            raise RuntimeError('RAG requested but chromadb/sentence-transformers are not installed.')
        self.enabled = True
        self.cfg = cfg
        # Disable telemetry to avoid noisy warnings
        settings = _ChromaSettings(anonymized_telemetry=False) if _ChromaSettings else None
        self.client = chromadb.PersistentClient(path=str(cfg.chroma_path), settings=settings)
        self.collection = self.client.get_or_create_collection(name=cfg.collection)
        self.embedder = SentenceTransformer(cfg.embed_model)
        # Respect the model's own max length (e.g., 512 for BGE); avoid forcing larger than supported
        tok_max = getattr(self.embedder, 'tokenizer', None)
        tok_max = getattr(tok_max, 'model_max_length', 512)
        if not isinstance(tok_max, int) or tok_max <= 0 or tok_max > 100000:
            tok_max = 512
        try:
            self.embedder.max_seq_length = tok_max
        except Exception:
            pass

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.embedder.encode(texts, normalize_embeddings=True).tolist()

    def top_k(self, query_text: str) -> List[str]:
        if not self.enabled:
            return []
        qv = self.embed([query_text])[0]
        res = self.collection.query(query_embeddings=[qv], n_results=self.cfg.k, include=['documents'])
        docs = []
        for group in res.get('documents', []):
            docs.extend(group)
        return docs

    def build_context_block(self, query_text: str, model: str) -> str:
        if not self.enabled:
            return ''
        snippets = self.top_k(query_text)
        if not snippets:
            return ''
        enc = get_encoding(model)
        acc: List[str] = []
        tok_total = 0
        for s in snippets:
            stoks = len(enc.encode(s))
            if tok_total + stoks > self.cfg.context_tokens:
                break
            acc.append(s.strip())
            tok_total += stoks
        if not acc:
            return ''
        return "\n\n[RAG context]\n" + "\n---\n".join(acc) + "\n\n"

# ---------- Core processing ----------

def process_text(
    raw_text: str,
    *,
    task_key: str,
    model: str,
    prompts: PromptLibrary,
    rag: Optional[RAG] = None,
    max_context_tokens: int = 6000,
    overlap_tokens: int = 200,
    include_similarity: bool = False,
    stream: bool = False,
) -> Tuple[str, List[str]]:
    if task_key not in prompts.TASKS:
        raise KeyError(f"Unknown task '{task_key}'. Options: {sorted(prompts.TASKS)}")
    task_prompt = prompts.TASKS[task_key]
    chunks = chunk_text(raw_text, model, max_tokens=max_context_tokens, overlap_tokens=overlap_tokens)
    outputs: List[str] = []
    logger.info("Processing %d chunks with '%s'%s...", len(chunks), task_key, ' + RAG' if rag and rag.enabled else '')

    for i, ch in enumerate(chunks, 1):
        logger.info(" → Chunk %d/%d (%d tokens)", i, len(chunks), count_tokens(ch, model))
        ch_with_ctx = ch
        if rag and rag.enabled:
            ctx = rag.build_context_block(ch, model)
            if ctx:
                ch_with_ctx = ctx + ch
        out = call_chat_model(
            task_prompt,
            ch_with_ctx,
            model=model,
            system_prompt=prompts.SYSTEM_RESEARCH_ASSISTANT,
            stream=stream,
        )
        if include_similarity:
            sim = similarity_ratio(ch, out)
            out = f"[Similarity≈{sim*100:.1f}% — n-gram overlap proxy]\n\n" + out
        outputs.append(out)
    stitched = '\n\n'.join(outputs)
    return stitched, outputs

# ---------- Figure workflow convenience ----------

def run_figure_suggestions(text: str, *, model: str, prompts: PromptLibrary, rag: Optional[RAG] = None, stream: bool = False) -> str:
    stitched, _ = process_text(
        text,
        task_key='figure_suggestions',
        model=model,
        prompts=prompts,
        rag=rag,
        max_context_tokens=5500,
        overlap_tokens=150,
        include_similarity=False,
        stream=stream,
    )
    return stitched

# ---------- Filesystem orchestration ----------

def safe_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info('Wrote %s', path)

# ---------- RAG: ingest and query utilities (CLI subcommands) ----------

def _pdf_to_text(file_path: Path) -> str:
    try:
        return read_text_from_pdf(file_path)
    except Exception:
        return ''


def _split_into_chunks(text: str, *, chunk_chars: int = 1500, chunk_overlap: int = 200) -> List[str]:
    text = text or ''
    chunks: List[str] = []
    i = 0
    while i < len(text):
        end = min(len(text), i + chunk_chars)
        chunks.append(text[i:end])
        i = max(i + chunk_chars - chunk_overlap, end)
    return [c.strip() for c in chunks if c.strip()]


def rag_ingest(inputs: Path, *, chroma_path: Path, collection: str, embed_model: str, chunk_chars: int, chunk_overlap: int, embed_token_limit: Optional[int] = None):
    if chromadb is None or SentenceTransformer is None:
        raise RuntimeError('Install chromadb and sentence-transformers for ingestion.')
    settings = _ChromaSettings(anonymized_telemetry=False) if _ChromaSettings else None
    client = chromadb.PersistentClient(path=str(chroma_path), settings=settings)
    coll = client.get_or_create_collection(name=collection)
    embedder = SentenceTransformer(embed_model)
    # Respect the model's own token limit; avoid artificially inflating it
    tok_max = getattr(embedder, 'tokenizer', None)
    tok_max = getattr(tok_max, 'model_max_length', 512)
    if not isinstance(tok_max, int) or tok_max <= 0 or tok_max > 100000:
        tok_max = 512
    try:
        embedder.max_seq_length = tok_max
    except Exception:
        pass

    pdfs = sorted([p for p in inputs.glob('**/*') if p.suffix.lower() == '.pdf'])
    if not pdfs:
        logger.warning('No PDFs found in %s', inputs)
    for pdf in pdfs:
        logger.info('Ingesting %s', pdf.name)
        text = _pdf_to_text(pdf)
        # Prefer token-aware chunks to ensure every embedded chunk <= model limit
        limit = embed_token_limit if (embed_token_limit and embed_token_limit > 0) else (tok_max - 32 if tok_max > 64 else max(tok_max - 4, 32))
        chunks = token_aware_chunks(text, embedder.tokenizer, max_tokens=limit)
        if not chunks:
            logger.warning('No text extracted from %s', pdf)
            continue
        vectors = embedder.encode(chunks, normalize_embeddings=True).tolist()
        ids = [f"{pdf.name}_{i:05d}" for i in range(len(chunks))]
        metadatas = [{"source": pdf.name, "path": str(pdf)} for _ in chunks]
        B = 256
        for b in range(0, len(chunks), B):
            coll.add(ids=ids[b:b+B], embeddings=vectors[b:b+B], documents=chunks[b:b+B], metadatas=metadatas[b:b+B])
    logger.info('Ingestion complete into collection=%s at %s', collection, chroma_path)


def rag_query(query: str, *, chroma_path: Path, collection: str, embed_model: str, k: int) -> List[str]:
    if chromadb is None or SentenceTransformer is None:
        raise RuntimeError('Install chromadb and sentence-transformers for querying.')
    settings = _ChromaSettings(anonymized_telemetry=False) if _ChromaSettings else None
    client = chromadb.PersistentClient(path=str(chroma_path), settings=settings)
    coll = client.get_or_create_collection(name=collection)
    embedder = SentenceTransformer(embed_model)
    qv = embedder.encode([query], normalize_embeddings=True).tolist()[0]
    res = coll.query(query_embeddings=[qv], n_results=k, include=['documents','metadatas'])
    docs: List[str] = []
    for group in res.get('documents', []):
        docs.extend(group)
    return docs

# ---------- Pipeline ----------

def run_pipeline(
    input_path: Path,
    outdir: Path,
    tasks: List[str],
    model: str,
    include_similarity: bool,
    prompts: PromptLibrary,
    rag_cfg: Optional[RAGConfig] = None,
    stream: bool = False,
):
    text = read_text_generic(input_path)
    rag = RAG(rag_cfg) if (rag_cfg and rag_cfg.enabled) else None

    for task in tasks:
        stitched, per_chunks = process_text(
            text,
            task_key=task,
            model=model,
            prompts=prompts,
            rag=rag,
            include_similarity=include_similarity,
            stream=stream,
        )
        base = outdir / input_path.stem
        safe_write(base.with_suffix(f'.{task}.stitched.txt'), stitched)
        for i, chunk in enumerate(per_chunks):
            safe_write(base.parent / f'{input_path.stem}.{task}.part{i+1:03d}.txt', chunk)

# ---------- CLI ----------

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Paper Rewriter Plus — academic rewriting toolkit with RAG, local LLMs, and optional streaming')
    sub = p.add_subparsers(dest='cmd', required=True)

    # process subcommand
    pp = sub.add_parser('process', help='Run rewriting/summarization/extraction tasks')
    pp.add_argument('--input', required=True, type=Path, help='Input file (.pdf, .docx, .txt, .md, .html)')
    pp.add_argument('--outdir', default=Path('outputs'), type=Path, help='Output directory')
    pp.add_argument('--task', action='append', required=True, help='Task key (repeatable). Choices: ' + ', '.join(PromptLibrary().TASKS.keys()))
    pp.add_argument('--model', default=os.getenv('OPENAI_MODEL', 'gpt-4o-mini'), help='Model name (OpenAI or local)')
    pp.add_argument('--similarity', action='store_true', help='Append similarity proxy vs. source')
    pp.add_argument('--stream', action='store_true', help='Stream model output to stdout while accumulating result')
    # RAG options
    pp.add_argument('--rag', action='store_true', help='Enable retrieval-augmented prompting from ChromaDB')
    pp.add_argument('--chroma_path', type=Path, default=Path('./ragdb'), help='ChromaDB persistent directory')
    pp.add_argument('--collection', type=str, default='papers', help='Chroma collection name')
    pp.add_argument('--embed_model', type=str, default='BAAI/bge-large-en-v1.5', help='Sentence-Transformers model')
    pp.add_argument('--k', type=int, default=6, help='Top-k retrieved snippets per chunk')
    pp.add_argument('--context_tokens', type=int, default=1000, help='Max tokens for RAG context per chunk')

    # ingest subcommand
    pi = sub.add_parser('ingest', help='Ingest PDFs into ChromaDB')
    pi.add_argument('--inputs', required=True, type=Path, help='Directory containing PDFs')
    pi.add_argument('--chroma_path', type=Path, default=Path('./ragdb'))
    pi.add_argument('--collection', type=str, default='papers')
    pi.add_argument('--embed_model', type=str, default='BAAI/bge-large-en-v1.5')
    pi.add_argument('--chunk_chars', type=int, default=1500)
    pi.add_argument('--chunk_overlap', type=int, default=200)
    pi.add_argument('--embed_token_limit', type=int, default=480, help='Max tokens per embedded chunk (<= model limit). Use ~480 for BGE-large.')

    # query subcommand
    pq = sub.add_parser('query', help='Query ChromaDB and print top snippets')
    pq.add_argument('--chroma_path', type=Path, default=Path('./ragdb'))
    pq.add_argument('--collection', type=str, default='papers')
    pq.add_argument('--embed_model', type=str, default='BAAI/bge-large-en-v1.5')
    pq.add_argument('--k', type=int, default=8)
    pq.add_argument('query', type=str, help='Free-text query')

    return p


def main(argv: Optional[Sequence[str]] = None):
    args = build_argparser().parse_args(argv)

    if args.cmd == 'ingest':
        rag_ingest(
            inputs=args.inputs,
            chroma_path=args.chroma_path,
            collection=args.collection,
            embed_model=args.embed_model,
            chunk_chars=args.chunk_chars,
            chunk_overlap=args.chunk_overlap,
            embed_token_limit=args.embed_token_limit,
        )
        return

    if args.cmd == 'query':
        snippets = rag_query(
            query=args.query,
            chroma_path=args.chroma_path,
            collection=args.collection,
            embed_model=args.embed_model,
            k=args.k,
        )
        print('\n\n'.join(snippets))
        return

    if args.cmd == 'process':
        if not args.input.exists():
            logger.error('Input not found: %s', args.input)
            sys.exit(2)
        tasks: List[str] = []
        for t in args.task:
            tasks.extend([s.strip() for s in t.split(',') if s.strip()])
        valid = set(PromptLibrary().TASKS.keys())
        for t in tasks:
            if t not in valid:
                logger.error("Unknown task '%s' (valid: %s)", t, sorted(valid))
                sys.exit(2)
        prompts = PromptLibrary()
        rag_cfg = None
        if args.rag:
            rag_cfg = RAGConfig(
                enabled=True,
                chroma_path=args.chroma_path,
                collection=args.collection,
                embed_model=args.embed_model,
                k=args.k,
                context_tokens=args.context_tokens,
            )
        run_pipeline(
            input_path=args.input,
            outdir=args.outdir,
            tasks=tasks,
            model=args.model,
            include_similarity=args.similarity,
            prompts=prompts,
            rag_cfg=rag_cfg,
            stream=args.stream,
        )
        return


if __name__ == '__main__':
    main()
