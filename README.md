# Paper Rewriter Plus

Robust, model-agnostic toolkit to rewrite/summarize/extract academic content with optional RAG (retrieval-augmented generation) over your own PDF library. Works with OpenAI and OpenAI-compatible **local** endpoints (Ollama, LM Studio, vLLM, oobabooga’s OpenAI API mode).

---

## Features
- 📥 **Ingest PDFs** into a local ChromaDB index with **token-aware** chunking (prevents 512-token embedding errors).
- 🔎 **RAG**: retrieve top-K relevant snippets per chunk and auto-inject them into prompts.
- ✍️ **Multi-task processor**: rewrite, edit+review, extract structured data, outline, figure suggestions, etc.
- 🧩 **Local LLMs**: point the OpenAI SDK at a localhost `/v1` endpoint via `OPENAI_BASE_URL`.
- 📡 **Optional streaming** (CLI flag) for live output in interactive runs.
- 🧮 Safe **token counting** (`tiktoken`) and overlap-aware chunking for long inputs.

---

## Repository layout (suggested)
```
.
├── paper_rewriter_plus.py          # main CLI script (RAG + streaming capable)
├── requirements.txt                # Python dependencies
├── README.md                       # this file
├── input_pdfs/                     # put your PDFs here (for ingest)
├── ragdb/                          # ChromaDB persistent dir (created on ingest)
└── outputs/                        # generated text outputs (created on process)
```

> If your canvas shows a file called **“Paper Rewriter Plus (token-aware RAG fix)”**, it’s the same script content ready to save as `paper_rewriter_plus.py`.

---

## Installation

> Python 3.10+ recommended. Create a virtual environment to keep things clean.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### PyTorch note
`sentence-transformers` installs a default CPU torch. For GPU acceleration, install a platform-specific build from [pytorch.org] and then re-run `pip install -r requirements.txt`.

---

## Configure an LLM backend

### Option A: OpenAI
```bash
export OPENAI_API_KEY=sk-...  # required
# Optional: choose model at runtime with --model
```

### Option B: Local OpenAI-compatible server
(e.g., Ollama, vLLM, LM Studio, oobabooga OpenAI API)
```bash
export OPENAI_API_KEY=sk-local          # any non-empty string
export OPENAI_BASE_URL=http://127.0.0.1:5000/v1
# or Ollama: export OPENAI_BASE_URL=http://localhost:11434/v1
```
> Some servers ignore `model`; supply the exact loaded model name if needed.

---

## Step-by-step Quickstart

1) **Prepare input PDFs**
   - Put the articles you want the model to consider in `./input_pdfs/`.

2) **Ingest to Chroma (build the RAG index)**
```bash
python paper_rewriter_plus.py ingest \
  --inputs ./input_pdfs \
  --chroma_path ./ragdb \
  --collection papers \
  --embed_model "BAAI/bge-large-en-v1.5" \
  --chunk_chars 1000 --chunk_overlap 30 \
  --embed_token_limit 480
```
- `--embed_token_limit` keeps each embedding chunk at ≤480 tokens (safe for BGE-large’s 512 limit).
- Telemetry is disabled automatically to avoid noisy warnings.

3) **Rewrite a document with RAG enabled**
```bash
python paper_rewriter_plus.py process \
  --input my_paper.pdf \
  --outdir outputs \
  --task rewrite_academic \
  --model gpt-4o-mini \
  --rag --chroma_path ./ragdb --collection papers \
  --k 6 --context_tokens 1200
```
This produces:
- `outputs/my_paper.rewrite_academic.stitched.txt` – the concatenated result
- `outputs/my_paper.rewrite_academic.partXXX.txt` – per-chunk outputs

4) **(Optional) Stream tokens during processing**
```bash
python paper_rewriter_plus.py process ... --stream
```
Streaming is for UX; output files are written only after each chunk completes.

---

## Common tasks (the `--task` flag)
- `rewrite_academic` – high-quality rewrite with citation markers `[n]`
- `edit_and_review` – polish + section title + notes
- `summary_cim` – ≤650-char executive summary
- `extract_quanti` – table of chemicals/materials/devices/p-values
- `outline` – hierarchical outline (H1–H4)
- `figure_suggestions` – suggested figure slots + graded captions
- `citation_scout` – add `[Ref Needed]` and search keywords
- `peer_review` – reviewer-style critique

> You can run **multiple tasks** by repeating `--task` or comma-separating (e.g., `--task rewrite_academic --task figure_suggestions`).

---

## RAG internals (how it works)
- **Ingest**: PDFs are converted to text, split with a **token-aware** splitter so each chunk fits the embedding model’s positional limit, then embedded with Sentence-Transformers (default: `BAAI/bge-large-en-v1.5`) and stored in Chroma.
- **Process**: For each input chunk, the script fetches top-K snippets from Chroma and injects them at the top of the prompt as a `[RAG context]` block, bounded by `--context_tokens`.

### Choosing an embedding model
- Default: `BAAI/bge-large-en-v1.5` (fast, strong, **512-token limit**). Keep `--embed_token_limit` around 480.
- Larger-context embedding models exist; once you pick one, just raise `--embed_token_limit` accordingly.

---

## CLI reference

### Ingest PDFs
```
python paper_rewriter_plus.py ingest \
  --inputs ./input_pdfs \
  --chroma_path ./ragdb \
  --collection papers \
  --embed_model BAAI/bge-large-en-v1.5 \
  --chunk_chars 1500 --chunk_overlap 200 \
  --embed_token_limit 480
```

### Query the index directly
```
python paper_rewriter_plus.py query \
  --chroma_path ./ragdb \
  --collection papers \
  --embed_model BAAI/bge-large-en-v1.5 \
  --k 8 \
  "your search text here"
```

### Process a document (with RAG)
```
python paper_rewriter_plus.py process \
  --input my_paper.pdf \
  --outdir outputs \
  --task rewrite_academic \
  --model gpt-4o-mini \
  --rag --chroma_path ./ragdb --collection papers \
  --k 6 --context_tokens 1200 \
  --stream   # optional
```

---

## Troubleshooting

**RuntimeError: The size of tensor a (<+512>) must match tensor b (512)**
- Cause: embedding model has a 512-token limit; a chunk exceeded it.
- Fix: token-aware ingest is already enabled; keep `--embed_token_limit` ≤512 (use the default 480 for BGE-large).

**Chroma telemetry warnings (`capture() takes 1 positional argument ...`)**
- Telemetry is disabled in code. If you still see noise, set:
  ```bash
  export CHROMADB_TELEMETRY_DISABLED=1
  ```

**Local endpoint ignores `model`**
- Supply the exact model name used by your server; some backends disregard or enforce the value.

**GPU out-of-memory during embeddings**
- Use a smaller embedding model (e.g., `bge-small-en-v1.5`) or run with CPU torch.

**Slow ingest**
- Reduce `--chunk_chars` or increase batch size in code (see `B = 256`).

---

## Tips for quality
- Keep your **RAG library** specific to your domain (papers you cite or build upon).
- Use `figure_suggestions` after `rewrite_academic` to plan visuals; iterate.
- Run `citation_scout` to generate search terms for missing references.

---

## Roadmap (optional)
- Config file (YAML) for repeatable runs.
- Pluggable retrieval filters (by source, year, keywords).
- Support for long-context embedding models (raise `--embed_token_limit`).

---

## License
MIT
