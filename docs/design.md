# Design Decisions

## Scope and context

Solo-built for CS6120 (NLP), Fall 2026. Proposal due Nov 16, final project
due Dec 7. Team may grow to 2-3 later; module boundaries are kept clean
specifically so contributions stay attributable by git history even
though one person wrote the initial version.

## Why the resume is the query, not the indexed document

The project spec requires "a database that is no less than 10k entries"
and "clickable citation to the data source (article and passage)." A
single resume and a single job posting can't satisfy either requirement
— there's nothing to scale and nothing to cite into. Reframing the job-
postings corpus (124,000+ rows) as the indexed database, with the
resume as the query, satisfies both: the corpus is large and real, and
every retrieved result carries a `job_id` / `job_posting_url` that
resolves to an actual LinkedIn posting.

## Why no LangChain/LlamaIndex

A RAG framework optimizes for not having to understand chunking,
embedding, and retrieval — the opposite of what an NLP course rubric
rewards. Writing these ~150-200 lines by hand means every design choice
is explainable in a defense, not "whatever the framework defaults to."

## Why local-only LLM (Ollama), not a paid API

- No dependency on a key that could be revoked or rate-limited before
  the deadline.
- The spec explicitly requires "LLMs entirely local (i.e., on GCP or
  metal) / native" — this isn't a preference, it's a requirement.
- Reproducible for a grader without needing personal API credentials.

## Why ChromaDB over raw FAISS

FAISS is index-only — no persistence, no metadata filtering, build that
yourself. ChromaDB gives local file-based persistence and metadata
filtering (needed for citations: job_id, section, url, title) for free,
with no server process to run. At this data scale (chunks from 124k
postings, not millions), FAISS's raw speed advantage doesn't matter.

## The query-correction tie-in to Assignment 3

Resume text extracted from a PDF frequently has OCR/extraction
artifacts (dropped characters, run-together words). A single corrupted
token can hurt embedding-similarity retrieval even though a human
reader wouldn't notice it. Assignment 3's edit-distance substitution
logic (`probable_substitutes`/`min_edit_distance`) is reused directly,
with the vocabulary built from a SAMPLE of job-posting text (so tech
jargon like "k8s," "faiss," "pytorch" survives as "known" rather than
being "corrected" into something else) — the same generative mechanism
from the coursework, applied to a genuinely different, useful problem.

## Dataset

**LinkedIn Job Postings (2023-2024)**, Kaggle, uploaded by `arshkon`
(scraper: [ArshKA/LinkedIn-Job-Scraper](https://github.com/ArshKA/LinkedIn-Job-Scraper)),
with 100,000 additional postings contributed by `zoeyyuzou`. 124,000+
rows in `job_postings.csv`, plus linked `companies.csv`,
`benefits.csv`, `employee_counts.csv`. License: **CC BY-SA 4.0** —
attribution required; any derivative/processed form distributed from
this project must remain under the same license.

Why this dataset specifically: `job_posting_url` makes citations verify
against the live source, not just an opaque internal ID — directly
targets the rubric's "Verifiability" criterion. `description` and
`skills_desc` give two independently citable text fields per posting.

Downloaded via `kagglehub` at container-run time (not baked into the
image), satisfying the spec's "AutomatedDownload" requirement.
Requires `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars — documented in
README, not committed (see `.gitignore`).

## Deployment: two profiles, one codebase, and the budget math

- **Course demo** (`docker-compose.gcp.yml`): GCP VM, T4 GPU, $50
  course credit. At ~$0.50-0.55/hr, that's ~90 hours total — the index
  is built LOCALLY (CPU, free) and `scp`'d to the VM rather than
  rebuilt there, so the VM only ever pays for LLM inference, not
  embedding 124k rows. `gcp/start.sh`/`gcp/stop.sh` exist specifically
  so stopping the VM is a one-command habit, not something to remember
  at 1am.
- **Portfolio demo** (post-course, `docker-compose.local.yml` as the
  base): a persistent GPU VM costs ~$350+/month — not viable
  indefinitely. The plan is to redeploy to a cheap/free CPU-only host
  with a smaller quantized model after the course ends, not compete
  with course deadlines for that work now.

## Known risks / open questions (update as the project progresses)

- **Resolved**: all modules are now implemented (chunker, query
  correction, embedder, vector store, resume parsing, Ollama client,
  gap-analysis parsing). Unit tests pass for everything that doesn't
  require the real dataset, a local embedding model, or a running
  Ollama instance (see "What's verified vs. not yet" below).
- Chunk-boundary quality still hasn't been verified against *real*
  posting text (only the hand-written fixtures in
  `tests/test_chunker.py`). **Action**: once `build_index.py` runs
  against the real dataset, print and read actual chunks from 5-10
  real postings before trusting them at scale.
- The domain vocabulary for query correction is now built and
  persisted by `build_index.py` (`data/processed/domain_vocab.json`,
  sampled from the first ~3,000 indexed postings), but hasn't been
  checked against a real resume yet — same OOV-threshold risk as
  Assignment 4's StyleGram applies until that's done.
- `parse_response` in `gap_analysis.py` now raises `ValueError` (with
  the raw response attached) on a missing header, a missing `[N]`
  citation, or an out-of-range `[N]` — it never silently returns an
  empty gap analysis. What's NOT yet built: a retry-with-format-
  reminder fallback when a local model drifts from the format. Worth
  adding once you see how often `llama3.2:3b`/`llama3.1:8b` actually
  drift in practice — don't build the retry logic speculatively.
- `company_id` -> company name (via `companies.csv`) is loaded in
  `job_postings_loader.load_company_names` but not yet wired into
  chunk metadata or the Streamlit display — nice-to-have for citation
  readability ("Posting at Acme Corp" vs. a bare job_id), not required.
- Firewall/networking setup for the GCP endpoint (opening port 8501
  publicly) isn't scripted yet — needs to happen during Phase 4 and be
  tested well before Dec 6, not discovered for the first time on
  presentation day.
