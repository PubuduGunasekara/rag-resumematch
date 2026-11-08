# ResumeMatch

A RAG system that retrieves relevant job postings from a 124,000+ posting
corpus against an uploaded resume, and produces a cited gap analysis —
built for CS6120 (NLP), Fall 2026, Northeastern University.

## What this is (and isn't)

- **Is**: a from-scratch RAG pipeline (chunking, embedding, retrieval,
  grounded generation) over a real, citable dataset, with a
  query-correction preprocessing step adapted from Assignment 3's
  edit-distance autocorrect, a required Streamlit front end, Docker
  containerization, and a GCP deployment path.
- **Isn't**: a LangChain/LlamaIndex wrapper — every retrieval step is
  plain Python so the mechanics are visible and explainable in a defense.
- **Isn't**: perpetually hosted on GPU infra. The course demo runs on a
  GCP VM torn down after presentation day; a cheap/free CPU-only
  redeploy is the plan for keeping this as a resume portfolio piece
  afterward (see `docs/design.md`).

## Direction (read this before the code — it's the one thing that
flipped partway through design)

**The resume is the query. The job-postings corpus is what's indexed.**
Upload a resume → retrieve the most relevant postings from the corpus →
generate a per-posting gap analysis (matched / missing requirements),
every claim citable back to a specific posting via `job_id` /
`job_posting_url`.

## Pipeline

```
postings.csv (123,849 rows, Kaggle v13)
        │  scripts/build_index.py (one-time batch job)
        ▼
  chunk ─► embed ─► ChromaDB (persisted, with citation metadata)


resume.pdf ─► parse ─► query_correction ─► embed ─► retrieve top-k postings
                                                            │
                                                            ▼
                                     resume + retrieved postings ─► local LLM (Ollama)
                                                            │
                                                            ▼
                                        cited gap analysis (Streamlit UI)
```

## Stack

| Concern          | Choice                                 | Why |
|-------------------|-----------------------------------------|-----|
| Dataset           | LinkedIn Job Postings 2023-2024 (Kaggle, `arshkon`, 124k+ rows, CC BY-SA 4.0) | Verified row count and license; `job_posting_url` field gives real, clickable citations |
| Dataset download  | `kagglehub`                            | Official API, requires a Kaggle token at runtime (not baked into the image) |
| Resume parsing     | `pypdf`, `python-docx`                 | The two realistic resume formats |
| Embeddings         | `sentence-transformers` (bge-small)    | Free, local, good quality/size tradeoff, runs fine on CPU |
| Vector store       | `chromadb`                             | Local persistence + metadata filtering, no server to run |
| Query correction   | Custom, from Assignment 3              | Reuses verified edit-distance/frequency logic against OCR/typo artifacts in resume text |
| Generation         | Ollama (local LLM)                     | `llama3.2:3b` for local/CPU dev, `llama3.1:8b` on the GCP GPU profile |
| Front end          | Streamlit                              | Required by the project spec |
| Containerization   | Docker + docker-compose (two profiles) | Required by the project spec; separate `local` (CPU) and `gcp` (GPU) compose files share one codebase |
| Tests              | `pytest`                               | |
| Lint/format        | `ruff`, `black`                        | One command each, no enforced CI gate |

## Setup

```bash
cp .env.example .env   # fill in KAGGLE_USERNAME / KAGGLE_KEY (see comments in the file)
```

### Local (CPU, laptop)

```bash
docker-compose -f docker-compose.local.yml up --build
```
First run downloads the dataset and builds the index automatically
(`entrypoint.sh`) — this is slow once, then cached in a Docker volume.
For fast dev iteration, build a small index directly instead:
```bash
python scripts/build_index.py --limit 2000
streamlit run src/resumematch/frontend/streamlit_app.py
```

### GCP (GPU, course demo)

```bash
source .env  # or export GCP_PROJECT_ID etc.
./gcp/start.sh
# SSH in once: install nvidia-container-toolkit + docker-compose, clone this repo
# build the index LOCALLY first (free), then:
#   gcloud compute scp --recurse ./chroma_db <vm-name>:~/resumematch/chroma_db --zone=<zone>
docker-compose -f docker-compose.gcp.yml up --build -d
# ... when done:
./gcp/stop.sh
```

## Project layout

```
src/resumematch/
├── ingest/            parse_resume.py, job_postings_loader.py (Kaggle download + batch iteration)
├── preprocess/        chunker.py (job postings), query_correction.py (resume text, from Assignment 3)
├── retrieval/         embedder.py, vector_store.py
├── generation/        ollama_client.py, gap_analysis.py (prompt + citation-resolved parsing)
├── frontend/          streamlit_app.py — the required UI
└── pipeline.py        query-time orchestration (assumes the index is already built)

scripts/build_index.py  one-time batch job: download corpus -> chunk -> embed -> index
tests/                  unit tests per module
docs/design.md          architecture decisions + dataset citation, written for the grader
gcp/start.sh, stop.sh   budget guardrail for the $50 GCP credit — use every time
docker-compose.local.yml / docker-compose.gcp.yml   same code, CPU vs GPU profile
data/raw/               downloaded dataset CSVs (gitignored)
data/processed/         cached intermediate output (gitignored)
```

## Dataset citation

LinkedIn Job Postings (2023–2024), scraped by [ArshKA](https://github.com/ArshKA/LinkedIn-Job-Scraper),
with an additional 100,000 postings contributed by `zoeyyuzou`.
[Kaggle dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings),
licensed [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
Any processed/derivative form of this data distributed from this project
remains under the same license.

## Status

Scaffold only — modules contain signatures/docstrings and
`raise NotImplementedError` placeholders, not implementations. See
`docs/design.md` for what's built vs. planned, and for the course
timeline this is being built against.
