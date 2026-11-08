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
with the vocabulary built from the FULL job-posting corpus (so tech
jargon like "faiss," "pytorch," "redis" survives as "known" rather than
being "corrected" into something else) — the same generative mechanism
from the coursework, applied to a genuinely different, useful problem.

**What went wrong first, and the fix.** Run as-is (vocabulary from a
2k-posting sample, "correct any unknown word"), it changed 42 words on a
clean test resume and every change was wrong (Redis→Hedis,
OpenAI→Openam, Khoury→Hours). Assignment 3 assumed a short, typo-heavy
query; a resume is long, mostly clean, and full of legitimately unknown
proper nouns. Two changes, each driven by the observed failures:

1. Vocabulary from all ~124k postings (one ~15s streaming pass;
   14k → 160k words). This alone took false corrections 42 → 8.
2. Guard rails for the remaining patterns: leave URL/email spans
   untouched (dwhitaker→whitaker), skip ALL-CAPS acronyms (NSBM→SBM)
   and mixed-case brand names (OpenAI), skip words under 4 letters
   (Lua→La), and allow edit-distance-2 fixes only on words of 8+
   letters (Khoury→Hours). Result: 0 false corrections on the clean
   resume. Title-case words are still corrected, so a typo at the start
   of a bullet ("Devloped") is caught.

The original `min_confidence_margin` check was removed: it compared the
candidate against the unknown word's own probability, which is always 0,
so it could never block a correction.

**Measured effect** (`scripts/eval_query_correction.py`): inject random
non-word typos into eligible resume words, then compare the top-10
postings retrieved against the clean resume's top-10. Dummy test resume,
20k-posting index, sectioned retrieval (see next section), 5 seeds:

| typo rate | overlap, correction OFF | overlap, correction ON | typos restored exactly |
|---|---|---|---|
| 5%  | 0.86 | 0.96 | 77% |
| 10% | 0.72 | 0.92 | 82% |
| 20% | 0.60 | 0.78 | 76% |
| 30% | 0.62 | 0.84 | 76% |

Under the earlier single-vector retrieval the gain was smaller (10%:
0.74 → 0.84), because typos past the embedder's 512-token window had no
effect either way. Caveats: one resume, synthetic typos (real OCR damage
also merges/splits words, which edit-1/2 on single tokens can't fix),
and 5 seeds is noisy (the 30% row beating 20% is noise, not a trend).
Re-run on the full index and several resumes before quoting final
numbers.

## Why the resume is searched as several queries, not one

The embedder (bge-small) reads at most 512 tokens and silently drops the
rest. Embedding the whole resume as one query meant the 1,017-token test
resume was retrieved on its first 510 tokens only: its entire EXPERIENCE
and TECHNICAL SKILLS sections never reached retrieval, with no error.

Fix (`preprocess/resume_sections.py`, `retrieval/fusion.py`):

1. Split the resume on its section headings (known names in any case,
   or short ALL-CAPS lines; the first line is the name, never a
   heading). Each section is capped at 480 tokens measured with the
   embedder's own tokenizer (whitespace words undercount ~2x); an
   oversized section is cut between lines into evenly sized pieces
   with the heading repeated. No detectable headings -> fixed windows.
2. Query each section separately (per_section_k = 30 chunks).
3. Combine with reciprocal rank fusion, per POSTING:
   score = sum over sections of 1 / (60 + rank). Rank-based, so
   sections with different distance scales combine fairly; a posting
   several sections agree on beats one a single section matches
   closely; and results are distinct postings (each with its best
   chunk) rather than several chunks of one posting.

Why sections rather than fixed windows: each query vector then means one
thing ("what they built", "what tools they know"), and the UI can show
which resume sections matched each posting.

Measured (`scripts/eval_resume_coverage.py`, test resume, 20k postings):

- Coverage: single vector sees 510 / 1,015 tokens (50%); sectioned sees
  all of it, as 6 sections.
- Sensitivity, the direct test for truncation: replace the resume's
  TECHNICAL SKILLS with a nurse's skills. Single-vector top-10 overlap
  with the original: **1.00** (results identical; the section was never
  seen). Sectioned: **0.60**.
- Top-10 overlap between the two methods: 0.70.

Open questions: every section gets an equal vote, including the
contact/header block; and near-duplicate postings (one recruiter
posting the same text under several job_ids, e.g. six "Junior data
scientist/java programmer" variants) still crowd the results -- per-
posting grouping can't merge different job_ids. See posting boilerplate
below.

## Generating the gap analysis: what the local models actually do

Measured on the dummy resume with fixed retrieval (5 postings), repeated
runs, temperature 0.2. "Usable" = the response parsed into the format.

**Baseline** (original prompt, `MATCHED: - <req> -- evidenced by resume,
required by [N]`, llama3.2:3b): 2/5 runs usable. Failures were format
drift (`- None` bullets, MISSING lines without `[N]`) and once the
800-token output cap. The usable runs claimed 22-28 matches -- e.g.
GoLang, Struts, OpenShift, PL/SQL and a SAFe certification all
"evidenced by resume", none of which the resume mentions, with MISSING
"None". The format let the model assert evidence without showing any.

**Change 1 -- quote, then verify in code.** MATCHED lines must quote a
short exact phrase from the resume (`| resume: "<phrase>" | [N]`), and
`evidence_in_resume` checks the quote occurs in the resume (case,
punctuation and PDF line wraps normalized; whole words). Claims whose
quote isn't found go to `unverified` -- shown separately in the UI,
never counted as matches, never silently dropped. Verifiability comes
from code, not from trusting the model.

- First attempt: 0 verified matches in 8/8 runs. The 3B model "quoted"
  job-posting text ("Excellent written and verbal communication
  skills") as resume evidence -- it couldn't tell the two documents
  apart, which is the root of the over-claiming.
- Resume moved AFTER the excerpts, fenced (`<<<RESUME ... RESUME>>>`),
  plus one example line: real verified matches (Java <- "Languages:
  Java, Python", React <- "Web: React, Next.js, Vite").

**Change 2 -- reliability plumbing.** Explicit `num_ctx: 8192`
(Ollama's default 4,096 was 96% full, and overflow drops prompt text
silently); `done_reason == "length"` raised as `GenerationTruncatedError`;
one retry with a format reminder naming the actual problem; truncated
output salvaged (complete lines kept, `truncated` flag, UI warning)
instead of retried; `- none` accepted for an empty list.

With those, end to end:

| model | usable | avg time / resume | notes |
|---|---|---|---|
| llama3.2:3b | 7/8 (8/8 with salvage) | 32 s | ignores the 8-line cap; 5/8 runs looped to 40+ claims and were cut off before MISSING |
| llama3.1:8b | 5/8 | 197 s | all 3 failures hit the token cap; games the verifier with real-but-irrelevant quotes (OpenShift <- "Infra & DevOps: Linux", NLP/SAS/Tensorflow <- "Concepts: Distributed Systems") |

Takeaways: (1) 8B was not better here -- less reliable and 6x slower on
an M3 Pro GPU. (2) Prompt instructions about length and format are not
enforced by either model; salvage turns a crash into an answer whose
MISSING list may be empty. (3) Verification proves a quote is real, not
that it supports the claim.

**Change 3 (adopted) -- constrained decoding.** The hand-parsed text
format is gone. Ollama's `format` parameter takes a JSON Schema
(`gap_analysis.response_schema`) and compiles it into a grammar that
constrains generation token by token, so the model can only emit:
valid JSON; `matched` / `missing` lists of at most 8 items; an `excerpt`
integer between 1 and the number of retrieved chunks (an out-of-range
citation is impossible to generate); required fields present; strings
capped at 150 chars (stops runaway text). Truthfulness is still the
code's job: `resume_quote` is verified against the resume exactly as
before. The salvage path was removed (half a JSON object can't be
salvaged, and the caps make truncation unlikely); the one retry stays
as a safety net.

Why this is the right layer: prompts *request* a format; a grammar
*enforces* it. Every format failure we measured (missing `[N]`, `- None`
bullets, ignoring the 8-line limit, looping to 40+ claims, running into
the token cap) is a shape problem, and shape is exactly what a grammar
controls.

| llama3.2:3b, dummy resume | usable | first-attempt | avg time | items |
|---|---|---|---|---|
| text format + verification + retry | 7/8 | 4/8 | 32 s | 3-47 matched, 0-37 missing |
| JSON schema (constrained) | **8/8** | **8/8** | **12 s** | 6-8 matched, 8 missing |

Sample constrained run: matched React, Java, Spring Boot, Linux,
Docker, Git, AWS, each quoting the real resume line; the verifier moved
one claim ("Full Stack Java Developer", "evidenced" by a quote copied
from the job posting) to unverified.

llama3.1:8b with the schema: 4/4 usable on the first attempt, 29 s avg
(was 5/8 and 197 s with the text format). But it still games the
verifier with real-but-irrelevant quotes: "Oracle DB" and "PostgreSQL DB"
both "evidenced" by "Backend: Spring Boot, Node.js, Express" (the
resume's PostgreSQL is further along that same line, so even the right
answer had the wrong quote). On this resume, 3B's matches were the more
accurate of the two, at under half the latency -- which also favours
the CPU-only portfolio deployment. Not yet measured: 3B on a CPU-only
host, and either model across several resumes.

What constrained decoding does NOT fix -- reasoning quality: MISSING
can list requirements the resume partly covers ("C++ and GoLang" when
the resume has C++; "Cloud and AWS Administration" alongside an AWS
match), and a real quote can still be irrelevant. Candidates for a
code-side check: flag a MISSING item whose key terms appear verbatim in
the resume.

## Chunking job postings: sentences, boilerplate, duplicates

Reading real chunks (as this doc said to) turned up three problems.
Measured on the whole corpus first, so the fixes target real patterns:

- The old chunker split on whitespace and re-joined with spaces,
  throwing away the line breaks that 95.7% of descriptions have, so
  bullets ran together and chunks were cut mid-sentence. 53% of
  descriptions also have sentences glued with no space ("oil
  baths.Use standard").
- 12.9% of postings (16,015) are exact copies of an earlier posting's
  description -- one recruiter reposting the same text. Six of them
  ("Junior data scientist/java programmer" variants) were in the test
  resume's top 10.
- Legal and template text (EEO statements, accommodation notices,
  benefits blurbs, "this job description is subject to change") was in
  26.9% of chunks. It embeds close to everything and wins retrieval
  slots from real requirements.

Fixes (`preprocess/chunker.py`, `preprocess/boilerplate.py`,
`ingest/job_postings_loader.py`, `retrieval/fusion.py`):

1. **Sentence-aware chunking.** Split on line breaks, bullets and
   sentence ends (re-inserting the missing space in glued sentences);
   pack whole sentences into chunks of <= 300 words; overlap is whole
   trailing sentences (~50 words); a short tail is folded into the
   previous chunk; a skills_desc under 40 words ("Azure") is appended
   to the description instead of becoming its own 1-word chunk.
   Packing tracks sentence positions, not text, so a repeated sentence
   is never mistaken for overlap and dropped (an early version had
   exactly that bug).
2. **Boilerplate removal, two rules.** (a) Frequency, learned from the
   de-duplicated corpus: a sentence in >= 20 postings from >= 10
   *different companies*. Cross-company is the point: a sentence
   repeated within one company is usually its real duties and is
   kept. (b) A short legal-language pattern (EEO, non-discrimination,
   accommodation, E-Verify) for statements that vary by company name.
   Both rules exempt sentences that state a requirement (degree,
   experience, "must be able to", lifting/weight). That exemption list
   grew from reading the detected list: the first run flagged "The
   employee must occasionally lift and/or move up to 25 pounds" and a
   standard supervisory-duties sentence -- real requirements, now kept.
   The detected list is written to
   `data/processed/boilerplate_sentences.json` (402 sentences) so
   exactly what was removed can be inspected.
3. **Exact-duplicate postings are skipped** at index time (first
   job_id kept for citations).
4. **At most 2 postings per company** in retrieval results. Partial
   copies survive exact de-duplication: one staffing agency (CodeMonk)
   had five ~55%-identical "Senior Software Engineer" postings; without
   the cap it took 6 of the test resume's top 10, with it 2. Company
   names are now in chunk metadata, the prompt and the UI ("Senior
   Software Engineer at CodeMonk").

Measured on the 20k dev index (same `--limit`; "after" is the first 20k
*distinct* postings):

| | before | after |
|---|---|---|
| chunks containing EEO/accommodation text | 26.9% | **3.0%** |
| chunks ending at a sentence boundary | 25.9% | **66.6%** |
| chunks under 40 words | 390 | **221** |
| chunks whose opening repeats 10+ times | 4,788 | 2,970 |
| chunks per posting | 2.39 | 2.17 |

Retrieval for the test resume: the duplicate "Junior data scientist"
postings are gone; new in the top 10 are "Mobile Developer - React
Native" (matched by EXPERIENCE, PROJECTS and SKILLS together) and
"React Native Developer" -- React Native being the resume's main mobile
stack. The skills-swap sensitivity check is unchanged (0.60).

What's left, and why: the remaining short chunks are postings whose
entire description is short (nothing to merge into). "Ending at a
sentence boundary" is 66.6%, not 100%, mostly because bullet lines
often have no closing punctuation, not because of mid-sentence cuts.
Residual EEO text is mostly inside long run-on lines that also contain
requirements, which the exemption deliberately keeps. One company's own
"About us" paragraph repeated across its postings is still indexed --
the frequency rule can't tell it from real duties.

## Dataset

**LinkedIn Job Postings (2023-2024)**, Kaggle, uploaded by `arshkon`
(scraper: [ArshKA/LinkedIn-Job-Scraper](https://github.com/ArshKA/LinkedIn-Job-Scraper)),
with 100,000 additional postings contributed by `zoeyyuzou`. 123,849
rows in `postings.csv` (version 13, pinned), plus linked `companies/companies.csv`,
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
- **Resolved**: query correction checked against a real resume and
  reworked (see "The query-correction tie-in" above): full-corpus
  vocabulary, guard rails, 0 false corrections on the clean test resume,
  corrections shown in the UI with an on/off toggle, measured by
  `scripts/eval_query_correction.py`.
- **Resolved — format drift / output length**: constrained decoding
  (JSON schema), see "Generating the gap analysis" above. 8/8 runs
  usable on the first attempt with llama3.2:3b.
- **Partly resolved — over-claiming**: matches must quote the resume
  and the quote is verified in code. Open: MISSING items the resume
  partly covers, and real-but-irrelevant quotes (reasoning quality).
- **Resolved — resume truncation**: the resume is now searched as
  token-bounded sections fused with RRF (see "Why the resume is searched
  as several queries" above).
- **Resolved — chunking / boilerplate / duplicates**: see "Chunking job
  postings" above. EEO text 26.9% -> 3.0% of chunks; duplicate postings
  skipped; at most 2 postings per company in results. Open: one
  company's repeated "About us" text is still indexed.
- **Resolved**: company names are in chunk metadata, the LLM prompt and
  the UI (straight from postings.csv's company_name column; the separate
  companies.csv join was removed as unnecessary).
- Firewall/networking setup for the GCP endpoint (opening port 8501
  publicly) isn't scripted yet — needs to happen during Phase 4 and be
  tested well before Dec 6, not discovered for the first time on
  presentation day.
