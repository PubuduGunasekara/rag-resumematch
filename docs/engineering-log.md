# Engineering log

Every failure found, every experiment run, and what was decided. Numbers are measured, never estimated,
and each entry names the script or test that produced them. Dates are the commit dates of the work.
Rejected ideas stay in the log: they are part of how the system got here.

Add an entry whenever a change is measured. Keep the wording plain: no em dashes.

| # | Date | Area | Entry | Outcome |
|---|---|---|---|---|
| 1 | 2026-11-07 | Query correction | [Build the spelling vocabulary from all postings, not a sample](#vocab-full-corpus) | Adopted |
| 2 | 2026-11-07 | Query correction | [Does typo correction actually improve retrieval?](#correction-helps-retrieval) | Adopted |
| 3 | 2026-11-07 | Retrieval | [The embedding model silently read half the resume](#resume-truncation) | Adopted |
| 4 | 2026-11-08 | Generation | [Free-text answers broke the parser](#json-constrained-decoding) | Adopted |
| 5 | 2026-11-08 | Generation | [The model claimed matches the resume does not support](#two-sided-grounding) | Adopted |
| 6 | 2026-11-08 | Indexing | [A quarter of indexed text was legal boilerplate](#chunk-cleanup) | Adopted |
| 7 | 2026-11-08 | Deployment | [Docker on a Mac cannot use the GPU](#docker-metal) | Adopted |
| 8 | 2026-11-08 | Evaluation | [Testing beyond software resumes found four bugs](#four-resumes) | Adopted |
| 9 | 2026-11-08 | Retrieval | [The skills-swap check did not generalize](#skills-swap-outvoted) | Open |
| 10 | 2026-11-08 | Generation | [Telling the model not to list benefits made it worse](#benefits-prompt-ab) | Rejected |
| 11 | 2026-11-09 | Generation | [One LLM call per job was slower and noisier](#per-job-calls) | Rejected |
| 12 | 2026-11-09 | Results page | [A share-of-requirements score called a strong fit "not a match"](#ratio-tiers) | Adopted |
| 13 | 2026-11-09 | Generation | [Asking for requirements from every posting](#coverage-ab) | Adopted |
| 14 | 2026-11-09 | Input validation | [Wrong files went straight into the pipeline](#resume-check) | Adopted |
| 15 | 2026-11-09 | Generation | [The client defaulted to a model nobody had pulled](#model-default) | Adopted |
| 16 | 2026-11-09 | Privacy | [ChromaDB telemetry was on, and a version clash logged errors](#telemetry-off) | Adopted |
| 17 | 2026-11-09 | Generation | [The model wrote a summary nobody saw](#unused-summary) | Adopted |
| 18 | 2026-11-09 | Retrieval | [The vector search missed a fifth of the nearest chunks](#search-recall) | Adopted |
| 19 | 2026-11-09 | Privacy | [Error logs could have included resume text](#log-no-message) | Adopted |
| 20 | 2026-11-09 | Evaluation | [Stage-by-stage timing of the current pipeline](#benchmark) | Adopted |
| 21 | 2026-11-09 | Parsing | [Word tables were dropped from resumes](#docx-tables) | Adopted |
| 22 | 2026-11-09 | Parsing | [Skill labels in a PDF sidebar became sections](#sidebar-headings) | Adopted |
| 23 | 2026-11-09 | Retrieval | [Every shown claim traced back to the corpus](#grounding-audit) | Adopted |
| 24 | 2026-11-10 | Deployment | [A sample index so nobody needs a Kaggle account](#sample-index) | Adopted |

<a id="vocab-full-corpus"></a>
## 1. Build the spelling vocabulary from all postings, not a sample

**Query correction, 2026-11-07. Outcome: adopted.**

**Problem.** The corrector, an edit-distance autocorrect I had written earlier, rewrote real words on a clean resume (Redis, Vite, OpenAI, common first names) because a 2,000-posting vocabulary had never seen them.

**Hypothesis.** A vocabulary from the full corpus knows those words, so it stops correcting them.

**Method.** Count false corrections on the clean test resume with each vocabulary.

**Results.**

| Variant | False corrections |
|---|---|
| 2k-posting vocabulary (~14k words) | 42 |
| Full-corpus vocabulary (159,808 words) | 8 |
| Full corpus plus guard rails (length, acronyms, mixed case, URLs) | 0 |

**Decision.** Full-corpus vocabulary, built in one ~15 s streaming pass, plus guard rails.

**Evidence.** `scripts/eval_query_correction.py`, `tests/test_query_correction.py`

<a id="correction-helps-retrieval"></a>
## 2. Does typo correction actually improve retrieval?

**Query correction, 2026-11-07. Outcome: adopted.**

**Problem.** Correction is only worth having if it changes which postings come back.

**Hypothesis.** With typos injected, retrieval with correction stays closer to the clean result.

**Method.** Inject non-word typos into 5 to 20% of eligible words (5 seeds), retrieve the top 10 with correction off and on, and measure overlap with the clean resume's top 10.

**Results.**

| Variant | Overlap |
|---|---|
| 10% typos, correction off | 0.8 |
| 10% typos, correction on | 0.98 |
| 20% typos, correction off | 0.7 |
| 20% typos, correction on | 0.84 |

**Decision.** Keep correction on by default, with a toggle and a table of every change in the UI.

**Evidence.** `scripts/eval_query_correction.py`

<a id="resume-truncation"></a>
## 3. The embedding model silently read half the resume

**Retrieval, 2026-11-07. Outcome: adopted.**

**Problem.** bge-small reads at most 512 tokens and drops the rest without an error. The 1,015-token test resume lost its experience and skills sections, the most job-relevant text.

**Hypothesis.** Searching each resume section separately and fusing the results sees all of it.

**Method.** Count the tokens each method can see; swap the last section for another profession's skills and check whether results move.

**Results.**

| Variant | Tokens seen | Skills swap overlap |
|---|---|---|
| Whole resume as one query | 510 of 1,015 (50%) | 1.0 |
| Sections plus reciprocal rank fusion | 1,015 of 1,015 (100%) | 0.3 |

**Decision.** Token-bounded sections (480 tokens), fused per posting with RRF (k = 60).

**Evidence.** `scripts/eval_resume_coverage.py`, `tests/test_resume_sections.py`, `tests/test_fusion.py`

<a id="json-constrained-decoding"></a>
## 4. Free-text answers broke the parser

**Generation, 2026-11-08. Outcome: adopted.**

**Problem.** The local model's answers drifted in format and were sometimes cut off at the token limit, so citations could not be parsed reliably.

**Hypothesis.** Constraining decoding to a JSON schema (Ollama's format option) makes every answer parseable.

**Method.** Repeated runs on the test resume with llama3.2:3b, counting answers usable on the first attempt.

**Results.**

| Variant | Usable first try |
|---|---|
| Free text with a format instruction | unreliable |
| JSON schema: max 8 items per list, excerpt number bounded by retrieved count | 8 of 8 |

**Decision.** JSON schema plus one retry with a format reminder. An out-of-range citation cannot be generated.

**Evidence.** `tests/test_gap_analysis.py`

<a id="two-sided-grounding"></a>
## 5. The model claimed matches the resume does not support

**Generation, 2026-11-08. Outcome: adopted.**

**Problem.** A 3B model quoted job-posting text as resume evidence, matched "Java" to a posting that never mentions Java, and listed AWS as missing for a resume that lists AWS.

**Hypothesis.** Checking every claim in code from both sides removes unsupported claims without hiding real ones.

**Method.** Each claim must cite a posting that mentions it; a match must quote the resume exactly and the quote must mention the requirement; a gap must not be plainly on the resume. Wrong citations are moved to the posting that does mention the requirement. Repeated runs on the test resume.

**Results.**

| Variant | Matched | Gaps | Format failures |
|---|---|---|---|
| Verified per run (6 runs) | 4.5 | 3.5 | 0 |

**Decision.** Every claim is checked; failures are shown separately with the reason, never counted.

**Evidence.** `tests/test_gap_analysis.py`

<a id="chunk-cleanup"></a>
## 6. A quarter of indexed text was legal boilerplate

**Indexing, 2026-11-08. Outcome: adopted.**

**Problem.** Equal-opportunity and accommodation text filled 26.9% of chunks, and chunks were cut mid-sentence.

**Hypothesis.** Sentences repeated across many companies are boilerplate; packing whole sentences keeps chunks readable; exact duplicate postings add nothing.

**Method.** Stats over the full index before and after.

**Results.**

| Variant | Before | After |
|---|---|---|
| Chunks with EEO text | 26.9% | 2.9% |
| Chunks ending at a sentence boundary | 25.9% | 68.4% |
| Exact duplicate postings skipped | 0 | 16,314 |

**Decision.** Boilerplate = a sentence in at least 20 postings from at least 10 companies, or legal language, with an exemption for requirement sentences (lifting limits, degrees). Sentence-aware packing.

**Evidence.** `tests/test_boilerplate.py`, `tests/test_chunker.py`

<a id="docker-metal"></a>
## 7. Docker on a Mac cannot use the GPU

**Deployment, 2026-11-08. Outcome: adopted.**

**Problem.** Ollama inside Docker runs on CPU on macOS.

**Hypothesis.** Running the app in Docker and Ollama natively keeps the container setup and the GPU speed.

**Method.** Time one resume end to end, warm.

**Results.**

| Variant | Seconds |
|---|---|
| Everything in Docker (CPU) | 37 |
| App in Docker, Ollama native (Metal) | 11 |

**Decision.** The laptop demo uses native Ollama; a GPU VM is the later path to a public demo.

**Evidence.** `docker-compose.local.yml`, `README.md`

<a id="four-resumes"></a>
## 8. Testing beyond software resumes found four bugs

**Evaluation, 2026-11-08. Outcome: adopted.**

**Problem.** Every number so far came from one software resume.

**Hypothesis.** Resumes from other fields will expose assumptions.

**Method.** Synthetic nurse, accountant and mechanical engineer resumes through every eval script and 4 gap-analysis runs each.

**Results.**

| Variant | Before | After |
|---|---|---|
| Title Case headings (Work History) | whole resume became one query | sections detected |
| Company name on a clean resume | Midstate corrected to Mistake | 0 false corrections on all 4 resumes |
| Benefits listed as missing skills | 403(b) plan, Pension, Dental | rejected in code |
| Plural mismatch (Certifications vs certification) | false gap | matched |

**Decision.** Fixed all four, each with a regression test.

**Evidence.** `eval/resumes/`, `tests/test_resume_sections.py`, `tests/test_query_correction.py`, `tests/test_gap_analysis.py`

<a id="skills-swap-outvoted"></a>
## 9. The skills-swap check did not generalize

**Retrieval, 2026-11-08. Outcome: open.**

**Problem.** On short resumes, swapping the skills section for another profession's left the results unchanged (overlap 1.00), which looked like the section was invisible again.

**Hypothesis.** The section is read but outvoted, because fusion rewards postings several sections agree on.

**Method.** Same swap test on the nurse and accountant resumes, compared with the single-vector method.

**Results.**

| Variant | Skills swap overlap |
|---|---|
| Sections plus RRF | 1.0 |
| Single vector | 0.6 |

**Decision.** Documented as a property, not fixed. Good for coherent resumes; a career changer's results will follow their experience more than their new skills.

**Evidence.** `scripts/eval_resume_coverage.py`

<a id="benefits-prompt-ab"></a>
## 10. Telling the model not to list benefits made it worse

**Generation, 2026-11-08. Outcome: rejected.**

**Problem.** The model listed employer benefits as missing skills.

**Hypothesis.** A prompt instruction ("never list what the employer offers") stops it.

**Method.** A/B on the same retrieved postings, 6 runs per arm, two resumes.

**Results.**

| Variant | Verified gaps per run | Benefit claims |
|---|---|---|
| Software resume, without the line | 2.8 | 6 |
| Software resume, with the line | 0.2 | 6 |
| Accountant, without the line | 4.3 |  |
| Accountant, with the line | 2.0 |  |

**Decision.** Rejected. With a 3B model, extra negative instructions read as "list less" and did not reduce benefit claims at all. Benefits are filtered in code instead.

**Evidence.** `src/resumematch/generation/gap_analysis.py`

<a id="per-job-calls"></a>
## 11. One LLM call per job was slower and noisier

**Generation, 2026-11-09. Outcome: rejected.**

**Problem.** The results page needs a checklist for each job, not one list for all five.

**Hypothesis.** Analyzing each posting in its own call, with its full text, gives better per-job checklists.

**Method.** Nurse and engineer resumes, 5 per-job calls vs one pooled call over filtered requirement text.

**Results.**

| Variant | Seconds | Rejected per job | Notes |
|---|---|---|---|
| 5 per-job calls | 60 to 68 | 5 to 14 | company blurbs as requirements, unstable between runs |
| 1 pooled call, claims grouped by cited posting | 16 to 21 | 0 to 5 | NP license gap found reliably |

**Decision.** Rejected per-job calls. One pooled call; each claim already cites a posting, so grouping by posting gives per-job checklists for free.

**Evidence.** `src/resumematch/generation/job_report.py`, `tests/test_job_report.py`

<a id="ratio-tiers"></a>
## 12. A share-of-requirements score called a strong fit "not a match"

**Results page, 2026-11-09. Outcome: adopted.**

**Problem.** Tiering jobs by the share of requirements met labelled an RN ICU posting "not a match" (1 of 9) for a nurse with six years of ICU experience. The model listed granular items ("experience in renal failure") that a resume never names word for word.

**Hypothesis.** A word check can confirm a requirement is on a resume but cannot prove it is absent, so a ratio of "met" is not a fair verdict.

**Method.** Inspect tiers on the nurse and engineer resumes under both rules.

**Results.**

| Variant | Rn icu posting | Stroke np posting |
|---|---|---|
| Tier by share met | not a match |  |
| Tier by retrieval rank, blocked only by a missing required credential | worth a look | needs an NP credential |

**Decision.** Tiers come from retrieval similarity plus a hard-credential gate. Checklists show "Shown", "Partly shown" and "Not found on your resume", never "missing".

**Evidence.** `tests/test_job_report.py`

<a id="coverage-ab"></a>
## 13. Asking for requirements from every posting

**Generation, 2026-11-09. Outcome: adopted.**

**Problem.** With one call over five postings, most checked items came from one or two of them.

**Hypothesis.** A higher item limit plus "include requirements from every excerpt" spreads coverage.

**Method.** A/B, 3 runs per arm, three resumes. A = 8 items per list, no line. B = 12 items plus the line.

**Results.**

| Variant | Jobs with items | Verified items | Seconds |
|---|---|---|---|
| Nurse A / B | 1.7 / 3.3 | 2.3 / 7.7 | 14 / 18 |
| Accountant A / B | 2.7 / 4.0 | 7.3 / 14.3 | 12 / 18 |
| Engineer A / B | 2.7 / 2.0 | 8.3 / 7.7 | 18 / 22 |

**Decision.** Adopted B: coverage doubles on two resumes and is even on the third (3 runs, within noise), for about 5 s more. A positive instruction helped where a negative one hurt.

**Evidence.** `src/resumematch/generation/gap_analysis.py`

<a id="resume-check"></a>
## 14. Wrong files went straight into the pipeline

**Input validation, 2026-11-09. Outcome: adopted.**

**Problem.** Only the file extension and size were checked. A text file renamed to .pdf, a broken
.docx and a password-protected PDF crashed the parser (a 500 in the API, a Python traceback in
Streamlit). A scanned PDF parsed as 0 words and kept going. A banana bread recipe ran the whole
pipeline and would have received a confident "closest match" report.

**Hypothesis.** Similarity to postings would separate resumes from other documents.

**Method.** Measured 4 resumes and 5 other documents on retrieval similarity (mean over the top 5
postings) and on structural signals: recognized section headings, contact details, dated roles,
job-ad phrases.

**Results.**

| Document | Headings | Contact | Dated roles | Similarity |
|---|---|---|---|---|
| Resumes (4) | 2 to 5 | yes | 2 to 5 | 0.556 to 0.687 |
| Plain text, recipe, news article | 0 | no | 0 | 0.241 to 0.277 |
| Cover letter | 0 | yes | 0 | 0.710 |
| Job posting | 0 | no | 0 | 0.806 |

**Decision.** The hypothesis was wrong for the hardest cases: a cover letter and a job posting are
MORE similar to postings than any real resume, so similarity cannot be the guard. Validation runs
before any model call, in four layers: the bytes match the extension; the file opens (damaged and
password-protected files get their own messages); there is text (scans are told to export a text
PDF); and at least two of three resume signals are present (headings, contact details, dated roles).
The last layer is a soft warning with "Check it anyway", because a heuristic must not block an
unusual real resume. Job postings get their own message. If an analysis still confirms nothing on
the document, the report says "We couldn't connect this document to job requirements" instead of
naming a closest match. Tested in the browser on both front ends: all 4 resumes pass; every other
file gets the right message.

**Evidence.** `src/resumematch/ingest/validate.py`, `tests/test_validate.py`, `tests/test_api.py`, `tests/test_job_report.py`

<a id="model-default"></a>
## 15. The client defaulted to a model nobody had pulled

**Generation, 2026-11-09. Outcome: adopted.**

**Problem.** `OllamaClient` used `llama3.1:8b` when `OLLAMA_MODEL` was not set, while the README,
the Docker setup and every measurement use `llama3.2:3b`. A run without that variable asked Ollama
for a model the setup steps never pulled. `.env.example` had the same stale default.

**Hypothesis.** The default should be the model the documentation tells people to pull.

**Method.** A unit test for the default with `OLLAMA_MODEL` unset.

**Results.**

| Variant | Default model |
|---|---|
| Before | llama3.1:8b |
| After | llama3.2:3b |

**Decision.** Default changed in the client and in `.env.example`; the environment still overrides it.

**Evidence.** `tests/test_ollama_client.py`

<a id="telemetry-off"></a>
## 16. ChromaDB telemetry was on, and a version clash logged errors

**Privacy, 2026-11-09. Outcome: adopted.**

**Problem.** ChromaDB sends anonymized usage events by default. Every start also printed "Failed to
send telemetry event": chromadb 0.5.23 calls `posthog.capture()` with the signature from before
posthog 6, and the installed posthog 7.86 raised before checking whether telemetry was disabled.
No event was sent, but only because of the clash.

**Hypothesis.** Turning telemetry off in the client settings and pinning a compatible posthog removes
both the events and the errors.

**Method.** Unit test on the client settings; count the error lines when opening the index.

**Results.**

| Variant | Telemetry setting | Error lines on start |
|---|---|---|
| Before | on (default) | 2 per start |
| Settings off, posthog 7.86 | off | 2 per start |
| Settings off, posthog 3.7.0 | off | 0 |

**Decision.** `anonymized_telemetry=False` in `retrieval/vector_store.py` and `posthog==3.7.0` pinned.

**Evidence.** `tests/test_retrieval.py`, `requirements.txt`

<a id="unused-summary"></a>
## 17. The model wrote a summary nobody saw

**Generation, 2026-11-09. Outcome: adopted.**

**Problem.** The JSON schema required a free-text summary of up to 800 characters on every call.
Neither interface showed it; the page summary is computed in code.

**Hypothesis.** Removing it from the prompt and schema saves generation time without changing what
the model finds.

**Method.** `scripts/benchmark.py`, three synthetic resumes, 3 timed runs each after a warm-up, M3 Pro,
Ollama 0.40.15 on the Metal GPU, `llama3.2:3b`. Same index and settings for both arms.

**Results.**

| Variant | Generation, median | Total per resume, median | Claims per run | Kept per run | Runs |
|---|---|---|---|---|---|
| With summary | 14.5 s | 15.0 s | 23.2 | 12.1 | 9 |
| Without summary | 13.3 s | 13.8 s | 23.0 | 11.7 | 9 |

**Decision.** Removed. About 1.2 s faster per resume with the same number of claims; the difference in
kept claims is within run-to-run variation at this sample size.

**Evidence.** `scripts/benchmark.py`, `tests/test_gap_analysis.py`

<a id="search-recall"></a>
## 18. The vector search missed a fifth of the nearest chunks

**Retrieval, 2026-11-09. Outcome: adopted.**

**Problem.** Two identical runs of `scripts/eval_query_correction.py` gave 0.76 and 0.80 overlap for
the software resume at 10% typos (correction off), the conflict between two numbers in the docs.

**Hypothesis.** Something in the search depends on the process.

**Method.** First, the same resume and seeds under different `PYTHONHASHSEED` values, comparing the
noisy text, the section embeddings, the ChromaDB hits and the fused ranking. Then
`scripts/eval_search_recall.py`: for every section of the 17 PDF resumes in `eval/resumes/` (91
sections), ChromaDB's top 30 against an exact search over all 237,902 vectors (squared L2, the
collection's space), under two hash seeds. The new index was built by
`scripts/rebuild_index_settings.py`, which copies the stored embeddings into a collection created
with new HNSW settings, so the vectors are identical. M3 Pro.

**Results.**

| Index | Mean recall@30 | Worst section | Sections below 0.9 | Same results across hash seeds | Query time per section |
|---|---|---|---|---|---|
| ChromaDB defaults (search_ef 10) | 0.791 | 0.000 | 57 of 91 | no | about 7 ms |
| search_ef 200, construction_ef 200, M 16 | 0.972 | 0.700 | 6 of 91 | yes | about 7 ms |

Typo-correction overlap at 10%, two runs each on the new index: identical results across runs; the
software resume is 0.82 without correction and 0.98 with.

Setting `hnsw:search_ef` on the existing collection was saved but changed nothing: ChromaDB 0.5.23
applies HNSW settings only when a collection is created. A second ChromaDB quirk found while
copying: once another client has created a collection, `get()` on the source returns ids but no
embeddings, so the rebuild script reads everything before it writes.

**Decision.** New indexes are created with these settings (`HNSW_SETTINGS` in
`retrieval/vector_store.py`), and the full index was rebuilt from its stored embeddings in 399 s
instead of re-embedding the corpus. The old index is kept in `data/processed/` as a backup.

**Evidence.** `scripts/eval_search_recall.py`, `scripts/rebuild_index_settings.py`

<a id="log-no-message"></a>
## 19. Error logs could have included resume text

**Privacy, 2026-11-09. Outcome: adopted.**

**Problem.** The API's last-resort handler used `log.exception()`, which writes the exception message.
A message built from the input would put part of a resume into the server log.

**Hypothesis.** Logging the exception type and stack, without the message, keeps errors debuggable and
keeps resume text out.

**Method.** Unit test: an exception whose message is built from resume text at runtime.

**Results.**

| Variant | Type in log | Stack in log | Resume text in log |
|---|---|---|---|
| Before | yes | yes | yes |
| After | yes | yes | no |

**Decision.** Type and stack only.

**Evidence.** `tests/test_api.py`

<a id="benchmark"></a>
## 20. Stage-by-stage timing of the current pipeline

**Evaluation, 2026-11-09. Outcome: adopted.**

**Problem.** The documented times came from earlier versions of the pipeline (11 s and 37 s on
2026-11-08), and the README rounded a later range to "about 20 s".

**Hypothesis.** Timing each stage on the current code shows where the time goes and replaces the old
numbers.

**Method.** `scripts/benchmark.py` on the three synthetic resumes, one warm-up then timed runs, full
index. Native: M3 Pro (18 GB), Ollama 0.40.15 on the Metal GPU, 3 runs per resume, after entry 17.
Docker: Docker Desktop on the same Mac (11 CPUs, 8 GB), Ollama 0.40.15 in a container on CPU, 2 runs
per resume, measured before entry 17.

**Results.**

| Setup | Total per resume, median | Range | Runs | Generation share |
|---|---|---|---|---|
| Native, Metal GPU | 13.8 s | 11.4 to 19.4 s | 9 | about 96% |
| Docker, CPU only | 57.0 s | 35.5 to 72.3 s | 6 | about 99% |

Every stage other than generation took under 0.4 s per resume (search and fusion: 0.08 to 0.23 s).

**Decision.** These replace the older end-to-end numbers. A 2,000-posting index took 87 s to build on
the same Mac with the dataset already downloaded (531 MB), and is 66 MB on disk (36 MB compressed).

**Evidence.** `scripts/benchmark.py`

<a id="docx-tables"></a>
## 21. Word tables were dropped from resumes

**Parsing, 2026-11-09. Outcome: adopted.**

**Problem.** `12_sysadmin_helpdesk_typos.docx` keeps its skills in a table. The parser read only
`doc.paragraphs`, so the whole table disappeared: 194 words against 240 in the PDF of the same
resume, "VMware" missing, and 3 sections found instead of 4.

**Hypothesis.** python-docx lists tables separately from paragraphs; reading the document body in
order, paragraphs and tables together, keeps the table where it sits.

**Method.** Tests on the DOCX and PDF of the same resume: word count, a skill only the table contains,
the order of summary, table and experience, and the sections found. Sections for all 36 fixture
files were recorded before and after the change.

**Results.**

| Variant | Words (DOCX) | Words (PDF) | "VMware" present | Sections |
|---|---|---|---|---|
| Paragraphs only | 194 | 240 | no | SUMMARY, EXPERIENCE, CERTIFICATIONS |
| Body in order, rows as lines | 242 | 240 | yes | SUMMARY, TECHNICAL SKILLS, EXPERIENCE, CERTIFICATIONS |

Of the 36 fixture files, only this one and entry 22's changed.

**Decision.** The parser walks the document body in order; each table row becomes one line
("Category: items" for two cells), and a merged cell is read once.

**Evidence.** `src/resumematch/ingest/parse_resume.py`, `tests/test_resume_fixtures.py`

<a id="sidebar-headings"></a>
## 22. Skill labels in a PDF sidebar became sections

**Parsing, 2026-11-09. Outcome: adopted.**

**Problem.** In `04_ml_engineer_llm.pdf`, a two-column layout, pypdf extracts the sidebar first. Its
skill categories come out as short standalone lines, and section detection read "Languages" (a known
heading name) and "LLM" (an all-caps line) as headings. The real "Skills" heading, left with no lines
of its own, was merged into the next group, so it disappeared.

**Hypothesis.** Inside a skills section, short labels are categories. Only a major heading
(experience, education, projects, certifications, summary and similar) should end a skills section.

**Method.** Test on the sections found for this resume; the same before-and-after comparison over all
36 fixture files.

**Results.**

| Variant | Sections found |
|---|---|
| Before | HEADER, LANGUAGES, LLM, CERTIFICATIONS, SUMMARY, EXPERIENCE, PROJECTS |
| After | HEADER, SKILLS, CERTIFICATIONS, SUMMARY, EXPERIENCE, PROJECTS |

The other 34 files kept the same sections. The short Education block in this resume is still merged
into the next section by the existing rule for fragments under 12 words; that is separate from this
bug.

**Decision.** While the current section is a skills section, a line starts a new section only if it
is a major heading. Outside a skills section, heading detection is unchanged.

**Evidence.** `src/resumematch/preprocess/resume_sections.py`, `tests/test_resume_fixtures.py`

<a id="grounding-audit"></a>
## 23. Every shown claim traced back to the corpus

**Retrieval, 2026-11-09. Outcome: adopted.**

**Problem.** A RAG system should answer only from the documents it retrieved. The checks were tested
on hand-built cases; they had not been audited on real runs end to end.

**Hypothesis.** On real runs, every claim shown cites a retrieved posting, its excerpt is exactly the
text the model was shown for that posting, that text contains the requirement, and every "shown"
quote is in the resume.

**Method.** `scripts/audit_grounding.py`: the full pipeline on four resumes (the software test resume
and the nurse, accountant and engineer resumes), 2 runs each, `llama3.2:3b` on an M3 Pro. Each shown
claim's excerpt is compared with the requirement text rebuilt from the index for its posting.

**Results.**

| Measure | Count |
|---|---|
| Claims proposed by the model | 184 |
| Shown after the checks | 91 |
| Dropped because no retrieved posting mentions them | 26 |
| Shown claims failing any grounding check | 0 |

**Decision.** The audit script stays as a regression check for changes to retrieval or the checks.
The 26 dropped claims are requirements the model added from its own training, not from the
retrieved postings; the checks are what keep them off the page.

**Evidence.** `scripts/audit_grounding.py`

<a id="sample-index"></a>
## 24. A sample index so nobody needs a Kaggle account

**Deployment, 2026-11-10. Outcome: adopted.**

**Problem.** Trying ResumeMatch required a Kaggle account, a 531 MB download and an index build.
That rules out a quick look.

**Hypothesis.** A small prebuilt index, published as a release asset under the dataset's CC BY-SA
4.0 license, lets anyone run the app after one download.

**Method.** Before packing, the first 2,000 distinct postings were scanned for contact details.
`scripts/make_sample_index.py` builds the index with `--redact-contacts`, keeps posting details for
those postings only, adds the license and an attribution file listing every change, re-scans the
packed text for emails and phone numbers, and records the SHA-256. `scripts/fetch_sample_index.py`
checks that hash and refuses unsafe archive paths. The fetched index was run through the full
pipeline on two resumes. M3 Pro.

**Results.**

| Measure | Result |
|---|---|
| Postings with an email address / a phone number, before redaction | 271 / 72 of 2,000 |
| Emails or phone numbers left in the packed index text | 0 |
| Archive | 37.3 MB, 2,000 postings, 3,927 chunks, search_ef 200 |
| Build time | about 2 minutes, dataset already downloaded |
| Backend engineer resume on the sample | closest match "Java Technical Lead", 14 claims shown |
| Marketing manager resume on the sample | closest match "Integrated Marketing Manager", 15 claims shown |

**Decision.** The quick start uses the sample. It is published as the GitHub Release
`sample-index-v1`; the full index still needs the dataset. Fetch safety is covered by tests:
checksum mismatch, unsafe paths and an existing index.

**Evidence.** `scripts/make_sample_index.py`, `scripts/fetch_sample_index.py`,
`src/resumematch/preprocess/redact.py`, `tests/test_fetch_sample_index.py`, `tests/test_redact.py`

---

# Appendix: design notes, Nov 7 to Nov 9

The narrative below was written while the system was being built and lived in docs/design.md
until 2026-11-09. It is kept here in full, in its original order, as the record of how each
decision was reached. Numbers marked *superseded* came from the 2k or 20k development
indexes; docs/design.md lists the current values and their sources.

### Why the resume is the query, not the indexed document

I wanted a large, real corpus and citations that click through to the
source passage. A single resume and a single job posting give neither:
there's nothing to scale and nothing to cite into. Reframing the job-
postings corpus (123,849 rows; 107,528 distinct postings indexed) as the indexed database, with the
resume as the query, satisfies both: the corpus is large and real, and
every retrieved result carries a `job_id` / `job_posting_url` that
resolves to an actual LinkedIn posting.

### Why no LangChain/LlamaIndex

A RAG framework optimizes for not having to understand chunking,
embedding, and retrieval. Writing those steps myself means every design
choice is one I can explain, not "whatever the framework defaults to."

### Why local-only LLM (Ollama), not a paid API

- No dependency on a key that could be revoked or rate-limited.
- Resumes are personal data; a local model keeps them on the machine.
- Anyone can reproduce the results without personal API credentials.

### Why ChromaDB over raw FAISS

FAISS is index-only: no persistence, no metadata filtering, build that
yourself. ChromaDB gives local file-based persistence and metadata
filtering (needed for citations: job_id, section, url, title) for free,
with no server process to run. At this data scale (chunks from 124k
postings, not millions), FAISS's raw speed advantage doesn't matter.

### Query correction: an edit-distance autocorrect on resume text

Resume text extracted from a PDF frequently has OCR/extraction
artifacts (dropped characters, run-together words). A single corrupted
token can hurt embedding-similarity retrieval even though a human
reader wouldn't notice it. An edit-distance autocorrect I had written
earlier (`probable_substitutes`/`min_edit_distance`) is reused,
with the vocabulary built from the FULL job-posting corpus (so tech
jargon like "faiss," "pytorch," "redis" survives as "known" rather than
being "corrected" into something else).

**What went wrong first, and the fix.** Run as-is (vocabulary from a
2k-posting sample, "correct any unknown word"), it changed 42 words on a
clean test resume and every change was wrong (Redis→Hedis,
OpenAI→Openam, Khoury→Hours). The original autocorrect assumed a short, typo-heavy
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
20k-posting index, sectioned retrieval (see next section), 5 seeds.
*Superseded: measured on the 20k development index; the full-index runs are in
entries 2 and 8 and in docs/design.md.*

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

### Why the resume is searched as several queries, not one

The embedder (bge-small) reads at most 512 tokens and silently drops the
rest. Embedding the whole resume as one query meant the test
resume (1,015 tokens by the tokenizer count used in later runs) was retrieved on its first 510 tokens only: its entire EXPERIENCE
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
   closely; and results are distinct postings (each with its closest
   chunk) rather than several chunks of one posting.

Why sections rather than fixed windows: each query vector then means one
thing ("what they built", "what tools they know"), and the UI can show
which resume sections matched each posting.

Measured (`scripts/eval_resume_coverage.py`, test resume, 20k postings; *superseded by the
full-index numbers below*):

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

### Generating the gap analysis: what the local models actually do

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
*enforces* it. Every format failure I measured (missing `[N]`, `- None`
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

**Change 4 -- check every claim from both sides (reasoning quality).**
Constrained decoding fixed the shape, not the judgement. Counting
errors over 6 runs (llama3.2:3b, after the chunking fixes):

- 35/48 (73%) "matched" requirements did not appear in the posting they
  cited. "Java", "Kafka", "Spring Boot" were cited to a React Native
  posting that mentions none of them -- the model was listing the
  candidate's own skills as "matched requirements" with an arbitrary
  citation. The quote-in-resume check couldn't catch it: the quotes
  were real; the link to the posting was invented.
- 25/48 (52%) "missing" items were skills the resume has (AWS, Docker,
  Python, C++, REST APIs); 14 weren't in the cited posting either.
- 6/48 matched quotes didn't mention the requirement ("Backend" <-
  "Node.js, Express, Kafka").

`check_claims` now verifies, on key terms (the requirement minus filler
like "experience with", "strong knowledge of"):

- every claim: its requirement appears in the cited posting (at least
  half its key terms). If not, but ANOTHER retrieved posting mentions
  it, the citation is repaired to that posting (the model often gives a
  real requirement the wrong number) and flagged `recited`;
- matched: the quote is in the resume AND mentions the requirement;
- missing: the resume does not contain all of the requirement's key
  terms ("AWS Lambda" stays missing for a resume with AWS but no Lambda;
  "AWS" alone is a contradiction).

Failures go to `unverified` with the reason, shown in the UI; the
model's free-text summary is labelled as unchecked. One prompt line was
also added: write requirements in the excerpt's own words.

| llama3.2:3b, 6 runs | verified matched / run | verified missing / run | correct on inspection |
|---|---|---|---|
| quote check only | 8.0 | 8.0 | no -- see error counts above |
| + two-sided checks | 1.7 | 3.3 | yes |
| + "excerpt's own words" prompt | 3.7 | 4.0 | yes |
| + citation repair | **3.8** | **4.7** | yes |

Kept items across those runs -- matched: React Native, React, Redux,
JavaScript, Node.js, AWS, Distributed Systems; missing: Jest, Enzyme,
Webpack, Babel, ESLint, MobX, Context API, HTML/CSS, AWS Lambda, CDK,
SQS, EKS -- all checked by hand against the resume and postings.
"Correct on inspection" is my reading, not an independent label set.

Limits: the checks are lexical, so synonyms fail toward "unverified"
(a quote of "Unit Testing" for "test-driven development" is rejected),
and a paraphrased quote ("Infra & DevOps: AWS" for a resume line with
more items in between) is rejected rather than fuzzily accepted --
loosening that would let invented quotes through. The output is smaller
but trustworthy; recall is the cost. The summary is still free model
text and can contradict the checked lists.

### Chunking job postings: sentences, boilerplate, duplicates

Reading real chunks (as this doc said to) turned up three problems.
Measured on the whole corpus first, so the fixes target real patterns:

- The old chunker split on whitespace and re-joined with spaces,
  throwing away the line breaks that 95.7% of descriptions have, so
  bullets ran together and chunks were cut mid-sentence. 53% of
  descriptions also have sentences glued with no space ("oil
  baths.Use standard").
- 12.9% of postings (16,015; *superseded: the full build skipped 16,314, measured from
  postings.csv on 2026-11-09*) are exact copies of an earlier posting's
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
*distinct* postings). *Superseded by the full-index row below: 2.9% EEO, 68.4% at a
sentence boundary.*

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

### Full-scale results (the numbers to quote)

Full index, built with `python scripts/build_index.py` (no `--limit`)
in ~40 min on an M3 Pro: 123,849 postings -> 107,528 distinct (16,314
exact duplicates skipped) -> **237,902 chunks**, 3.6 GB on disk.
Earlier sections' numbers came from 2k/20k dev indexes; these supersede
them. Measured on the one dummy test resume; see "Results across four
resumes" below for three more fields, which revise two of these rows.

| what | result | script |
|---|---|---|
| chunks with EEO/accommodation text | 2.9% (26.9% before cleanup) | stats over the index |
| chunks ending at a sentence boundary | 68.4% (25.9% before) | stats over the index |
| resume tokens retrieval sees | 1,015 / 1,015 (single vector: 510, 50%) | `eval_resume_coverage.py` |
| skills-swap sensitivity, top-10 overlap | sectioned **0.30**; single vector 1.00 (blind) -- does not generalize, see below | `eval_resume_coverage.py` |
| false corrections on the clean resume | 0 | `eval_query_correction.py` |
| retrieval kept under 10% typos | 0.80 -> **0.98** with correction | `eval_query_correction.py` |
| retrieval kept under 20% typos | 0.70 -> 0.84 with correction | `eval_query_correction.py` |
| gap analysis, verified per run (6 runs) | 4.5 matched, 3.5 missing; 0 format failures -- superseded below | llama3.2:3b |
| latency | retrieval 0.04 s; full pipeline 11 s (LLM-bound) | M3 Pro, Metal GPU |

Top 10 for the test resume (a React Native / Java / Spring developer):
Senior React Native Developer, React Native Developer, react native
lead, Senior Frontend/App Developer, Fullstack React/Node.js Developer,
Android Developer, Senior Software Engineer (x2, different companies)...
Verified matches across runs: Java, Kafka, Docker, React Native, Redux,
JavaScript. Verified gaps: Jest, Enzyme, Webpack, Babel, ESLint, MobX,
Context API, HTML/CSS, Kafka Streams, Kubernetes.

Still to do before quoting these as general: time the pipeline on the
CPU-only host intended for the portfolio deploy.

### Results across four resumes (Nov 8)

Three synthetic resumes in `eval/resumes/` (fictional people, safe to
commit), chosen to be unlike the software test resume: an ICU **nurse**,
a staff **accountant**, and a manufacturing **mechanical engineer** whose
resume uses Title Case headings ("Work History"). All three are short
(326-420 tokens), unlike the 1,015-token software resume.

**Retrieval is on-target in every field.** Top 5: ICU / cardiac ICU /
ED nurse roles at Boston-area hospitals; Staff/Senior Accountant roles in
Chicago; BIW tooling designer, manufacturing and process engineer roles.

**The skills-swap check does not generalize.** Replacing the last
section with another profession's skills left the sectioned top 10
unchanged (overlap 1.00) for the nurse and the accountant, while the
single vector moved (0.60). The section is read; it's outvoted: RRF
rewards postings found by several sections, so one off-topic section's
hits can't outrank postings that 3-4 sections agree on. That is the
intended consensus behaviour for a coherent resume, but it means a
career-changer whose skills section points somewhere else than their
experience will get results driven by the experience. The software
resume's 0.30 came from its sections being more varied. Not changed;
weighting sections is an option if this matters.

**Bugs found and fixed:**

| bug | example | fix |
|---|---|---|
| Title Case headings not detected -> whole resume one query | "Work History", "Certifications and Tools" | heading = Title Case line with a heading keyword ("history", "skills", ...); job titles ("Manufacturing Engineer II") have none. 2 tests |
| company name "corrected" | Midstate -> **Mistake** (clean resume) | capitalized words mid-line (after a dash separator too) get edit-1 fixes only, no edit-2: still fixes "Kubernets", can't rewrite names. 4 tests |
| employer benefits listed as missing skills | "403(b) Retirement Savings plan", "Pension", "Medical, Dental, Vision", "Relocation assistance" | rejected in code by a narrow pattern ("computer vision", "medical-surgical" pass). 10 tests |
| false gaps from plurals/filler | "Current BLS certification" vs "CERTIFICATIONS ... BLS"; "Microsoft Excel" vs "Excel" | crude plural stemming on both sides; "current", "microsoft" are filler. 2 tests |

A prompt fix for the benefits was tried first and **rejected on
measurement**: adding "never list what the employer offers" to the
prompt cut verified gaps from 2.8 to 0.2 per run (software resume) and
4.3 to 2.0 (accountant), and didn't reduce benefit claims (6 vs 6 in 6
runs). With a 3B model, extra negative instructions read as "list less".

**After the fixes** (`eval_query_correction.py`, 5 seeds; gap analysis
4 runs each, llama3.2:3b, native Ollama on an M3 Pro):

| resume | false corrections (clean) | overlap at 10% typos, off -> on | verified matched / run | verified missing / run | s / run |
|---|---|---|---|---|---|
| software (dummy) | 0 | 0.76 -> 0.98 | 3.2 | 0.2 | 8 |
| nurse | 0 | 0.80 -> 0.94 | 4.0 | 5.2 | 11 |
| accountant | 0 | 0.60 -> 0.96 | 5.2 | 4.2 | 11 |
| mech. engineer | 0 (was 1) | 0.86 -> 1.00 | 1.5 | 7.8 | 13 |

0 format failures in 16 runs. Gaps kept read as real gaps: nurse --
sepsis/septic shock, cardiogenic shock, renal failure, arrhythmias;
accountant -- SOX compliance, SAP, Big Four experience, people
management; engineer -- auto assembly strategies, NX CAD, axle/plastic
component design.

The software resume's verified gaps dropped from 3.5 to 0.2 per run
because the plural fix now (correctly) rejects "REST APIs"/"OpenAI API"
for a resume that says "API"; the model's other "missing" items for it
(React, Next.js, Node.js) are on the resume. For a well-matched
candidate, most of what a 3B model calls missing isn't.

**Still open (seen in these runs, not fixed):**
- Abbreviation/synonym gaps: "ICU experience" for "Medical Intensive
  Care Unit"; "Bachelor's degree in engineering" for "B.S. Mechanical
  Engineering". Fails toward a wrong gap; needs a synonym list.
- The engineer gets few verified matches (1.5 / run): 14 of the model's
  31 quotes weren't in the resume. Quote fidelity varies by resume.
- Some "requirements" are posting fragments with glued sentences
  ("...Body in White AssemblyApproximately 10 years...") from postings
  whose source text has no spaces between sentences.
- Synthetic resumes written by us, so they're cleaner than real ones;
  real PDFs (columns, odd layouts) are still untested.

### Dataset

**LinkedIn Job Postings (2023-2024)**, Kaggle, uploaded by `arshkon`
(scraper: [ArshKA/LinkedIn-Job-Scraper](https://github.com/ArshKA/LinkedIn-Job-Scraper)),
with 100,000 additional postings contributed by `zoeyyuzou`. 123,849
rows in `postings.csv` (version 13, pinned), plus linked `companies/companies.csv`,
`benefits.csv`, `employee_counts.csv`. License: **CC BY-SA 4.0**:
attribution required; any derivative/processed form distributed from
this project must remain under the same license.

Why this dataset specifically: `job_posting_url` makes citations verify
against the live source, not just an opaque internal ID, so every claim
links to its source posting. `description` and
`skills_desc` give two independently citable text fields per posting.

Downloaded via `kagglehub` at container-run time (not baked into the
image), so a fresh clone downloads the data itself.
Requires `KAGGLE_USERNAME`/`KAGGLE_KEY` env vars, documented in
README, not committed (see `.gitignore`).

### Deployment: two profiles, one codebase, and the budget math

- **GPU VM** (`docker-compose.gcp.yml`): GCP VM with a T4 GPU, about
  $0.50 to $0.55 an hour. The index
  is built LOCALLY (CPU, free) and `scp`'d to the VM rather than
  rebuilt there, so the VM only ever pays for LLM inference, not
  embedding 124k rows. `gcp/start.sh`/`gcp/stop.sh` exist specifically
  so stopping the VM is a one-command habit, not something to remember
  at 1am.
- **Long-running public demo** (`docker-compose.local.yml` as the
  base): a GPU VM left running costs ~$350+/month, which isn't viable.
  The plan is a cheap or free CPU-only host with a smaller quantized
  model.

**Docker, verified (Nov 8).** Problems found and fixed before the first
build: no `.dockerignore`, so `COPY . .` would have baked `.env` (the
Kaggle key), `.venv` and the 3.6 GB index into the image; Ollama started
with no model; the correction vocabulary lived outside any volume, so
typo correction would silently switch off after the first restart; a
crashed index build looked "complete" to the entrypoint; the default
CUDA torch build (~2.5 GB) would be pulled on x86 for an app that only
embeds a few short queries; `ollama:latest` was unpinned although
schema-constrained output needs a recent Ollama; Ollama's port was
published (on GCP that exposes the GPU to the internet).

Now: CPU torch + the embedding model baked in (image 2.19 GB, no
secrets -- checked); `scripts/wait_for_ollama.py` waits for Ollama and
pulls the model; `chroma_db/` and `data/processed/` are bind-mounted and
reused when `BUILD_COMPLETE.json` (written last by `build_index.py`)
exists, otherwise rebuilt from an empty folder; Ollama pinned to 0.40.15, no
published port. Verified: existing full index reused (no rebuild);
fresh-clone path (empty folders) downloads from Kaggle, builds, starts
the app; end-to-end inside the container on CPU-only Ollama: **37 s per
resume warm** (69 s cold), identical gap-analysis output to native.
Ollama used ~3.4 GB RAM with llama3.2:3b -- a CPU-only portfolio host
needs >= 4-6 GB RAM, which rules out the smallest free VMs.

**Decision (Nov 8): laptop first.**
Development and demos run on the M3 Pro: the app in Docker, the LLM in
native Ollama on the Metal GPU (`OLLAMA_HOST_FOR_APP`), measured at
~11 s per resume warm -- faster than any other setup tested, free, and
no network or VM dependency. Fully containerized on the Mac (CPU-only
Ollama) is ~37 s. The GCP T4 profile is the step toward a public link
and is set up later.

GCP default model changed to llama3.2:3b (`OLLAMA_MODEL_GCP` overrides):
it beat llama3.1:8b on accuracy and latency in every measurement. Not
yet verified: the GCP profile itself (NVIDIA runtime, scp of the index).

### Known risks list, as of Nov 9 (all items below were resolved)

- **Resolved**: all modules are now implemented (chunker, query
  correction, embedder, vector store, resume parsing, Ollama client,
  gap-analysis parsing). Unit tests pass for everything that doesn't
  require the real dataset, a local embedding model, or a running
  Ollama instance (see docs/design.md, "Verified and not verified").
- **Resolved**: query correction checked against a real resume and
  reworked (see "The query-correction tie-in" above): full-corpus
  vocabulary, guard rails, 0 false corrections on the clean test resume,
  corrections shown in the UI with an on/off toggle, measured by
  `scripts/eval_query_correction.py`.
- **Resolved (format drift, output length)**: constrained decoding
  (JSON schema), see "Generating the gap analysis" above. 8/8 runs
  usable on the first attempt with llama3.2:3b.
- **Resolved (over-claiming, contradictory gaps)**: every claim is
  checked from both sides (requirement in the cited posting, quote in
  the resume and on-topic, "missing" not plainly in the resume), with
  citation repair. Open: lexical checks miss synonyms (fails safe, to
  "unverified"); the model's summary is unchecked free text.
- **Resolved (resume truncation)**: the resume is now searched as
  token-bounded sections fused with RRF (see "Why the resume is searched
  as several queries" above).
- **Resolved (chunking, boilerplate, duplicates)**: see "Chunking job
  postings" above. EEO text 26.9% -> 3.0% of chunks on the 20k development index (*superseded: 2.9% on the full index*); duplicate postings
  skipped; at most 2 postings per company in results. Open: one
  company's repeated "About us" text is still indexed.
- **Resolved**: company names are in chunk metadata, the LLM prompt and
  the UI (straight from postings.csv's company_name column; the separate
  companies.csv join was removed as unnecessary).
