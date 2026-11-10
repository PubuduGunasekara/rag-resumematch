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

<a id="vocab-full-corpus"></a>
## 1. Build the spelling vocabulary from all postings, not a sample

**Query correction, 2026-11-07. Outcome: adopted.**

**Problem.** The corrector, adapted from the Assignment 3 edit-distance autocorrect, rewrote real words on a clean resume (Redis, Vite, OpenAI, common first names) because a 2,000-posting vocabulary had never seen them.

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

**Decision.** Laptop demo uses native Ollama; GCP T4 for the course deployment.

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

