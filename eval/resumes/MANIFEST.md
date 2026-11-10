# Synthetic test resumes for ResumeMatch

17 fictional people (names, employers, schools, emails and phones are invented; emails use example.com, phones use 555-01xx). None is derived from a real resume, so these are safe to commit under eval/resumes/. Each exists as PDF; all except #04 also as DOCX.

Token counts are ESTIMATES (words x 2.0; the design doc measured about 2 tokens per word on resumes). The embedder reads 512 tokens, so anything above about 250 words exercises the multi-section path.

| # | File | Role | Layout | Words | Est. tokens | What it tests |
|---|---|---|---|---|---|---|
| 01 | 01_backend_java_engineer | Backend Engineer | one column, ALL-CAPS headings | 377 | ~754 | Mid-level backend, Java/Spring/Kafka. Should retrieve backend, platform and payments roles. |
| 02 | 02_frontend_react_engineer | Frontend Engineer | one column, Title Case headings ('Work History') | 340 | ~680 | Frontend React/TypeScript with accessibility focus. Title Case headings ('Work History'). |
| 03 | 03_fullstack_python_startup | Full-Stack Software Engineer | one column, ALL-CAPS headings | 304 | ~608 | Startup full-stack, Python/Django + React. Broad skills; tests varied-section voting. |
| 04 | 04_ml_engineer_llm | Machine Learning Engineer | two-column sidebar (PDF only) | 318 | ~636 | ML engineer with LLM/RAG experience, two-column PDF layout. Tests column extraction order. |
| 05 | 05_data_scientist_entry_level | Data Scientist (Entry Level) | one column, ALL-CAPS headings | 168 | ~336 | Short entry-level resume (about 350 tokens), student with one internship. |
| 06 | 06_data_engineer_spark | Data Engineer | one column, Title Case headings ('Work History') | 304 | ~608 | Data engineer: Spark, Airflow, dbt, Snowflake. Title Case headings. |
| 07 | 07_devops_sre_kubernetes | Site Reliability Engineer | one column, Title Case headings ('Work History') | 311 | ~622 | DevOps/SRE: Kubernetes, Terraform, observability. Title Case headings, includes Go. |
| 08 | 08_cloud_solutions_architect_senior | Cloud Solutions Architect | one column, ALL-CAPS headings | 443 | ~886 | Senior, long (1,000+ tokens, two pages). Tests the 480-token section cap and multi-piece sections. |
| 09 | 09_security_analyst_soc | Cybersecurity Analyst | one column, ALL-CAPS headings | 334 | ~668 | SOC analyst: SIEM, incident response, certifications. Acronym-heavy (tests typo-correction guard rails). |
| 10 | 10_qa_sdet_automation | SDET / QA Automation Engineer | one column, Title Case headings ('Work History') | 320 | ~640 | SDET: Selenium, Cypress, REST Assured. Tests whether QA roles are retrieved distinctly from dev roles. |
| 11 | 11_android_kotlin_developer | Android Developer | one column, ALL-CAPS headings | 284 | ~568 | Mobile: Kotlin, Jetpack Compose. Compare against React Native postings and iOS roles. |
| 12 | 12_sysadmin_helpdesk_typos | Systems Administrator | skills in a table | 240 | ~480 | IT support/sysadmin with a skills TABLE (DOCX tables were skipped by parse_resume until 2026-11-09) and about 8 natural typos. Tests parsing and typo correction. |
| 13 | 13_it_business_analyst | IT Business Analyst | one column, Title Case headings ('Work History') | 322 | ~644 | IT-adjacent, non-coding. Tests whether BA/product-analyst roles are separated from engineering roles. |
| 14 | 14_embedded_firmware_engineer | Embedded Software Engineer | one column, ALL-CAPS headings | 185 | ~370 | Embedded/firmware: C, RTOS. Tests a niche that the LinkedIn corpus covers thinly. |
| 15 | 15_marketing_manager | Digital Marketing Manager | one column, Title Case headings ('Work History') | 167 | ~334 | Non-IT control: retrieval should return marketing roles, not software roles. |
| 16 | 16_high_school_teacher | High School Mathematics Teacher | one column, ALL-CAPS headings | 150 | ~300 | Non-IT control, short resume. |
| 17 | 17_career_changer_chef_to_web | Junior Web Developer (Career Changer) | one column, ALL-CAPS headings | 190 | ~380 | Career changer: experience says kitchen, skills say web dev. Tests section voting (known open issue). |

## Parser findings from running these through the code (Nov 9)

1. **DOCX tables were dropped (fixed 2026-11-09, engineering log entry 21).** `parse_resume` reads only `doc.paragraphs`. In `12_sysadmin_helpdesk_typos.docx` the whole skills table disappears (194 words vs 240 in the PDF; 'VMware' is missing), and the section splitter then finds 3 sections instead of 4. Real resumes use tables often. Fix: also read table cells, in document order.
2. **Two-column PDF (fixed 2026-11-09, engineering log entry 22).** pypdf emits the sidebar first, then the main column. Section detection then treats short skill-category lines ('Languages', 'LLM') as headings: `04_ml_engineer_llm.pdf` splits into HEADER, LANGUAGES, LLM, CERTIFICATIONS, SUMMARY, EXPERIENCE, and the 'Skills' and 'Education' headings are lost. Retrieval still sees all the text, but section labels in the UI are wrong.
3. **Tiny sections are merged forward.** Education (and sometimes the header) merges into the next section, so 'EDUCATION' often does not appear as its own section. Probably fine; confirm the UI does not rely on it.
4. `12_sysadmin_helpdesk_typos` has 7 deliberate typos (enviroments, troubleshoting, Acitve Directory, Exchnage Online, mailbxoes, Adminster, onbaording; the first appears in the summary, the other six are in skills and bullets). Use it to check typo correction on a realistic, non-injected case.

## Suggested use

- Run all 17 through the pipeline and save the top 5 per resume. Label those by hand (relevance 0/1/2) for precision@5.
- IT roles to check separately: backend vs full-stack vs frontend vs SDET vs SRE vs cloud architect vs data engineer vs ML engineer vs security vs sysadmin vs BA vs embedded vs Android.
- Controls: #15 marketing and #16 teacher should return no software roles. #17 (sous chef, but web-dev skills) tests the known open issue where one off-topic section is outvoted by the others.
- 'Expected' results are hypotheses, not ground truth. The corpus is LinkedIn postings from 2023-2024; judge by the posting text, not by the title.

The script that generated these files is not in this repository.
