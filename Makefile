# Shortcuts for the common tasks. Each target is one command you can also run by hand.
PY := .venv/bin/python
RESUMES := eval/resumes/nurse.docx eval/resumes/accountant.docx eval/resumes/mecheng.docx

.PHONY: setup test lint sample-index index-small run api web eval benchmark

setup:          ## create .venv and install the Python package and dependencies
	python3.11 -m venv .venv
	.venv/bin/pip install -r requirements.txt
	.venv/bin/pip install -e .

test:           ## unit tests (no model, index or network needed)
	$(PY) -m pytest -q

lint:           ## ruff, with the rules in pyproject.toml
	.venv/bin/ruff check src tests scripts

sample-index:   ## download the prebuilt 2,000-posting index (no Kaggle account needed)
	$(PY) scripts/fetch_sample_index.py

index-small:    ## 2,000-posting index (needs Kaggle credentials in .env)
	$(PY) scripts/build_index.py --limit 2000

run:            ## Streamlit app on http://localhost:8501
	.venv/bin/streamlit run src/resumematch/frontend/streamlit_app.py

api:            ## JSON API for the Next.js site, on http://127.0.0.1:8000
	.venv/bin/uvicorn resumematch.api.app:app --host 127.0.0.1 --port 8000

web:            ## Next.js site on http://localhost:3100 (start the API first)
	cd web && npm install && npm run dev

eval:           ## typo-correction and resume-coverage evaluations on the synthetic resumes
	$(PY) scripts/eval_query_correction.py $(RESUMES)
	$(PY) scripts/eval_resume_coverage.py eval/resumes/accountant.docx

benchmark:      ## time each pipeline stage (needs the index and a running Ollama)
	$(PY) scripts/benchmark.py
