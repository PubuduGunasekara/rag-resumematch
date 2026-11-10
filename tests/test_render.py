"""Results-page HTML: escaped, highlight-correct, and free of AI tells."""

import re

from resumematch.frontend import render
from resumematch.generation.job_report import JobAnalysis, JobReport, LearnItem, Requirement

RESUME = "Maria Delgado, RN\nACLS, BLS, PALS — American Heart Association\nSystems: Epic, <Pyxis>"


def job(tier="best"):
    reqs = [Requirement("ACLS", True, "required", False, "ACLS, BLS", "ACLS required."),
            Requirement("APQP tools including FMEA", False, "unspecified", False, None, None, partly=["fmea"]),
            Requirement("Siemens NX", False, "preferred", False, None, None),
            Requirement("Teamwork", False, "unspecified", False, None, None, soft=True)]
    return JobAnalysis(1, "1", "RN ICU", "Southcoast <Health>", "https://x/1",
                       {"location": "New Bedford, MA", "work_type": "Part-time", "salary": None}, reqs,
                       tier=tier, reason="2 requirements not found on your resume yet.")


def all_html():
    j = job()
    report = JobReport([j], [LearnItem("Siemens NX", 1, False)], "Your closest match is RN ICU", "These are the 1 postings closest to your resume.", best=j)
    return "".join([render.header(report, 107528), render.best_fit(j), render.job_body(j),
                    render.learn_first(report.learn_first, 1), render.highlighted_resume(RESUME, ["ACLS, BLS"])])


def test_no_em_dashes_or_emoji_in_our_own_text():
    out = all_html().replace("— American", "")  # the resume's own dash is a quote, shown as written
    assert "—" not in out and "–" not in out
    assert not re.search("[\U0001F300-\U0001FAFF☀-➿]", out)


def test_user_text_is_escaped():
    out = all_html()
    assert "Southcoast &lt;Health&gt;" in out and "&lt;Pyxis&gt;" in out and "<Pyxis>" not in out


def test_statuses_and_counts_are_plain_words():
    out = render.best_fit(job())
    assert "1 of 3 requirements shown on your resume, 1 partly" in out  # soft skill not counted
    assert ">Shown<" in out and ">Partly shown<" in out and ">Not found<" in out
    assert "Missing" not in out and "missing" not in out
    assert "general strengths: Teamwork" in out


def test_quote_is_highlighted_across_punctuation_and_case():
    out = render.highlighted_resume(RESUME, ["acls bls"])
    assert "<mark>ACLS, BLS</mark>" in out


def test_overlapping_quotes_merge_into_one_mark():
    out = render.highlighted_resume("Skills: Python, SQL, Docker", ["Python, SQL", "SQL, Docker"])
    assert out.count("<mark>") == 1 and "<mark>Python, SQL, Docker</mark>" in out


def test_facts_skip_missing_fields():
    assert render.facts({"location": "Boston, MA", "salary": None}) == '<div class="facts"><span>Boston, MA</span></div>'
    assert render.facts({}) == ""


def test_header_states_the_real_count_or_none_at_all():
    j = job()
    report = JobReport([j], [], "Your closest match is RN ICU", "Summary.", best=j)
    assert "Searched 2,000 LinkedIn postings." in render.header(report, 2000)
    assert "Searched" not in render.header(report, None)
