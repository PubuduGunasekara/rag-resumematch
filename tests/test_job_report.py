"""Tier, importance and learn-first logic of the per-job report, on
hand-built claims (no Ollama, no index)."""

from resumematch.generation.gap_analysis import CitedClaim, GapAnalysisResult
from resumematch.generation.job_report import (
    JobAnalysis, Requirement, assign_tiers, build_report, classify_importance, learn_first,
)


def req(text, met, importance="unspecified", credential=False):
    return Requirement(text, met, importance, credential, "q" if met else None, None)


def job(rank, reqs, job_id=None):
    return JobAnalysis(rank, job_id or str(rank), f"Job {rank}", "Co", None, {}, reqs)


def test_importance_is_read_from_the_posting_sentence():
    assert classify_importance("CCRN certification preferred.") == "preferred"
    assert classify_importance("Epic experience is a plus") == "preferred"
    assert classify_importance("Must have 2 years of ICU experience.") == "required"
    assert classify_importance("Minimum qualifications: BSN") == "required"
    assert classify_importance("You will work with Epic daily.") == "unspecified"
    assert classify_importance(None) == "unspecified"


def test_missing_required_credential_is_not_a_match():
    # The Stroke NP posting: needs a master's and NP certification.
    j = job(1, [req("RN license", True), req("NIH Stroke Scale", True),
                req("Master's degree in nursing", False, "required", True)])
    assert assign_tiers([j]) is None
    assert j.tier == "no" and "master's degree in nursing" in j.reason


def test_preferred_credential_gap_does_not_block():
    j = job(1, [req("ICU", True), req("MSN", False, "preferred", True)])
    assert assign_tiers([j]) is j and j.tier == "best"


def test_best_fit_is_the_closest_posting_without_a_credential_gap():
    # The real failure this replaces: an RN ICU posting for an ICU nurse
    # scored 1 of 9 on granular items and was called "not a match".
    np_job = job(1, [req("Nurse practitioner license", False, "required", True)])
    icu = job(2, [req("ICU", True)] + [req(f"Experience in condition {i}", False) for i in range(8)])
    ed = job(3, [req("ACLS", True)])
    assert assign_tiers([np_job, icu, ed]) is icu
    assert [x.tier for x in (np_job, icu, ed)] == ["no", "best", "potential"]
    assert icu.reason == "8 requirements not found on your resume yet."


def test_soft_skills_are_shown_apart_and_not_counted():
    from resumematch.generation.job_report import is_soft
    assert is_soft("Expert clinical judgment and ability to be self-directive")
    assert is_soft("Full command (verbal and written) of the English language")
    assert not is_soft("Ventilator management")
    j = job(1, [req("Ventilators", True), Requirement("Teamwork", False, "unspecified", False, None, None, soft=True)])
    assert (j.met, j.checked, len(j.gaps)) == (1, 1, 0)


def test_learn_first_groups_wording_variants_and_counts_jobs():
    j1 = job(1, [req("Cardiogenic shock experience", False), req("SAP", False)], "1")
    j2 = job(2, [req("Cardiogenic shock", False)], "2")
    items = learn_first([j1, j2])
    assert (items[0].text, items[0].jobs) == ("Cardiogenic shock", 2)
    assert items[1].text == "SAP" and items[1].jobs == 1


def test_build_report_groups_one_pooled_analysis_by_posting():
    t1 = "Must have BLS certification. Requires a master's degree in nursing."
    t2 = "ICU experience required. Epic experience preferred."
    calls = []

    def analyze(resume, excerpts):
        calls.append(excerpts)
        def mk(r, job, q=None):
            return CitedClaim(r, job, None, t1 if job == "1" else t2, q)
        return GapAnalysisResult(
            matched=[mk("BLS certification", "1", "BLS"), mk("ICU experience", "2", "ICU")],
            missing=[mk("Master's degree in nursing", "1"), mk("Epic experience", "2"), mk("Epic", "2")],
            overall_summary="x — y", raw_response="{}")

    retrieved = [{"job_id": "1", "title": "Stroke NP", "company_name": "BMC", "job_posting_url": "u1", "text": t1},
                 {"job_id": "2", "title": "RN ICU", "company_name": "Southcoast", "job_posting_url": "u2", "text": t2}]
    report = build_report(retrieved, "BLS ICU", analyze, lambda _id: [], {"2": {"location": "New Bedford, MA"}})
    assert len(calls) == 1 and len(calls[0]) == 2          # one LLM call, one excerpt per posting
    np_job, icu = (next(j for j in report.jobs if j.job_id == i) for i in ("1", "2"))
    assert np_job.tier == "no" and icu.tier == "best" and report.best is icu
    assert icu.details["location"] == "New Bedford, MA"
    assert [(r.text, r.met, r.importance) for r in icu.requirements] == [
        ("ICU experience", True, "required"), ("Epic experience", False, "preferred")]  # "Epic" deduped
    assert report.headline == "Your closest match is RN ICU at Southcoast"
    assert "—" not in report.summary


def test_importance_follows_the_nearest_clause():
    s = "BLS required; ACLS preferred."
    assert classify_importance(s, "ACLS") == "preferred"
    assert classify_importance(s, "BLS") == "required"
    assert classify_importance("ACLS certification required, CCRN a plus", "ACLS certification") == "required"


def test_requirement_excerpt_drops_employer_sentences():
    from resumematch.generation.job_report import requirement_excerpt
    text = ("We offer child care and student loan subsidies. Our mission is to care for every family. "
            "Must have 1 year of critical care RN experience. Current BLS and ACLS certification required. "
            "Experience with CRRT, ventilators and arterial lines preferred. Strong charting in Epic. "
            "Able to work 12 hour night shifts. Precepting new graduate nurses is a plus for this unit.")
    out = requirement_excerpt([{"job_id": "1", "job_posting_url": None, "text": text, "section": "description"}])
    assert len(out) == 1 and "child care" not in out[0]["text"] and "mission" not in out[0]["text"]
    assert "ACLS" in out[0]["text"] and out[0]["job_id"] == "1"


def test_gap_with_half_its_terms_on_the_resume_is_partly_shown():
    # Real: "Knowledge of APQP tools including FMEA, SPC..." shown as not
    # found for a resume listing APQP and FMEA.
    from resumematch.generation.job_report import build_requirements
    posting = "Knowledge of APQP tools including FMEA and SPC required."
    def mk(r):
        return CitedClaim(r, "1", None, posting, None)
    resume = "Tools: FMEA, 8D, PPAP, APQP, SMED"
    reqs = build_requirements([(mk("APQP tools including FMEA and SPC"), False), (mk("Siemens NX"), False)], posting, resume)
    assert [(r.text, r.status) for r in reqs] == [("APQP tools including FMEA and SPC", "partly"), ("Siemens NX", "not_found")]
    assert reqs[0].partly == ["apqp", "fmea"]


def test_one_common_word_is_not_partly_shown():
    from resumematch.generation.job_report import _partly_shown
    assert _partly_shown("Multidisciplinary team", "Member of the rapid response team") == []


def test_no_evidence_anywhere_means_no_closest_match():
    # A non-resume pushed through with "Check it anyway" must not get a
    # confident "Your closest match is..." report.
    from resumematch.generation.job_report import NO_EVIDENCE_HEADLINE

    def analyze(resume, excerpts):
        claims = [CitedClaim("Kubernetes", e["job_id"], None, e["text"], None) for e in excerpts]
        return GapAnalysisResult([], claims, "", "{}")

    retrieved = [{"job_id": str(i), "title": f"Job {i}", "company_name": "Co", "job_posting_url": None,
                  "text": "Requires Kubernetes."} for i in range(3)]
    report = build_report(retrieved, "banana bread", analyze, lambda _id: [])
    assert report.headline == NO_EVIDENCE_HEADLINE and report.best is None
    assert all(j.tier == "potential" for j in report.jobs)
