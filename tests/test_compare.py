"""Comparing two resume versions against the same postings (no model)."""

from resumematch.generation.compare import changed_lines, check, compare

OLD = """Maria Delgado, RN
Staff nurse, medical ICU, 6 years.
Certifications: BLS
Charge nurse for a 20-bed unit."""
NEW = """Maria Delgado, RN
Staff nurse, medical ICU, 6 years.
Certifications: BLS, ACLS, CCRN
Managed ECMO circuits for post-operative
cardiac patients."""
JOBS = [{"job_id": "7", "title": "RN CICU", "company": "Acme",
         "requirements": ["ACLS certification", "ECMO experience", "Charge nurse experience",
                          "Strong communication skills", "BLS certification", "ACLS certification"]}]


def test_one_line_with_half_the_key_terms_is_shown_with_that_line_as_evidence():
    assert check("ACLS certification", NEW) == ("shown", "Certifications: BLS, ACLS, CCRN")
    assert check("ACLS certification", OLD)[0] == "not_found"


def test_wrapped_lines_are_read_together():
    status, line = check("ECMO for cardiac patients", NEW)
    assert status == "shown" and "ECMO" in line


def test_changes_between_versions_better_first():
    result = compare(OLD, NEW, JOBS)
    job = result.jobs[0]
    assert job.total == 4                       # soft trait and the duplicate are skipped
    assert (job.shown_before, job.shown_after) == (2, 3)
    assert [(c.requirement, c.before, c.after) for c in job.changes] == [
        ("ACLS certification", "not_found", "shown"),
        ("ECMO experience", "not_found", "shown"),
        ("Charge nurse experience", "shown", "not_found")]
    assert job.changes[0].evidence_after == "Certifications: BLS, ACLS, CCRN"
    assert job.changes[2].evidence_before == "Charge nurse for a 20-bed unit."


def test_summary_sentences():
    assert compare(OLD, NEW, JOBS).summary() == "Your revised resume shows 3 of 4 requirements in this posting, up from 2."
    assert compare(OLD, OLD, JOBS).summary().startswith("Both versions show 2 of 4")
    assert "nothing to compare" in compare(OLD, NEW, [{"job_id": "1", "requirements": []}]).summary()


def test_changed_lines_ignore_spacing_and_case():
    added, removed = changed_lines("A line\nSame  LINE\n", "same line\nNew line\n")
    assert added == ["New line"] and removed == ["A line"]


def test_postings_without_requirements_are_not_counted():
    jobs = JOBS + [{"job_id": "8", "title": "Not checked", "requirements": []}]
    assert compare(OLD, NEW, jobs).summary().endswith("in this posting, up from 2.")
