"""Tests for sentence splitting and the cross-company boilerplate rule."""

from resumematch.preprocess.boilerplate import find_boilerplate, is_boilerplate, split_sentences

EEO = "Reasonable accommodations may be made to enable individuals with disabilities to perform the essential functions."
DUTY = "Assists in the preparation of a work schedule for all nursing personnel on the unit."
DEGREE = "Bachelor's degree in computer science, information technology, or related field preferred."
BENEFITS = "We offer a competitive salary and comprehensive benefits package to every employee here."


def test_split_handles_newlines_bullets_and_glued_sentences():
    text = "About us\n• Build APIs in Go.• Write tests.Deploy weekly to production.\n\nApply now!"
    assert split_sentences(text) == ["About us", "Build APIs in Go.", "Write tests.", "Deploy weekly to production.", "Apply now!"]


def corpus():
    rows = []
    for i in range(30):  # 30 postings across 30 companies share EEO + benefits + degree lines
        rows.append((f"company{i}", f"Unique duty number {i} for this role. {EEO} {BENEFITS} {DEGREE}"))
    for i in range(25):  # one company repeats its own duty sentence 25 times
        rows.append(("hospital", f"{DUTY} Unique hospital line {i} here."))
    return rows


def test_cross_company_repeats_are_boilerplate():
    frequent = find_boilerplate(corpus(), min_postings=20, min_companies=10)
    assert EEO.lower() in frequent
    assert BENEFITS.lower() in frequent


def test_single_company_repeats_are_kept():
    frequent = find_boilerplate(corpus(), min_postings=20, min_companies=10)
    assert DUTY.lower() not in frequent


def test_frequent_requirement_sentences_are_exempt():
    frequent = find_boilerplate(corpus(), min_postings=20, min_companies=10)
    assert DEGREE.lower() not in frequent
    assert not is_boilerplate(DEGREE, {DEGREE.lower()})


def test_legal_wording_is_boilerplate_even_when_unseen():
    assert is_boilerplate("Acme Corp is an equal opportunity employer and welcomes all applicants.", set())
    assert not is_boilerplate("Equal opportunity.", set())  # too short to judge


def test_short_legal_statement_is_caught():
    assert is_boilerplate("Delaware North is an equal opportunity employer.", set())


def test_accommodation_text_mentioning_required_is_still_legal():
    text = "Information will be used only to the extent required to provide needed reasonable accommodations."
    assert is_boilerplate(text, set())


def test_physical_requirements_in_template_wording_are_kept():
    # Seen in the detected list before this cue existed: real requirements.
    lift = "The employee must occasionally lift and/or move up to 25 pounds."
    duties = "Responsibilities include interviewing, hiring, and training employees; planning and directing work."
    rows = [(f"c{i}", f"{lift} {duties} Line {i}.") for i in range(30)]
    frequent = find_boilerplate(rows, min_postings=20, min_companies=10)
    assert lift.lower() not in frequent and duties.lower() not in frequent
