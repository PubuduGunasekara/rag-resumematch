"""Bullet rewrites and cover letters: extraction, the grounding checks, and
parsing (fake model, no Ollama)."""

import json

import pytest

from resumematch.generation import tailor
from resumematch.generation.tailor import (
    Evidence,
    LetterSentence,
    Rewrite,
    arrange_letter,
    check_rewrite,
    check_sentence,
    resume_bullets,
    unsupported_words,
    verify_evidence,
)

RESUME = """Daniela Marquez
Backend Engineer
SUMMARY
Backend engineer with 6 years building payment services in Java and Spring Boot. Event-driven design.
TECHNICAL SKILLS
Java 17, Kotlin, SQL, Kubernetes, Redis
EXPERIENCE
Software Engineer II, Brightlane Payments Mar 2021 - Present
Moved nightly batch reconciliation to Kafka consumers, cutting end-of-day processing from 4 hours to 35
minutes.
Led the on-call rotation for 5 engineers and wrote runbooks that reduced mean time to recovery by 40%.
PROJECTS
Ledger Kata (Java 17, Spring Boot, PostgreSQL)
EDUCATION
B.S. Computer Science, 2017
"""
BULLET = "Moved nightly batch reconciliation to Kafka consumers, cutting end-of-day processing from 4 hours to 35 minutes."


def test_bullets_join_wrapped_lines_and_skip_job_and_title_lines():
    bullets = resume_bullets(RESUME)
    assert bullets[0] == BULLET                      # the wrapped "minutes." is joined
    assert not any("Brightlane Payments Mar 2021" in b for b in bullets)
    assert not any("Ledger Kata" in b for b in bullets)
    assert not any("Kotlin" in b for b in bullets)   # skills section is not a bullet section


@pytest.mark.parametrize("text, issue", [
    ("Migrated nightly batch reconciliation to Kafka consumers, reducing end-of-day processing from 4 hours to 35 minutes.", None),
    ("Re-architected nightly reconciliation as event-driven Kafka consumers, cutting processing from 4 hours to 35 minutes.", None),
    ("Moved reconciliation to Kafka consumers, cutting processing by 90%.", "adds a number"),
    ("Moved reconciliation to Kafka consumers on Kubernetes, cutting processing from 4 hours to 35 minutes.", "this bullet doesn't mention"),
    ("Moved reconciliation to Kafka and Airflow, cutting processing from 4 hours to 35 minutes.", "your resume doesn't mention"),
    ("Moved reconciliation to Kafka consumers, demonstrating expertise, from 4 hours to 35 minutes.", "expertise"),
    ("Wrote runbooks for the team that helped the team.", "changes what the bullet says"),
    (BULLET, "no change"),
])
def test_rewrite_checks(text, issue):
    rw = Rewrite(BULLET, text, "Kafka")
    check_rewrite(rw, RESUME)
    assert (rw.issue is None) if issue is None else (issue in rw.issue)


def test_numbers_glued_to_letters_are_not_words():
    assert unsupported_words("p99 under 180 ms at $85M and 99th percentile", "p99 180 ms 85 99 percentile") == []


def test_letter_sentences_may_name_the_job_but_not_claim_its_tools():
    posting = "Work on the Clinical Data Hub with Snowflake and Kafka. 3+ years required."
    names = "Lead Java Developer NLB Services"
    ok = LetterSentence("I would be glad to help NLB Services build the Clinical Data Hub.", 1, 0)
    claim = LetterSentence("My experience with Snowflake makes me a fit.", 2, 0)
    number = LetterSentence("I have 9 years of Java experience.", 2, 0)
    for s in (ok, claim, number):
        check_sentence(s, [], RESUME, posting, names)
    assert ok.issue is None
    assert "Snowflake" in claim.issue
    assert "9" in number.issue


def test_letter_sentence_must_talk_about_the_match_it_cites():
    ev = [Evidence("Experience with Kafka", "Kafka consumers")]
    good = LetterSentence("I moved batch work to Kafka consumers.", 2, 1)
    off = LetterSentence("I enjoy solving hard problems.", 2, 1)
    for s in (good, off):
        check_sentence(s, ev, RESUME, "Kafka experience", "")
    assert good.issue is None and off.issue == "cites a match it doesn't talk about"


def test_evidence_from_the_browser_is_checked_again():
    items = [Evidence("Experience with Kafka", "Kafka consumers"),
             Evidence("Experience with Kafka", "Kafka streams at scale"),     # quote not in the resume
             Evidence("Snowflake", "Kafka consumers"),                       # requirement not in the posting
             Evidence("", "Led the on-call rotation for 5 engineers")]       # a plain resume line
    kept = verify_evidence(items, RESUME, "We need Kafka experience.")
    assert [e.resume_quote for e in kept] == ["Kafka consumers", "Led the on-call rotation for 5 engineers"]


def test_letter_is_arranged_without_repeats_or_extra_filler():
    s = [LetterSentence("I am excited to apply for this role.", 1, 0),
         LetterSentence("I am excited to apply for this role at NLB.", 1, 0),   # repeat
         LetterSentence("I moved reconciliation to Kafka.", 1, 1),
         LetterSentence("I bring energy.", 1, 0),
         LetterSentence("I wrote runbooks.", 2, 2),
         LetterSentence("I am a team player.", 3, 0),
         LetterSentence("Thank you for your time.", 3, 0),
         LetterSentence("Sincerely, Daniela.", 3, 0),
         LetterSentence("Bad one.", 3, 0, issue="x")]
    paragraphs, left_out = arrange_letter(s)
    assert [[x.text for x in p] for p in paragraphs] == [
        ["I am excited to apply for this role."],
        ["I moved reconciliation to Kafka.", "I wrote runbooks."],
        ["Thank you for your time.", "Sincerely, Daniela."]]
    assert left_out == 3   # the repeat, "I bring energy." and "I am a team player."; rejected ones are counted apart


class FakeClient:
    def __init__(self, response):
        self.response, self.prompts = response, []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        assert kwargs["schema"]["required"]
        return json.dumps(self.response)


def test_suggest_rewrites_end_to_end():
    client = FakeClient({"rewrites": [
        {"bullet": 1, "text": "Migrated nightly batch reconciliation to Kafka consumers, cutting end-of-day processing from 4 hours to 35 minutes.", "requirement": "Kafka"},
        {"bullet": 2, "text": "Led the on-call rotation for 5 engineers on Snowflake, reducing mean time to recovery by 40%.", "requirement": "on-call"},
        {"bullet": 9, "text": "out of range", "requirement": "x"}]})
    result = tailor.suggest_rewrites(RESUME, "Backend Engineer", "Acme", "Kafka on-call runbooks Java", client)
    assert [r.issue is None for r in result.rewrites] == [True, False]
    assert "Snowflake" in result.rewrites[1].issue
    assert "[1] Moved nightly" in client.prompts[0]


def test_draft_letter_adds_resume_lines_as_evidence():
    client = FakeClient({"sentences": [
        {"text": "I am applying for the Backend Engineer role at Acme.", "paragraph": 1, "evidence": 0},
        {"text": "I moved nightly reconciliation to Kafka consumers.", "paragraph": 2, "evidence": 1},
        {"text": "I also know Snowflake well from my experience.", "paragraph": 2, "evidence": 0},
        {"text": "Thank you for your time.", "paragraph": 3, "evidence": 0}]})
    result = tailor.draft_letter(RESUME, "Backend Engineer", "Acme", "Kafka experience and on-call runbooks",
                                 [Evidence("Kafka experience", "Kafka consumers")], client)
    assert result.evidence[0].resume_quote == "Kafka consumers" and len(result.evidence) >= 2
    assert [s.issue is None for s in result.sentences] == [True, True, False, True]


def test_unusable_model_output_raises_after_retries():
    class Broken:
        def generate(self, prompt, **kwargs):
            return "not json"
    with pytest.raises(ValueError):
        tailor.suggest_rewrites(RESUME, "Backend Engineer", None, "Kafka", Broken())
