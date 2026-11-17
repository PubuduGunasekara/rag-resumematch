"""Interview prep: statuses recomputed from the resume, question checks,
template fallback (fake model, no Ollama)."""

import json

from resumematch.generation.interview import (
    PrepItem,
    ReportRequirement,
    check_question,
    prepare,
    prepare_items,
    template_question,
)

RESUME = """Maria Delgado, RN
Six years in intensive care. ACLS, BLS, PALS.
Charge nurse for a 20-bed unit; coordinate admissions and bed flow.
Docker and Kubernetes at a hospital IT rotation."""
POSTING = ("Requirements: ACLS certification. ECMO experience required. Experience with Docker and microservices. "
           "Strong communication skills. Charge nurse experience preferred. Data reporting.")


def test_statuses_are_recomputed_from_the_resume():
    reqs = [ReportRequirement("ACLS certification", "ACLS, BLS"),
            ReportRequirement("ECMO experience", "ECMO for 3 years"),          # quote not in the resume
            ReportRequirement("Docker and microservices"),
            ReportRequirement("Strong communication skills"),
            ReportRequirement("Telemetry experience"),                          # not in the posting
            ReportRequirement("Data reporting")]
    items = {i.requirement: i for i in prepare_items(reqs, RESUME, POSTING)}
    assert items["ACLS certification"].status == "shown"
    assert items["ECMO experience"].status == "not_found"
    assert items["Docker and microservices"].status == "partly" and items["Docker and microservices"].partly == ["docker"]
    assert items["Strong communication skills"].status == "soft"
    assert "Telemetry experience" not in items
    assert items["Data reporting"].status == "not_found"     # "data" alone is too general to point at
    order = [i.status for i in prepare_items(reqs, RESUME, POSTING)]
    assert order == sorted(order, key=["not_found", "partly", "shown", "soft"].index)


def test_hints_come_from_code_not_the_model():
    shown = PrepItem("ACLS", "shown", resume_quote="ACLS, BLS")
    assert '"ACLS, BLS"' in shown.hint
    long = PrepItem("x", "shown", resume_quote="word " * 80)
    assert long.hint.count("word") < 40 and "..." in long.hint
    assert "credential" in PrepItem("RN license", "not_found", credential=True).hint
    assert "doesn't use these words" in PrepItem("ECMO", "not_found").hint


def test_question_checks():
    item = PrepItem("ECMO experience", "not_found")
    src = POSTING + RESUME
    assert check_question("Tell me about ECMO.", item, src) == "not a question"
    assert check_question("How have you used Epic charting?", item, src).startswith("mentions Epic")
    assert check_question("Why do you want this job?", item, src) == "doesn't ask about this requirement"
    assert check_question("What is your experience with ECMO?", item, src) is None


def test_templates_read_naturally():
    assert template_question(PrepItem("ECMO experience", "not_found")) == "Can you tell me about your ECMO experience?"
    assert template_question(PrepItem("Kafka", "partly")) == "Can you tell me about your experience with Kafka?"
    assert template_question(PrepItem("Cloud technologies", "partly")) == "Can you tell me about your experience with cloud technologies?"
    assert template_question(PrepItem("Strong communication skills", "soft")).startswith("Can you give an example")


class FakeClient:
    def __init__(self, response):
        self.response = response

    def generate(self, prompt, **kwargs):
        return self.response if isinstance(self.response, str) else json.dumps(self.response)


REQS = [ReportRequirement("ACLS certification", "ACLS, BLS"), ReportRequirement("ECMO experience")]


def test_model_questions_used_when_they_pass_and_templates_fill_the_rest():
    client = FakeClient({"questions": [
        {"requirement": 1, "question": "How have you managed a patient on ECMO?"},
        {"requirement": 2, "question": "Have you used Epic for ACLS documentation?"},   # Epic is nowhere
        {"requirement": 0, "question": "Why do you want to work in this unit?"}]})
    prep = prepare(RESUME, "RN ICU", "Acme", POSTING, REQS, client)
    by = {i.requirement: i for i in prep.items}
    assert by["ECMO experience"].question == "How have you managed a patient on ECMO?" and by["ECMO experience"].generated
    assert not by["ACLS certification"].generated and by["ACLS certification"].issue.startswith("mentions Epic")
    assert [g.question for g in prep.general] == ["Why do you want to work in this unit?"]


def test_unusable_model_output_still_gives_every_requirement_a_question():
    prep = prepare(RESUME, "RN ICU", None, POSTING, REQS, FakeClient("not json"))
    assert len(prep.items) == 2 and all(i.question and not i.generated for i in prep.items)


def test_nothing_checkable_means_nothing_to_prepare():
    prep = prepare(RESUME, "RN ICU", None, POSTING, [ReportRequirement("Telemetry")], FakeClient("{}"))
    assert prep.items == [] and prep.general == []
