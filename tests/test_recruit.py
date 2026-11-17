"""Recruiter mode: requirement checks, ranking, and blind redaction (fake model)."""

import json

from resumematch.generation.compare import check
from resumematch.recruit.blind import NAME_MARK, blind, blind_line, candidate_name
from resumematch.recruit.rank import check_requirements, extract_requirements, rank_candidates

JOB = """Senior Backend Engineer

Requirements:
- 5+ years of Java experience
- Kafka or RabbitMQ in production
- Bachelor's degree in Computer Science
- Strong communication skills

Preferred qualifications:
- Kubernetes
- Experience with Terraform is a plus

We offer 401(k) matching and dental insurance."""

ANA = """Ana Silva
Austin, TX | ana@example.com | (555) 010-0100 | linkedin.com/in/ana
EXPERIENCE
Built Java services on Kafka for payments; Ana led the on-call rotation.
Deployed to Kubernetes.
EDUCATION
B.S. Computer Science, University of Texas at Austin, 2015"""
BEN = """Ben Okoro
Chicago, IL | ben@example.com
EXPERIENCE
Python data pipelines on Spark.
Java for two years."""


def test_requirements_keep_only_what_the_job_says_with_importance_from_its_wording():
    reqs = {r.text: r for r in check_requirements(
        ["5+ years of Java experience", "Kafka or RabbitMQ", "Bachelor's degree in Computer Science",
         "Strong communication skills", "Kubernetes", "Terraform", "401(k) matching", "Rust", "Experience with"], JOB)}
    assert reqs["5+ years of Java experience"].importance == "required"        # heading "Requirements:"
    assert reqs["Kubernetes"].importance == "preferred"                        # heading "Preferred qualifications:"
    assert reqs["Terraform"].importance == "preferred"                         # "is a plus" in its sentence
    assert reqs["Strong communication skills"].soft
    assert reqs["401(k) matching"].issue == "a benefit, not a requirement"
    assert reqs["Rust"].issue == "not in the job description"
    assert reqs["Experience with"].issue in ("too vague to check", "an incomplete phrase")


def test_amounts_are_ignored_and_either_alternative_counts():
    assert check("5+ years of Java experience", "Java for two years.")[0] == "shown"
    assert check("Kafka or RabbitMQ in production", "RabbitMQ consumers in production.")[0] == "shown"
    assert check("Bash/Python scripting", "Python scripting for reports.")[0] == "shown"


def test_ranking_uses_the_same_rule_for_everyone_and_orders_by_required_first():
    reqs = check_requirements(["5+ years of Java experience", "Kafka or RabbitMQ", "Bachelor's degree in Computer Science",
                               "Kubernetes", "Strong communication skills"], JOB)
    ranked = rank_candidates(reqs, [("ben.pdf", BEN, "Ben Okoro"), ("ana.pdf", ANA, "Ana Silva")])
    assert [c.file_name for c in ranked] == ["ana.pdf", "ben.pdf"]
    ana, ben = ranked
    assert ana.count("required") == 3 and ana.count("preferred") == 1
    assert ben.count("required") == 1
    soft = [r for r, _s, _e in ana.checklist if r.soft]
    assert [r.text for r in soft] == ["Strong communication skills"]   # listed, never counted
    assert ana.count("required") + ana.count("preferred") == sum(s == "shown" for r, s, _e in ana.checklist if not r.soft)


def test_ties_keep_upload_order():
    reqs = check_requirements(["Kubernetes"], JOB)
    ranked = rank_candidates(reqs, [("b.pdf", "nothing", None), ("a.pdf", "nothing", None)])
    assert [c.file_name for c in ranked] == ["b.pdf", "a.pdf"]


def test_blind_removes_identity_but_keeps_the_work():
    assert candidate_name(ANA) == "Ana Silva" and candidate_name("Senior Engineer\nx") == "Senior Engineer"
    assert candidate_name("EXPERIENCE: lots") is None
    text = blind(ANA)
    for leak in ("Ana", "Silva", "Austin, TX", "ana@example.com", "555", "linkedin", "University of Texas", "2015"):
        assert leak not in text, leak
    assert "Built Java services on Kafka" in text and f"{NAME_MARK} led the on-call rotation" in text
    assert "B.S. Computer Science" in text


def test_blind_line_for_evidence():
    assert blind_line("Ana led the on-call rotation", "Ana Silva") == f"{NAME_MARK} led the on-call rotation"
    assert "University" not in blind_line("B.S., University of Texas, 2015", None)
    assert blind_line(None, "x") is None


def test_extraction_with_a_fake_model():
    class Fake:
        def generate(self, prompt, **kwargs):
            assert "JOB DESCRIPTION" in prompt and kwargs["schema"]["required"] == ["requirements"]
            return json.dumps({"requirements": ["Kubernetes", "Kubernetes", "Dental insurance"]})
    reqs = extract_requirements(JOB, Fake())
    assert [r.text for r in reqs] == ["Kubernetes", "Dental insurance"]
    assert reqs[1].issue == "a benefit, not a requirement"


def test_resumes_are_checked_against_the_employers_line_not_the_models_phrase():
    job = JOB.replace("- Kafka or RabbitMQ in production", "- PostgreSQL or another relational database")
    reqs = check_requirements(["PostgreSQL relational database"], job)   # the model dropped the "or"
    assert reqs[0].source == "PostgreSQL or another relational database"
    ranked = rank_candidates(reqs, [("a.pdf", "Stores: PostgreSQL, Redis", None)])
    assert ranked[0].checklist[0][1] == "shown"


def test_section_comes_from_the_best_matching_line_and_caps_lists_are_not_headings():
    job = "Senior Backend Engineer (Java)\n\nRequirements:\n- 5+ years of backend development in Java\n\n" \
          "Preferred qualifications:\n- AWS (EC2, S3, SQS)\n- Kubernetes\n" + "Filler text about the team. " * 10
    reqs = {r.text: r.importance for r in check_requirements(["5+ years Java backend development", "AWS EC2 S3 SQS"], job)}
    assert reqs == {"5+ years Java backend development": "required", "AWS EC2 S3 SQS": "preferred"}
    source = check_requirements(["5+ years Java backend development"], job)[0].source
    assert source == "5+ years of backend development in Java"       # not the title line
