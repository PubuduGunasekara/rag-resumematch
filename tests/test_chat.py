"""Ask your report: pieces, routing, and the checks on model sentences
(fake embedder and model; no index, no Ollama)."""

import json

from resumematch.chat import ask
from resumematch.chat.ask import Piece, ReportPosting, Sentence

RESUME = """Maria Delgado, RN
Six years of acute care in intensive care units at a 400-bed Level I trauma center.
ACLS, BLS, PALS
Precepted 14 new graduate nurses and led the sepsis bundle
compliance initiative."""
POSTS = [
    ReportPosting("1", "RN CICU", "Boston Children's", "https://x/1",
                  "Pediatric ICU experience preferred. Level II trauma center. ACLS required.",
                  {"location": "Boston, MA", "work_type": "Full-time", "level": "Entry level", "remote": False, "salary": None},
                  True, [("ACLS", "shown", "ACLS, BLS, PALS"), ("Teaching hospital", "not_found", None)]),
    ReportPosting("2", "RN ICU", "Southcoast", "https://x/2", "Six months of ICU experience required.",
                  {"location": "New Bedford, MA", "work_type": "Part-time", "level": None, "remote": True,
                   "salary": "$40 to $50 an hour"}),
]


def embed(texts):
    """Similarity = shared words with the question (first text), scaled to look like cosine."""
    q = set(texts[0].lower().split())
    return [[1.0, 0.0]] + [[min(1.0, 0.5 + 0.1 * len(q & set(t.lower().split()))), 0.0] for t in texts[1:]]


class Model:
    def __init__(self, sentences):
        self.sentences, self.prompts = sentences, []

    def generate(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return json.dumps({"sentences": self.sentences, "not_in_pieces": False})


def test_resume_pieces_join_wrapped_lines():
    pieces = ask.resume_pieces(RESUME)
    assert "Precepted 14 new graduate nurses and led the sepsis bundle compliance initiative." in pieces
    assert "ACLS, BLS, PALS" in pieces and all(len(p.split()) >= 3 for p in pieces)


def test_facts_state_absences_and_checks_summarize_the_report():
    assert "salary: not listed" in ask.facts_piece(POSTS[0].details) and "remote work: allowed" in ask.facts_piece(POSTS[1].details)
    check = ask.checks_piece(POSTS[0].checks)
    assert 'shown on your resume: ACLS (your resume: "ACLS, BLS, PALS")' in check and "not found on your resume: Teaching hospital" in check


def test_fit_questions_are_answered_from_the_report_checks_without_the_model():
    model = Model([])
    a = ask.answer("Why is my closest match a good fit?", RESUME, POSTS, model, embed)
    assert a.by_code and model.prompts == []
    text = " ".join(s.text for s in a.shown)
    assert "Your closest match is RN CICU at Boston Children's." in text
    assert "ACLS" in text and "It doesn't yet show Teaching hospital." in text
    assert all(s.sources for s in a.shown)


def test_facts_questions_are_answered_from_the_data_without_the_model():
    model = Model([])
    a = ask.answer("Which of these jobs list a salary?", RESUME, POSTS, model, embed)
    assert a.by_code and model.prompts == []
    assert [s.text for s in a.shown] == [
        "RN CICU at Boston Children's: Boston, MA, Full-time, Entry level, remote work not mentioned, salary not listed.",
        "RN ICU at Southcoast: New Bedford, MA, Part-time, level not listed, remote work allowed, salary $40 to $50 an hour."]


def test_off_topic_questions_are_declined_without_the_model():
    model = Model([])
    a = ask.answer("Tell me a joke", RESUME, POSTS, model, embed)
    assert a.no_answer and a.by_code and model.prompts == [] and a.sentences[0].text == ask.OFF_TOPIC


def test_a_question_with_its_key_word_in_the_postings_is_on_topic_even_if_far():
    model = Model([{"text": "RN CICU at Boston Children's prefers pediatric ICU experience.", "sources": [1, 2, 3, 4, 5, 6, 7][:4]}])
    ask.answer("Pediatric?", RESUME, POSTS, model, lambda texts: [[1.0, 0.0]] + [[0.1, 0.0] for _ in texts[1:]])
    assert len(model.prompts) == 1


def pieces():
    return {1: Piece(1, "resume", "Your resume", "Six years in intensive care units at a 400-bed Level I trauma center."),
            2: Piece(2, "posting", "Posting 1", "Pediatric ICU experience preferred. Level II trauma center.", name="RN CICU at Boston Children's"),
            3: Piece(3, "check", "Posting 1", "not found on your resume: Teaching hospital.", name="RN CICU at Boston Children's")}


def checked(text, sources, question="What does the job need?"):
    s = Sentence(text, sources)
    ask.check_sentence(s, pieces(), question, RESUME, "RN CICU at Boston Children's RN ICU at Southcoast")
    return s.issue


def test_sentence_checks():
    assert checked("Pediatric ICU experience is preferred.", [2]) is None
    assert checked("Pediatric ICU experience is preferred.", []) == "cites nothing"
    assert checked("It needs 5 years of pediatric ICU experience.", [2]).startswith("states a number")
    assert checked("It needs Epic charting.", [2]).startswith("mentions Epic")
    assert checked("The weather is lovely.", [2]) == "doesn't use what it cites"
    assert checked("RN ICU at Southcoast is another posting with trauma center work.", [2]) is None   # a report posting's name


def test_claims_about_the_candidate_cannot_borrow_from_a_posting():
    assert checked("You have experience in a Level II trauma center.", [1, 2]) == "says you have ii, which your resume doesn't show"
    assert "teaching" in checked("You have experience in a teaching hospital.", [3])
    assert checked("You have experience in a Level I trauma center.", [1]) is None
    assert checked("You do not have pediatric experience on your resume.", [1, 2]) is None    # an absence, not a claim


def test_citations_in_the_text_are_removed_and_postings_are_named():
    names = {1: "RN CICU at Boston Children's", 2: "RN ICU at Southcoast"}
    raw = json.dumps({"not_in_pieces": False, "sentences": [
        {"text": "Postings 1 and 2 mention ICU experience (piece 2).", "sources": [2]},
        {"text": "Postings 1 and 2 mention ICU experience (piece 2).", "sources": [2]},
        {"text": ": Pediatric ICU experience is preferred [2].", "sources": [2]},
        {"text": "not_in_pieces", "sources": [2]}]})
    out, _flag = ask.parse(raw, list(pieces().values()), "q", RESUME, names)
    assert [s.text for s in out] == [
        "RN CICU at Boston Children's and RN ICU at Southcoast mention ICU experience.",
        "Pediatric ICU experience is preferred."]


def test_model_answer_end_to_end_and_history_is_one_turn():
    model = Model([{"text": "Pediatric ICU experience is preferred.", "sources": [3, 4, 5, 6]},
                   {"text": "You have experience in a teaching hospital.", "sources": [1]}])
    history = [("first q", "first a"), ("second q", "second a")]
    a = ask.answer("What does the pediatric ICU posting ask for?", RESUME, POSTS, model, embed, history=history)
    assert not a.by_code and len(a.shown) == 1 and a.sentences[1].issue
    assert "second q" in model.prompts[0] and "first q" not in model.prompts[0]
    assert "Posting 1 (your closest match): RN CICU at Boston Children's" in model.prompts[0]


def test_wider_questions_add_search_results_from_other_postings():
    hits = [{"job_id": "1", "title": "dup", "company_name": "x", "text": "t", "job_posting_url": None},
            {"job_id": "9", "title": "ICU Nurse", "company_name": "Mercy", "text": "CCRN preferred.", "job_posting_url": "https://x/9"}]
    found, _ = ask.gather_pieces("What do other ICU postings usually ask for?", RESUME, POSTS, embed, lambda v: hits)
    search = [p for p in found if p.kind == "search"]
    assert [p.name for p in search] == ["ICU Nurse at Mercy"]
    found, _ = ask.gather_pieces("What does this posting ask for?", RESUME, POSTS, embed, lambda v: hits)
    assert not [p for p in found if p.kind == "search"]


def test_nothing_usable_is_a_plain_no_answer():
    a = ask.answer("What does the pediatric ICU posting ask for?", RESUME, POSTS,
                   Model([{"text": "It needs Epic.", "sources": [1]}]), embed)
    assert a.no_answer and not a.shown
