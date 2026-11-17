"""Role report: title matching, term selection and resume matching, with
a fake store (no index needed)."""

from collections import Counter

from resumematch.market import role_profile as rp


class FakeStore:
    def __init__(self, postings: dict[str, str]):
        self.postings = postings

    def get_postings_text(self, job_ids):
        return {j: [{"text": self.postings[j], "section": "description", "job_id": j}] for j in job_ids if j in self.postings}

    def iter_metadata(self):
        for j in self.postings:
            yield {"job_id": j, "title": f"Title {j}", "company_name": "Co", "job_posting_url": f"https://x/{j}"}


TITLES = {
    "1": ["Senior Data Engineer II", "A", None], "2": ["Data Engineers", "B", None],
    "3": ["Data Engineering Manager", "C", None], "4": ["Big Data Engineer", "D", None],
    "5": ["Registered Nurse - ICU", "E", None], "6": [None, None, None],
}


def test_title_match_is_whole_words_with_plural():
    assert rp.match_postings(TITLES, "Data Engineer") == ["1", "2", "4"]
    assert rp.match_postings(TITLES, "  data   ENGINEER ") == ["1", "2", "4"]
    assert rp.match_postings(TITLES, "") == []


def test_suggestions_are_counted_by_normalized_title():
    titles = {"1": ["Data Engineer", None, None], "2": ["data engineer", None, None], "3": ["Senior Data Engineer", None, None]}
    out = rp.suggest_titles(titles, "data eng")
    assert out[0]["postings"] == 2 and out[0]["title"] in ("Data Engineer", "data engineer")
    assert rp.suggest_titles(titles, "d") == []


def test_candidate_terms_skip_filler_numbers_and_boundaries():
    terms = rp.candidate_terms("3+ years of experience with SQL and Apache Spark. Strong team player")
    assert terms["sql"] == "SQL" and terms["apache spark"] == "Apache Spark"
    assert "years" not in terms and "experience with sql" not in terms
    assert "spark strong" not in terms          # never crosses a sentence boundary
    assert "team" not in terms                  # generic single word


def test_select_terms_contrasts_with_background_and_prefers_longer_forms():
    df = Counter({"sql": 9, "journal": 6, "journal entries": 5, "team player": 9, "data engineer": 10,
                  "patient": 5, "patients": 4})
    surface = {t: Counter({t: 1}) for t in df}
    background = {"postings": 100, "df": {"team player": 80}}
    terms = [t.term for t in rp.select_terms(df, surface, 10, background, "Data Engineer")]
    assert "sql" in terms and "journal entries" in terms
    assert "journal" not in terms               # mostly part of "journal entries"
    assert "team player" not in terms           # common everywhere
    assert "data engineer" not in terms         # only the role's own words
    assert terms.count("patient") + terms.count("patients") == 1


def test_resume_matching_handles_plurals_abbreviations_and_degrees():
    resume = "B.S. in Computer Science. BSN, Bachelor of Science in Nursing. RN. Reconciliation of accounts."
    assert rp.term_on_resume("nursing", resume)           # not lost to BSN folding
    assert rp.term_on_resume("Registered Nurse", resume)  # matches "RN"
    assert rp.term_on_resume("reconciliations", resume)
    assert rp.term_on_resume("degree in Computer Science", resume)
    assert rp.term_on_resume("health care", "Ten years in healthcare.")
    assert not rp.term_on_resume("Spark", resume)


def test_profile_end_to_end_with_resume(tmp_path):
    postings = {str(i): "Requirements: SQL and Airflow pipelines. Must know Python." for i in range(20)}
    postings.update({str(i): "We value a positive attitude and teamwork in our office." for i in range(20, 120)})  # background mostly other jobs
    titles = {j: ["Data Engineer" if int(j) < 20 else "Office Clerk", "Co", f"https://x/{j}"] for j in postings}
    store = FakeStore(postings)
    background = rp.load_background(store, titles, tmp_path)
    profile = rp.build_role_profile(store, titles, background, "data engineer", resume_text="Python and SQL daily.")
    by_term = {t.term: t for t in profile.terms}
    assert profile.matched == 20 and by_term["SQL"].shown is True and by_term["Airflow pipelines"].shown is False
    assert by_term["SQL"].examples[0]["url"].startswith("https://x/")
    headline, summary = rp.summarize(profile, "Data Engineer")
    assert "Airflow pipelines" in summary and headline.startswith("How your resume compares")


def test_too_few_postings_says_so():
    profile = rp.build_role_profile(FakeStore({}), TITLES, {"postings": 1, "df": {}}, "registered nurse")
    assert profile.terms == [] and rp.summarize(profile, "Registered Nurse")[0].startswith("Too few postings")


def test_titles_are_cached(tmp_path):
    store = FakeStore({"7": "x"})
    assert rp.load_titles(store, tmp_path) == {"7": ["Title 7", "Co", "https://x/7"]}
    store.postings = {}
    assert "7" in rp.load_titles(store, tmp_path)   # read from the cache, not the store


def test_facts_from_posting_details():
    details = {"1": {"location": "Boston, MA", "work_type": "Full-time", "remote": True, "salary": "$1"},
               "2": {"location": "Boston, MA", "work_type": "Contract", "remote": False, "salary": None}}
    facts = rp.role_facts(["1", "2", "9"], details)
    assert facts["remote_share"] == 0.5 and facts["top_locations"][0] == {"name": "Boston, MA", "postings": 2}
    assert facts["salary_listed_share"] == 0.5
