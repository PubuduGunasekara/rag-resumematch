"""Search filters: parsing, matching, salary conversion, and how retrieval
applies them (no index needed)."""

from resumematch.preprocess.resume_sections import ResumeSection
from resumematch.retrieval import fusion
from resumematch.retrieval.filters import (
    SearchFilters,
    allowed_ids,
    annual_salary,
    filter_options,
    parse_filters,
    state_of,
)

DETAILS = {
    "1": {"location": "Boston, MA", "work_type": "Full-time", "level": "Entry level", "remote": False,
          "salary": "$60,000 to $80,000 a year"},
    "2": {"location": "United States", "work_type": "Contract", "level": None, "remote": True,
          "salary": "$40 to $50 an hour"},
    "3": {"location": "Austin, TX", "work_type": "Full-time", "level": "Mid-Senior level", "remote": True,
          "salary": None},
    "4": {"location": "Remote", "work_type": "Part-time", "level": "Entry level", "remote": False,
          "salary": "EUR 50,000 a year"},
}


def test_salary_is_converted_to_a_yearly_range():
    assert annual_salary("$60,000 to $80,000 a year") == (60000, 80000)
    assert annual_salary("$40 to $50 an hour") == (83200, 104000)
    assert annual_salary("$5,000 a month") == (60000, 60000)
    assert annual_salary("$2,000 to $2,500 every two weeks") == (52000, 65000)
    assert annual_salary("EUR 50,000 a year") is None and annual_salary(None) is None


def test_state_comes_from_city_state_locations_only():
    assert state_of("Boston, MA") == "MA"
    assert state_of("United States") is None and state_of("Remote") is None and state_of(None) is None


def test_each_filter_narrows_the_postings():
    assert allowed_ids(DETAILS, SearchFilters()) == {"1", "2", "3", "4"}
    assert allowed_ids(DETAILS, SearchFilters(remote_only=True)) == {"2", "3"}
    assert allowed_ids(DETAILS, SearchFilters(states=("MA", "TX"))) == {"1", "3"}
    assert allowed_ids(DETAILS, SearchFilters(work_types=("Full-time",))) == {"1", "3"}
    assert allowed_ids(DETAILS, SearchFilters(levels=("Entry level",))) == {"1", "4"}
    # The range has to reach the minimum; no salary or another currency never passes.
    assert allowed_ids(DETAILS, SearchFilters(min_salary=75000)) == {"1", "2"}
    assert allowed_ids(DETAILS, SearchFilters(min_salary=90000)) == {"2"}
    assert allowed_ids(DETAILS, SearchFilters(remote_only=True, work_types=("Full-time",))) == {"3"}


def test_unknown_postings_pass_only_without_filters():
    assert allowed_ids(DETAILS, SearchFilters(), ["1", "99"]) == {"1", "99"}
    assert allowed_ids(DETAILS, SearchFilters(remote_only=True), ["2", "99"]) == {"2"}


def test_form_fields_are_parsed_leniently():
    f = parse_filters(True, "TX, MA,,", "Full-time", "", "80000")
    assert f == SearchFilters(True, ("MA", "TX"), ("Full-time",), (), 80000)
    assert parse_filters(min_salary="lots").min_salary is None
    assert parse_filters(min_salary="-5").min_salary is None
    assert not parse_filters().active


def test_description_and_caveats_say_what_was_left_out():
    f = SearchFilters(True, ("MA",), ("Full-time",), ("Entry level",), 80000)
    assert f.describe() == "Remote only; MA; Full-time; Entry level; at least $80,000 a year"
    caveats = " ".join(f.caveats())
    assert "level" in caveats and "salary" in caveats and "United States" in caveats
    assert SearchFilters(remote_only=True).caveats() == []


def test_options_count_indexed_postings():
    opts = filter_options(DETAILS, ["1", "2", "3"])
    assert opts["postings"] == 3 and opts["remote"] == 2 and opts["with_salary"] == 2 and opts["with_level"] == 2
    assert opts["states"] == [{"value": "MA", "postings": 1}, {"value": "TX", "postings": 1}]


class FakeEmbedder:
    def embed(self, texts):
        return [[0.0] for _ in texts]


class FakeStore:
    """Returns postings p0, p1, ... in order; records how it was asked."""

    def __init__(self, n=200):
        self.n, self.calls = n, []

    def query(self, vector, top_k=5, job_ids=None):
        self.calls.append((top_k, job_ids))
        ids = [f"p{i}" for i in range(self.n)]
        if job_ids is not None:
            ids = [j for j in ids if j in set(job_ids)]
        return [{"job_id": j, "distance": float(i), "company_name": None} for i, j in enumerate(ids[:top_k])]


SECTIONS = [ResumeSection("SKILLS", "python")]


def test_small_filter_sets_use_a_filtered_search():
    store = FakeStore()
    out = fusion.retrieve_for_resume(SECTIONS, FakeEmbedder(), store, top_k=3, per_section_k=10, allowed={"p50", "p7"})
    assert [r["job_id"] for r in out] == ["p7", "p50"]
    assert store.calls == [(10, ["p50", "p7"])]


def test_large_filter_sets_fetch_more_and_drop_the_rest(monkeypatch):
    monkeypatch.setattr(fusion, "EXACT_FILTER_LIMIT", 2)
    store = FakeStore()
    allowed = {f"p{i}" for i in range(0, 200, 3)}
    out = fusion.retrieve_for_resume(SECTIONS, FakeEmbedder(), store, top_k=3, per_section_k=10, allowed=allowed)
    assert [r["job_id"] for r in out] == ["p0", "p3", "p6"]
    assert store.calls == [(10 * fusion.OVERFETCH, None)]


def test_empty_filter_set_returns_nothing():
    assert fusion.retrieve_for_resume(SECTIONS, FakeEmbedder(), FakeStore(), allowed=set()) == []
