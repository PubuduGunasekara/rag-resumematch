"""Small helpers in pipeline.py."""

import json

from resumematch.pipeline import indexed_postings


def test_posting_count_comes_from_the_build_marker(tmp_path):
    # Bug: the interfaces printed a fixed 107,528 even on the 2,000-posting sample index.
    (tmp_path / "BUILD_COMPLETE.json").write_text(json.dumps({"postings": 2000, "chunks": 3927}))
    assert indexed_postings(str(tmp_path)) == 2000


def test_missing_or_broken_marker_gives_none(tmp_path):
    assert indexed_postings(str(tmp_path)) is None
    (tmp_path / "BUILD_COMPLETE.json").write_text("not json")
    assert indexed_postings(str(tmp_path)) is None
