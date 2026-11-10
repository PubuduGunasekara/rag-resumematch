"""The sample-index download refuses bad archives and never overwrites silently."""

import hashlib
import io
import sys
import tarfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import fetch_sample_index as fetch  # noqa: E402


def make_archive(tmp_path: Path, entries: dict[str, bytes]) -> tuple[Path, str]:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    path = tmp_path / "sample.tar.gz"
    path.write_bytes(buf.getvalue())
    return path, hashlib.sha256(buf.getvalue()).hexdigest()


GOOD = {"chroma_db/chroma.sqlite3": b"db", "data/processed/domain_vocab.json": b"{}",
        "ATTRIBUTION.md": b"credit", "DATA_LICENSE.txt": b"CC BY-SA 4.0"}


def run(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["fetch_sample_index.py", *map(str, args)])
    fetch.main()


def test_good_archive_unpacks_with_notes_out_of_the_repo_root(tmp_path, monkeypatch):
    archive, digest = make_archive(tmp_path, GOOD)
    dest = tmp_path / "repo"
    run(monkeypatch, "--file", archive, "--sha256", digest, "--dest", dest)
    assert (dest / "chroma_db/chroma.sqlite3").read_bytes() == b"db"
    assert (dest / "data/processed/ATTRIBUTION.md").exists() and not (dest / "ATTRIBUTION.md").exists()


def test_checksum_mismatch_extracts_nothing(tmp_path, monkeypatch):
    archive, _ = make_archive(tmp_path, GOOD)
    with pytest.raises(SystemExit, match="Checksum mismatch"):
        run(monkeypatch, "--file", archive, "--sha256", "0" * 64, "--dest", tmp_path / "repo")
    assert not (tmp_path / "repo").exists()


@pytest.mark.parametrize("bad", ["../evil.txt", "/etc/evil", "scripts/evil.py"])
def test_unsafe_paths_are_refused(tmp_path, monkeypatch, bad):
    archive, digest = make_archive(tmp_path, {**GOOD, bad: b"x"})
    with pytest.raises(SystemExit, match="Unsafe path"):
        run(monkeypatch, "--file", archive, "--sha256", digest, "--dest", tmp_path / "repo")


def test_existing_index_is_kept_unless_forced(tmp_path, monkeypatch):
    archive, digest = make_archive(tmp_path, GOOD)
    dest = tmp_path / "repo"
    (dest / "chroma_db").mkdir(parents=True)
    (dest / "chroma_db/old.sqlite3").write_bytes(b"full index")
    with pytest.raises(SystemExit, match="already exists"):
        run(monkeypatch, "--file", archive, "--sha256", digest, "--dest", dest)
    run(monkeypatch, "--file", archive, "--sha256", digest, "--dest", dest, "--force")
    moved = [p for p in dest.iterdir() if p.name.startswith("chroma_db.before-sample-")]
    assert len(moved) == 1 and (moved[0] / "old.sqlite3").read_bytes() == b"full index"
