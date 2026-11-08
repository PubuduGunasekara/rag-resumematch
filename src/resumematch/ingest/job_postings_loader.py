"""Downloads and batch-iterates the LinkedIn Job Postings dataset
(Kaggle: arshkon/linkedin-job-postings, version 13: 123,849 rows).

Why kagglehub and not a plain URL: this is the authoritative source we
actually verified the schema against (job_id, title, description,
skills_desc, job_posting_url, ...). A mirror might drift or disappear;
the original Kaggle dataset is what gets cited in the report, so it's
what gets downloaded. The tradeoff is a Kaggle API credential is
required at runtime (KAGGLE_USERNAME/KAGGLE_KEY env vars) — documented
in README setup, not baked into the image.

Why chunksize iteration, not pandas.read_csv(...) in one shot: the
project scope says 10k now, 100k+ later without a rewrite. Loading
124k rows of text columns fits in memory today, but "doesn't need a
rewrite at 10x scale" means never assuming the whole file fits in
memory in the first place.
"""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

DATASET_HANDLE = "arshkon/linkedin-job-postings/versions/13"


@dataclass
class JobPosting:
    job_id: str
    title: str
    description: str
    skills_desc: str | None
    company_id: str | None
    job_posting_url: str | None
    company_name: str | None = None  # straight from postings.csv; shown in citations


def download_dataset(dest_dir: str = "data/raw") -> Path:
    """
    Downloads the dataset via kagglehub to dest_dir if not already
    present there. Requires KAGGLE_USERNAME and KAGGLE_KEY env vars
    (see README: Kaggle account -> Settings -> API -> Create New Token).

    Returns the path to the directory containing postings.csv plus the
    jobs/, companies/ and mappings/ subfolders.
    """
    # kagglehub manages its own cache (~/.cache/kagglehub by default, or
    # $KAGGLEHUB_CACHE) and skips the download if the dataset is already
    # there, so dest_dir is not where the files land. Point KAGGLEHUB_CACHE
    # at a mounted volume in Docker to persist the download across runs.
    #
    # Pinned to version 13: unpinned, kagglehub fetches "latest", and the
    # schema has already changed once (job_postings.csv -> postings.csv,
    # company_details/ -> companies/). Bump deliberately, re-verify columns.
    import kagglehub

    return Path(kagglehub.dataset_download(DATASET_HANDLE))


def _description_key(description: str) -> bytes:
    """Identity of a posting's text, ignoring case and whitespace."""
    return hashlib.blake2b(" ".join(description.lower().split()).encode(), digest_size=16).digest()


def iter_job_postings(csv_dir: Path, batch_size: int = 1000, dedupe: bool = True) -> Iterator[list[JobPosting]]:
    """
    Yields batches of JobPosting so downstream chunking/embedding/
    indexing never holds the full 124k-row dataset in memory at once.

    Input:
        csv_dir: directory returned by download_dataset (or wherever
            postings.csv already lives).
        batch_size: rows per yielded batch.
        dedupe: skip a posting whose description (ignoring case and
            whitespace) already appeared -- 12.9% of the corpus (16,015
            postings) are exact copies, typically one recruiter reposting
            the same text, which otherwise fill several retrieval slots
            with the same job. The first job_id is kept for citations.
    """
    csv_path = csv_dir / "postings.csv"
    seen: set[bytes] = set()
    for batch_df in pd.read_csv(csv_path, chunksize=batch_size):
        batch = []
        for _, row in batch_df.iterrows():
            # description/skills_desc can be NaN — never index "nan" as text
            description = row["description"] if pd.notna(row["description"]) else ""
            skills_desc = row["skills_desc"] if pd.notna(row.get("skills_desc")) else None
            if not description.strip():
                continue  # nothing retrievable here, skip rather than index empty text
            if dedupe:
                key = _description_key(description)
                if key in seen:
                    continue
                seen.add(key)
            batch.append(JobPosting(
                job_id=str(row["job_id"]),
                title=str(row["title"]) if pd.notna(row["title"]) else "",
                description=description,
                skills_desc=skills_desc,
                company_id=str(row["company_id"]) if pd.notna(row.get("company_id")) else None,
                job_posting_url=str(row["job_posting_url"]) if pd.notna(row.get("job_posting_url")) else None,
                company_name=str(row["company_name"]) if pd.notna(row.get("company_name")) else None,
            ))
        if batch:
            yield batch


def iter_company_descriptions(csv_dir: Path, dedupe: bool = True, batch_size: int = 20000) -> Iterator[tuple[str | None, str]]:
    """(company_name, description) for every posting, with the SAME
    de-duplication as iter_job_postings -- feeds find_boilerplate, which
    must count each distinct posting once."""
    seen: set[bytes] = set()
    for batch_df in pd.read_csv(csv_dir / "postings.csv", usecols=["company_name", "description"], chunksize=batch_size):
        for company, description in zip(batch_df["company_name"], batch_df["description"]):
            if not isinstance(description, str) or not description.strip():
                continue
            if dedupe:
                key = _description_key(description)
                if key in seen:
                    continue
                seen.add(key)
            yield (company if isinstance(company, str) else None), description


def iter_posting_texts(csv_dir: Path, batch_size: int = 20000) -> Iterator[str]:
    """Every non-empty title/description/skills_desc string in the corpus,
    streamed -- feeds build_domain_vocabulary without loading the whole
    CSV or building JobPosting objects (only 3 of ~30 columns are read)."""
    columns = ["title", "description", "skills_desc"]
    for batch_df in pd.read_csv(csv_dir / "postings.csv", usecols=columns, chunksize=batch_size):
        for column in columns:
            yield from batch_df[column].dropna()
