"""Removes contact details from posting text before it is shared.

Used when building the downloadable sample index: 271 of its 2,000
postings contained an email address and 72 a phone number, some of them
personal (gmail.com). The postings are public job ads, but a bundle
anyone can download should not republish people's contact details.
The full local index is built without redaction.
"""

import re

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}")
_PHONE = re.compile(r"(?<![\w-])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?![\w-])")

EMAIL_MARK = "[email removed]"
PHONE_MARK = "[phone removed]"


def redact_contacts(text: str | None) -> str | None:
    if not text:
        return text
    return _PHONE.sub(PHONE_MARK, _EMAIL.sub(EMAIL_MARK, text))
