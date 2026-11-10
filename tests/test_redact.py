"""Contact details are removed from text that goes into the shared sample."""

from resumematch.preprocess.redact import redact_contacts


def test_emails_and_phones_are_removed():
    text = ("Send resumes to jane.doe+jobs@gmail.com or hr@dbhds.virginia.gov. "
            "Call (555) 010-0199, 555.010.0123 or +1 555-010-0100.")
    out = redact_contacts(text)
    assert "gmail" not in out and "virginia.gov" not in out and "555" not in out
    assert out.count("[email removed]") == 2 and out.count("[phone removed]") == 3


def test_requirements_and_numbers_are_kept():
    text = "Requires 3-5 years of experience, a $85,000-95,000 salary range, ISO 9001:2015, and 40 hours/week."
    assert redact_contacts(text) == text


def test_empty_values_pass_through():
    assert redact_contacts(None) is None and redact_contacts("") == ""
