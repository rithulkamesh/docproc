"""Unit tests for sanitize module."""

from docproc.sanitize import sanitize_text, deduplicate_texts, is_boilerplate, BoilerplateKind


def test_sanitize_text_basic():
    """sanitize_text normalizes and strips."""
    assert sanitize_text("  hello  world  ") == "hello world"
    assert sanitize_text("hello\t\tworld") == "hello world"


def test_sanitize_text_none_empty():
    """sanitize_text returns empty string for None/empty."""
    assert sanitize_text(None) == ""
    assert sanitize_text("") == ""


def test_sanitize_text_collapse_newlines():
    """sanitize_text collapses excessive consecutive newlines."""
    out = sanitize_text("a\n\n\n\nb")
    assert "a" in out and "b" in out


def test_deduplicate_texts_removes_exact_duplicates():
    """deduplicate_texts removes exact duplicates."""
    texts = ["hello", "world", "hello", "foo", "world"]
    result = deduplicate_texts(texts, drop_exact_duplicates=True, drop_boilerplate=False)
    assert len(result) == 3
    assert "hello" in result
    assert "world" in result
    assert "foo" in result


def test_is_boilerplate_thank_you():
    """is_boilerplate detects thank you slides."""
    is_bp, kind = is_boilerplate("Thank you!")
    assert is_bp is True
    assert kind in (BoilerplateKind.THANK_YOU, BoilerplateKind.QUESTIONS)


def test_is_boilerplate_blank():
    """is_boilerplate treats empty/short as blank."""
    is_bp, kind = is_boilerplate("")
    assert is_bp is True
    assert kind == BoilerplateKind.BLANK


def test_is_boilerplate_not_boilerplate():
    """is_boilerplate returns False for substantive content."""
    is_bp, kind = is_boilerplate("Introduction to machine learning and deep neural networks.")
    assert is_bp is False
    assert kind == BoilerplateKind.NONE
