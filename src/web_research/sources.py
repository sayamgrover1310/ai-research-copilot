"""Normalized, application-owned source records for web research."""

from dataclasses import dataclass


@dataclass(frozen=True)
class WebSource:
    """One web result kept as evidence and displayed as a trustworthy source."""

    label: str
    title: str
    url: str
    content: str
    rank: int


def format_web_source(source: WebSource) -> str:
    """Return a short citation line using provider-returned title and URL only."""
    return f"{source.label} {source.title} — {source.url}"
