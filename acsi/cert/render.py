from __future__ import annotations

BANNED_CERT_WORDS = ("guarantee", "guaranteed", "identical", "zero risk", "proven equivalent")


def assert_no_banned_words(rendered: str) -> None:
    lowered = rendered.lower()
    found = [word for word in BANNED_CERT_WORDS if word in lowered]
    if found:
        raise ValueError(f"Certificate contains banned wording: {', '.join(found)}")

