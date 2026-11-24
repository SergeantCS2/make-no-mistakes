"""A link as mnm keeps it: never with a secret in its query.

A Civitai link pasted with `?token=…` (or a signed CDN link: `signature=`, `X-Amz-Credential`,
`Expires`) kept as pasted in a workflow's meta and in the model-links ledger would go from
there into `mnm backup`'s archive. Every link mnm saves goes through `without_secrets`;
the words are the ones `report.redact` scrubs a signed link's query for, named once here."""

from __future__ import annotations

import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

SECRET_WORDS = ("token", "signature", "sig", "key", "expires", "credential")
# A parameter is secret when its whole name, or one of its `_`/`-` separated parts, is one of
# the words: `token`, `api_key`, `access-token`, `X-Amz-Signature` go; `designId`, `monkey`
# stay. `report.SIGNED_URL` matches names by the same rule (`NAME_PART`).
NAME_PART = rf"(?:[^=&\s\"'<>_-]*[_-])*(?:{'|'.join(SECRET_WORDS)})(?:[_-][^=&\s\"'<>]*)?"
_PARTS = re.compile(r"[_-]")


def secret_name(name: str) -> bool:
    """Whether a query parameter's name is a secret's (the part rule above)."""
    return any(part.lower() in SECRET_WORDS for part in _PARTS.split(name))


def without_secrets(link: str) -> str:
    """`link` without any query parameter whose name carries a secret word; anything that
    is not an http(s) link (a file name, a path) as it came."""
    text = link.strip()
    parts = urlsplit(text)
    if parts.scheme.lower() not in ("http", "https") or not parts.query:
        return text
    kept = [
        (name, value)
        for name, value in parse_qsl(parts.query, keep_blank_values=True)
        if not secret_name(name)
    ]
    return urlunsplit(parts._replace(query=urlencode(kept)))
