from __future__ import annotations

import re


_BRACKET_UNIT_RE = re.compile(r"\[(?P<unit>[^\[\]]{1,40})\]\s*$")
_PAREN_UNIT_RE = re.compile(r"\((?P<unit>[^()]{1,40})\)\s*$")


def split_column_unit(name: str) -> tuple[str, str]:
    """Return display name and unit parsed from common column labels.

    Supported examples:
    - ``pressure [Pa]`` -> ``("pressure", "Pa")``
    - ``temperature (C)`` -> ``("temperature", "C")``
    """
    text = str(name).strip()
    for pattern in (_BRACKET_UNIT_RE, _PAREN_UNIT_RE):
        match = pattern.search(text)
        if match:
            unit = match.group("unit").strip()
            label = text[: match.start()].strip()
            return label or text, unit
    return text, ""


def unit_for_column(name: str) -> str:
    return split_column_unit(name)[1]


__all__ = ["split_column_unit", "unit_for_column"]
