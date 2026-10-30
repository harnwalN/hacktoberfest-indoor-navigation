"""Small helpers shared by mapper.py and floors.py (kept here so neither imports the other)."""
import re


def text_tokens(texts: list[str]) -> set[str]:
    """Distinctive tokens only: things containing a digit (room numbers). 'EXIT' repeats on every floor."""
    out = set()
    for t in texts:
        for tok in re.split(r"[\s,;:/()]+", t.upper()):
            tok = re.sub(r"[^A-Z0-9]", "", tok)
            if len(tok) >= 2 and any(ch.isdigit() for ch in tok):
                out.add(tok)
    return out


def _next_id(prefix: str, existing: list[str]) -> int:
    nums = [int(m.group(1)) for x in existing if (m := re.fullmatch(rf"{prefix}(\d+)", x))]
    return max(nums, default=0) + 1
