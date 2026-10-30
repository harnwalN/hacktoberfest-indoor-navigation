"""Adversarial edge cases for ocr.grounded / ocr.floor_grounded (sentinel-backend)."""
import pytest

from app import ocr


@pytest.mark.parametrize("cand, lines, ok", [
    ("B12", ["B1"], False),
    ("B1", ["B12"], False),
    ("B12", ["B 12"], True),
    ("EXIT", ["EXIT"], True),
    ("exit", ["Exit →"], True),
    ("EXIT", ["EXITS"], False),
    ("ROOM 121", ["room–121"], True),   # en dash
    ("ROOM 121", ["ROOM—121"], True),   # em dash
    ("ROOM 121", ["ROOM", "121"], True),     # split over consecutive lines
    ("121", ["12", "1"], False),             # digits not glued across lines
    ("ROOM 121", ["ROOM 1210"], False),
    ("121A", ["121"], False),
    ("O12", ["012"], False),                 # letter O is not zero
    ("214", ["２１４"], False),  # fullwidth digits are never accepted (fail closed)
])
def test_grounded_adversarial(cand, lines, ok):
    assert ocr.grounded(cand, lines) is ok


@pytest.mark.parametrize("hint, lines, kept", [
    ("2", ["LEVEL 22"], False),
    ("2", ["12TH FLOOR"], False),
    ("2", ["FLOOR 2"], True),
    ("2", ["FLOOR"], False),
    ("1", ["FLOOR 1 ROOM 121"], True),
    ("", ["LEVEL 2"], False),
])
def test_floor_grounded_adversarial(hint, lines, kept):
    assert ocr.floor_grounded(hint, lines) is kept


@pytest.mark.xfail(strict=True, reason="digit runs are glued across tokens inside one line: '1 21' grounds '121'")
def test_digits_not_glued_within_a_line():
    assert ocr.grounded("121", ["1 21"]) is False


@pytest.mark.xfail(strict=True, reason="'-1' (basement) is read as floor 1: sign lost in _tokens")
def test_negative_floor_not_confused_with_floor_one():
    assert ocr.floor_grounded("-1", ["1ST FLOOR"]) is False
