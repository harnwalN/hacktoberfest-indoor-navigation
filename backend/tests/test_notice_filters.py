"""Notice text, partial plaques, image-edge numbers and label checks: no invented room numbers and no
poster-derived rooms or labels (live gemma3:4b read 'SEMINAR / 4PM ROOM 305' posters as 'Seminar Room 305',
a cut 'ROOM 21' plaque as room 21, and named spots after flyers: 'Math Tutoring Office').
OCR is faked; no Ollama needed."""
import pytest
from PIL import Image

from app import gemma, ocr
from app.matching import complete_room_id, cut_number, notice, place_tokens, room_ids
from app.schemas import FrameAnalysis


def _fa(**kw) -> FrameAnalysis:
    base = {"place_type": "room", "suggested_label": "", "description": ""}
    return FrameAnalysis(**(base | kw))


# ---------------------------------------------------------------- matching: notice / cut / complete
@pytest.mark.parametrize("text", [
    "4PM ROOM 305", "PM ROOM 305", "APM ROOM 305", "4INI ROOM 305", "FRIDAY 10AM", "KRIDAY 10AM", "10:30",
    "OCT 31", "10/31", "$12", "CALL 555-0147", "MEET IN ROOM 305", "TODAY", "TUE 5",
])
def test_notice_text(text):
    assert notice(text)


@pytest.mark.parametrize("text", [
    "ROOM 305", "SEMINAR ROOM 305", "LEVEL 2 ROOM 214", "214A", "107", "MAKER SPACE", "RM 224", "EXIT",
    "ROOMS 220-230", "LEVEL 2", "2ND FLOOR", "API ROOM 305",  # the last one is caught by the same-frame rule
])
def test_not_notice_text(text):
    assert not notice(text)


def test_cut_number():
    assert cut_number("ROOM 10?") and cut_number("ROOM 2'") and cut_number("21…")
    assert not cut_number("ROOM 101") and not cut_number("214A")


@pytest.mark.parametrize("rid, ok", [
    ("121", True), ("214A", True), ("B204", True), ("2-105", True), ("RM 118", True), ("ROOM 305", True),
    ("21", False), ("2", False), ("ROOM 10", False), ("MAKER SPACE", True),
])
def test_complete_room_id(rid, ok):
    assert complete_room_id(rid) is ok


def test_room_ids_and_place_tokens_skip_notice_and_cut_text():
    assert room_ids(["SEMINAR", "4PM ROOM 305"]) == set()
    assert room_ids(["ROOM 21?"]) == set()
    assert room_ids(["ROOM 305"]) == {"305"} and room_ids(["107", "MAKER SPACE"]) == {"107"}
    assert place_tokens(["FRIDAY 10AM"]) == set() and place_tokens(["ROOM 214"]) == {"214"}


# ---------------------------------------------------------------- ocr: sign lines, partial plaques, posters
def test_sign_line_excludes_notice_text():
    assert not ocr.sign_line("4PM ROOM 305") and not ocr.sign_line("FRIDAY 10AM")
    assert ocr.sign_line("ROOM 305") and ocr.sign_line("SEMINAR")


@pytest.mark.parametrize("line, partial", [
    ("ROOM 21", True), ("ROOM 10?", True), ("2", True), ("ROOM 2", True), ("21 LAB", True),
    ("ROOM 216", False), ("214A", False), ("107", False), ("RM 224", False), ("LEVEL 2", False),
    ("MAKER SPACE", False), ("ROOMS 220-230", False),
])
def test_partial_plaque(line, partial):
    assert ocr.partial_plaque(line) is partial
    if partial:
        assert not ocr.plaque_like(line)


def test_poster_lines():
    assert ocr.poster_lines(["ROOM 214", "SEMINAR"]) == []  # no notice text in the frame: nothing is a poster
    assert set(ocr.poster_lines(["SEMINAR", "4PM ROOM 305", "LEVEL 2", "ROOM 214", "EXIT", "ROOMS 220-230"])) == \
        {"SEMINAR", "4PM ROOM 305"}
    assert set(ocr.poster_lines(["ATH TUTORING", ":ALL 555-0199"])) == {"ATH TUTORING", ":ALL 555-0199"}


def test_edge_cut():
    assert ocr._edge_cut("ROOM 21", 0.7, 0.995, True, True)  # number runs off the right edge
    assert ocr._edge_cut("14A", 0.0, 0.1, True, True)  # ... or the left edge
    assert not ocr._edge_cut("OM 220", 0.0, 0.2, True, True)  # a cut word still says 220
    assert not ocr._edge_cut("ROOM 216", 0.4, 0.6, True, True)
    assert not ocr._edge_cut("ROOM 21", 0.7, 0.995, True, False)  # a crop's cut side is handled elsewhere


def test_read_text_with_crops_drops_numbers_at_the_image_edge(tmp_path, monkeypatch):
    p = tmp_path / "f.jpg"
    Image.new("RGB", (1000, 500), (200, 200, 200)).save(p)
    monkeypatch.setattr(ocr, "available", lambda: True)
    calls = []

    def observations(img):
        calls.append(img.size)
        if len(calls) == 1:  # full view: a plaque cut at the right image edge, a whole one in the middle
            return [("ROOM 21", 0.9, 0.999), ("ROOM 305", 0.4, 0.5)]
        if len(calls) == 2:  # left crop: its left side is the image edge
            return [("14A", 0.0, 0.1), ("EXIT", 0.3, 0.4)]
        return [("ROOM 21", 0.8, 0.995)]  # right crop: its right side is the image edge

    monkeypatch.setattr(ocr, "_observations", observations)
    assert ocr.read_text_with_crops(p) == ["ROOM 305", "EXIT"]


# ---------------------------------------------------------------- gemma._ground: room numbers
def test_notice_line_gives_no_room_number_or_text():
    g = gemma._ground(_fa(texts=["SEMINAR", "4PM ROOM 305"], room_numbers=["305", "4PM ROOM 305"],
                          suggested_label="Seminar Room 305", description="Long hallway with a room door"),
                      ["SEMINAR", "4PM ROOM 305"])
    assert g.room_numbers == [] and "4PM ROOM 305" not in g.texts
    assert "305" not in g.suggested_label and g.suggested_label == "Long hallway with a door"


def test_number_repeated_from_a_notice_line_is_no_room():
    """'API ROOM 305' is a misread of the same poster as '4INI ROOM 305' in this frame."""
    lines = ["LEVEL 2", "SEMINAR", "API ROOM 305", "4INI ROOM 305"]
    g = gemma._ground(_fa(texts=lines, room_numbers=["305"], suggested_label="Seminar Room 305",
                          description="Seminar room with a door"), lines)
    assert g.room_numbers == [] and g.texts == ["LEVEL 2", "SEMINAR"]
    assert g.suggested_label == "Room"  # the description is built on the poster too


def test_partial_plaque_gives_no_room_number():
    g = gemma._ground(_fa(texts=["ROOM 21"], room_numbers=["21"], suggested_label="Room 21",
                          description="Wooden door with room number 21"), ["ROOM 21"])
    assert g.room_numbers == [] and g.texts == [] and g.suggested_label == "Wooden door with room number"
    g = gemma._ground(_fa(texts=["LEVEL 2", "ROOM 2"], room_numbers=["2"], suggested_label="Level 2 Room 2",
                          description="A hallway with a door"), ["LEVEL 2", "ROOM 2"])
    assert g.room_numbers == [] and g.suggested_label == "A hallway with a door"


def test_room_number_must_be_complete():
    g = gemma._ground(_fa(room_numbers=["10"], suggested_label="Room 10", description="White hallway"),
                      ["0L NO0G", "ROOM 10"])
    assert g.room_numbers == [] and g.suggested_label == "White hallway"


@pytest.mark.parametrize("lines, rn, label", [
    (["ROOM 305"], "305", "Room 305"),
    (["SEMINAR ROOM 305"], "305", "Seminar Room 305"),
    (["214A"], "214A", "Room 214A"),
    (["107", "MAKER SPACE"], "107", "Maker Space 107"),
    (["107", "MAKER SP"], "107", "Room 107 Maker Space"),
    (["OM 105"], "105", "Room 105"),
    (["LEVEL 2", "-", "ROOM 214"], "214", "Room 214"),
])
def test_real_plaques_still_work(lines, rn, label):
    g = gemma._ground(_fa(texts=list(lines), room_numbers=[rn], suggested_label=label), list(lines))
    assert g.room_numbers == [rn] and g.suggested_label == label


def test_real_plaque_next_to_an_unrelated_poster():
    lines = ["214A", "SEMINAR", "4PM ROOM 305"]
    g = gemma._ground(_fa(texts=["214A", "SEMINAR"], room_numbers=["214A", "305"], suggested_label="Room 214A"),
                      lines)
    assert g.room_numbers == ["214A"] and g.suggested_label == "Room 214A"


# ---------------------------------------------------------------- gemma._ground: labels
def test_label_number_must_be_a_kept_room():
    g = gemma._ground(_fa(room_numbers=["216"], suggested_label="Room 216 and 214"), ["ROOM 216", "214"])
    assert g.room_numbers == ["216"] and g.suggested_label == "Room 216"  # 214 is real text but not kept
    g = gemma._ground(_fa(place_type="hallway", suggested_label="Lab 214"), ["ROOMS 210-220", "214"])
    assert "214" not in g.suggested_label


def test_invented_plaque_word_dropped():
    g = gemma._ground(_fa(texts=["ROCI"], suggested_label="ROCI Room"), ["ROCI"])
    assert g.suggested_label == "ROCI"
    g = gemma._ground(_fa(place_type="hallway", texts=["LOSTCAT"], suggested_label="Lost Cat Office"), ["LOSTCAT"])
    assert g.suggested_label == "Lost Cat"
    # read on a sign, or naming a kept room number: stays
    g = gemma._ground(_fa(texts=["CHEMISTRY LAB"], suggested_label="Chemistry Lab"), ["CHEMISTRY LAB"])
    assert g.suggested_label == "Chemistry Lab"
    g = gemma._ground(_fa(texts=["214A"], room_numbers=["214A"], suggested_label="Room 214A"), ["214A"])
    assert g.suggested_label == "Room 214A"


def test_poster_words_never_name_the_place():
    g = gemma._ground(_fa(suggested_label="Math Tutoring Office", description="Math tutoring office with a sign"),
                      ["ATH TUTORING", ":ALL 555-0199"])
    assert g.suggested_label == "Room"
    g = gemma._ground(_fa(place_type="hallway", texts=["UITAR LESSONS"], suggested_label="Guitar Lessons Hallway",
                          description="Long hallway with blue accents"), ["UITAR LESSONS", "'ALL 555-0147"])
    assert g.suggested_label == "Long hallway with blue accents"
    g = gemma._ground(_fa(place_type="hallway", texts=["BOOKFAIR", "EXIT"], suggested_label="Book Fair Exit",
                          description="Long hallway with blue border"), ["BOOKFAIR", "FRIDAY 10AM", "EXIT"])
    assert g.suggested_label == "Long hallway with blue border" and g.texts == ["BOOKFAIR", "EXIT"]


def test_poster_label_that_says_poster_is_kept():
    lines = ["BAKE SALE", "Oct 31 5 PM $12"]
    g = gemma._ground(_fa(place_type="landmark", texts=["BAKE SALE"], suggested_label="Bake sale poster"), lines)
    assert g.suggested_label == "Bake sale poster"
    g = gemma._ground(_fa(place_type="hallway", texts=["LOST CAT"], suggested_label="Lost Cat Notice"),
                      ["LOST CAT", "CALL 555-0147"])
    assert g.suggested_label == "Lost Cat Notice"


def test_wayfinding_sign_next_to_a_poster_still_names_the_place():
    lines = ["EXIT", "BOOK FAIR", "FRIDAY 10AM", "RM 112"]
    g = gemma._ground(_fa(place_type="hallway", texts=["EXIT", "RM 112"], room_numbers=["112"],
                          suggested_label="Exit Hallway"), lines)
    assert g.room_numbers == ["112"] and g.suggested_label == "Exit Hallway"


def test_directional_description_survives_as_label():
    g = gemma._ground(_fa(place_type="hallway", texts=["ROOMS 220-230"], suggested_label="Rooms 220 to 230",
                          description="Rooms 220 to 230 hallway"), ["ROOMS 220-230"])
    assert g.room_numbers == [] and g.suggested_label == "Rooms 220 to 230 hallway"


# ---------------------------------------------------------------- cached answers are re-checked
def test_tidy_rooms_rechecks_cached_answers():
    """A v6 answer cached before these checks: re-scored without a PROMPT_VERSION bump."""
    cached = _fa(texts=["SEMINAR", "4PM ROOM 305"], room_numbers=["305"], suggested_label="Seminar Room 305",
                 description="Long hallway with a room door", ocr=["SEMINAR", "4PM ROOM 305"], grounded=True,
                 has_sign=True)
    t = gemma._tidy_rooms(cached)
    assert t.room_numbers == [] and t.texts == ["SEMINAR"] and t.suggested_label == "Long hallway with a door"
    again = gemma._tidy_rooms(t.model_copy(deep=True))
    assert again.model_dump() == t.model_dump()  # idempotent


def test_tidy_rooms_keeps_good_cached_answers():
    cached = _fa(texts=["107", "MAKER SPACE"], room_numbers=["107"], suggested_label="Maker Space 107",
                 ocr=["107", "MAKER SPACE"], grounded=True, has_sign=True)
    t = gemma._tidy_rooms(cached)
    assert t.room_numbers == ["107"] and t.suggested_label == "Maker Space 107" and t.has_sign


def test_tidy_rooms_leaves_ungrounded_answers_alone():
    a = _fa(texts=["ROOM 21"], room_numbers=["21"], suggested_label="Room 21")  # no OCR on this machine
    assert gemma._tidy_rooms(a).suggested_label == "Room 21"
