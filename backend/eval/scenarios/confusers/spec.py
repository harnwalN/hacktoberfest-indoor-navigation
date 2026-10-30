"""Confusers: things that tempt the pipeline into wrong merges and invented room numbers.

One floor (floor 2; no walk is told its start floor). A 22 m corridor whose walls, floor tiles and fixtures
repeat every 9.6 m, so 'ROOM 214' and 'ROOM 216' sit at two spots that look identical (same door, same
'SEMINAR 4PM ROOM 305' poster beside it) apart from the plaque. Also: '214A' across the hall, a poster with a
phone number, a directional 'ROOMS 220-230' sign at the end of the corridor (its numbers are not this place),
and a branch with rooms 220 and RM 224. Room 305 does not exist: any map place called 305 is invented.
"""

PERIOD = 9.6  # metres; multiple of the 0.6 m tiles, 1.2 m wall panels and 2.4 m ceiling lights

GRID = [
    "####################..",  # y 7-8: branch x 20..22
    "####################..",
    "####################..",
    "####################..",
    "####################..",
    "####################..",  # y 2-3
    "......................",  # y 1-2: main corridor x 0..22
    "......................",  # y 0-1
]
TWIN_DOOR = (60, 95, 140)

SPEC = {
    "description": "Look-alike rooms 9.6 m apart, poster room numbers, directional range sign",
    "seed": 33,
    "size": (1280, 720),
    "levels": {2: {"z": 0.0, "grid": GRID, "stripe": (60, 130, 60), "period": PERIOD}},
    "decals": [
        {"kind": "door", "id": "d214", "level": 2, "wall": [3.5, 2, "S"], "plaque": "ROOM 214", "room": "214",
         "color": TWIN_DOOR, "place": {"id": "r214", "label": "Room 214"}},
        {"kind": "poster", "id": "p_sem1", "level": 2, "wall": [5.5, 2, "S"], "lines": ["SEMINAR", "4PM ROOM 305"],
         "look_key": "seminar", "bg": (240, 200, 150)},
        {"kind": "door", "id": "d216", "level": 2, "wall": [3.5 + PERIOD, 2, "S"], "plaque": "ROOM 216",
         "room": "216", "color": TWIN_DOOR, "place": {"id": "r216", "label": "Room 216"}},
        {"kind": "poster", "id": "p_sem2", "level": 2, "wall": [5.5 + PERIOD, 2, "S"],
         "lines": ["SEMINAR", "4PM ROOM 305"], "look_key": "seminar", "bg": (240, 200, 150)},
        {"kind": "door", "id": "d214a", "level": 2, "wall": [8.5, 0, "N"], "plaque": "214A", "room": "214A",
         "plaque_style": "plaque_dark", "color": (50, 70, 110), "place": {"id": "r214a", "label": "Room 214A"}},
        {"kind": "poster", "id": "p_cat", "level": 2, "wall": [17.5, 0, "N"], "lines": ["LOST CAT", "CALL 555-0147"],
         "bg": (150, 230, 250)},
        {"kind": "sign", "id": "s_range", "level": 2, "wall": [22, 1, "W"], "text": "ROOMS 220-230",
         "style": "directional", "arrow": "left", "height": 1.75},
        {"kind": "sign", "id": "s_level2", "level": 2, "wall": [0, 1, "E"], "text": "LEVEL 2", "style": "floor"},
        {"kind": "door", "id": "d220", "level": 2, "wall": [20, 3.6, "E"], "plaque": "ROOM 220", "room": "220",
         "color": (40, 110, 70), "place": {"id": "r220", "label": "Room 220"}},
        {"kind": "door", "id": "d224", "level": 2, "wall": [22, 7.2, "W"], "plaque": "RM 224", "room": "224",
         "color": (60, 60, 140), "place": {"id": "r224", "label": "Room 224"}},
    ],
    "videos": [
        {"id": "east", "start_floor": 2, "give_start_floor": False, "start": [1.0, 1.0], "heading": 90,
         "description": "west to east past both twins, into the branch",
         "script": [{"visit": "r214"}, {"visit": "r214a"}, {"visit": "r216"}, {"go": [21.0, 1.0]}, {"pause": 1.0},
                    {"visit": "r220"}, {"visit": "r224"}]},
        {"id": "west", "start_floor": 2, "give_start_floor": False, "start": [21.0, 7.4], "heading": 180,
         "description": "the reverse walk, ending at the LEVEL 2 sign",
         "script": [{"visit": "r224"}, {"visit": "r220"}, {"go": [21.0, 1.0]}, {"visit": "r216"},
                    {"visit": "r214a"}, {"visit": "r214"}, {"go": [1.0, 1.0]}]},
        {"id": "twins", "start_floor": 2, "give_start_floor": False, "start": [15.5, 1.0], "heading": 270,
         "description": "short walk past the two look-alike rooms only",
         "script": [{"visit": "r216"}, {"visit": "r214"}, {"go": [2.0, 1.0]}]},
    ],
    "must_not_merge": [["r214", "r216", "identical doors and posters 9.6 m apart"]],
    "routes": [
        {"from": "r214", "to": "r224", "floor_changes": 0},
        {"from": "r216", "to": "r214", "floor_changes": 0},
        {"from": "r220", "to": "r214a", "floor_changes": 0},
    ],
}
