"""Two floors joined by one straight stairwell (two flights with a mid landing).

Floor 1: corridor y 0..2 m, x 0..16 m, rooms 110/112/114, EXIT at the west end, STAIRS sign, LEVEL 1 sign.
Floor 2: corridor y 13..15 m, rooms 210/212/214A, LEVEL 2 sign facing the top of the stairs, a phone poster.
Stairwell: x 13..15, y 3..12, entered from floor 1 at its south end and left at its north end on floor 2.

Walks: up (start floor given), down the reverse way (start floor NOT given: must come from rooms/signs),
and a short floor-1 revisit (start floor not given).
"""

L1 = [
    "################",  # y 14-15
    "################",
    "################",  # y 12-13
    *["#############SS#"] * 9,  # y 3-12: stair shaft at x 13-15
    "#############oo#",  # y 2-3: opening from the corridor into the shaft
    "................",  # y 1-2
    "................",  # y 0-1
]
L2 = [
    "................",  # y 14-15: floor-2 corridor
    "................",  # y 13-14
    "#############oo#",  # y 12-13: opening from the shaft into the corridor
    *["#############SS#"] * 9,
    "################",
    "################",
    "################",
]

SPEC = {
    "description": "Two floors, one stairwell; up, down (reverse) and a short revisit",
    "seed": 21,
    "size": (1280, 720),
    "levels": {
        1: {"z": 0.0, "grid": L1, "stripe": (150, 90, 40)},
        2: {"z": 3.4, "grid": L2, "stripe": (60, 130, 60)},
    },
    "decals": [
        # floor 1
        {"kind": "door", "id": "d110", "level": 1, "wall": [3, 0, "N"], "plaque": "ROOM 110", "room": "110",
         "place": {"id": "r110", "label": "Room 110"}},
        {"kind": "door", "id": "d112", "level": 1, "wall": [6.5, 2, "S"], "plaque": "RM 112", "room": "112",
         "color": (50, 70, 110), "place": {"id": "r112", "label": "Room 112"}},
        {"kind": "door", "id": "d114", "level": 1, "wall": [11.0, 0, "N"], "plaque": "ROOM 114", "room": "114",
         "place": {"id": "r114", "label": "Room 114"}},
        {"kind": "door", "id": "dexit", "level": 1, "wall": [0, 1, "E"], "color": (90, 90, 95), "stand": 2.0,
         "sign": {"text": "EXIT", "style": "exit"}, "place": {"id": "exit_w", "label": "Exit", "type": "exit"}},
        {"kind": "sign", "id": "s_stairs1", "level": 1, "wall": [12.0, 2, "S"], "text": "STAIRS", "style": "stairs",
         "arrow": "right", "height": 1.8},
        {"kind": "sign", "id": "s_level1", "level": 1, "wall": [14.0, 0, "N"], "text": "LEVEL 1", "style": "floor"},
        {"kind": "poster", "id": "p_fair", "level": 1, "wall": [6.0, 0, "N"], "lines": ["BOOK FAIR", "FRIDAY 10AM"],
         "bg": (120, 200, 250)},
        # floor 2
        {"kind": "sign", "id": "s_level2", "level": 2, "wall": [14.0, 15, "S"], "text": "LEVEL 2", "style": "floor"},
        {"kind": "door", "id": "d214a", "level": 2, "wall": [11, 13, "N"], "plaque": "214A", "room": "214A",
         "plaque_style": "plaque_dark", "place": {"id": "r214a", "label": "Room 214A"}},
        {"kind": "door", "id": "d212", "level": 2, "wall": [6.5, 15, "S"], "plaque": "RM 212", "room": "212",
         "color": (40, 110, 70), "place": {"id": "r212", "label": "Room 212"}},
        {"kind": "door", "id": "d210", "level": 2, "wall": [4, 13, "N"], "plaque": "ROOM 210", "room": "210",
         "place": {"id": "r210", "label": "Room 210"}},
        {"kind": "poster", "id": "p_tutor", "level": 2, "wall": [4.5, 15, "S"], "lines": ["MATH TUTORING", "CALL 555-0199"],
         "bg": (250, 230, 140)},
    ],
    "stairs": [
        {"id": "A", "lower": 1, "upper": 2, "dir": "N", "across": [13.0, 15.0], "start": 4.0, "flights": [10, 10],
         "rise": 0.17, "run": 0.28, "landing": 1.4, "top_landing_end": 12.0, "bottom": [14.0, 2.5], "top": [14.0, 12.5]},
    ],
    "places": [
        {"id": "stairs_1", "label": "Stairs (floor 1)", "type": "stairs", "floor": 1, "stand": [14.0, 2.5], "group": "A"},
        {"id": "stairs_2", "label": "Stairs (floor 2)", "type": "stairs", "floor": 2, "stand": [14.0, 12.5], "group": "A"},
    ],
    "videos": [
        {"id": "up", "start_floor": 1, "give_start_floor": True, "start": [1.2, 1.0], "heading": 90,
         "description": "floor 1 west to east, up the stairs, floor 2 east to west",
         "script": [{"visit": "r110"}, {"visit": "r112"}, {"visit": "r114"}, {"go": [14.0, 1.0]},
                    {"stairs": "A", "dir": "up"}, {"go": [14.0, 14.0]}, {"visit": "r214a"}, {"visit": "r212"},
                    {"visit": "r210"}]},
        {"id": "down", "start_floor": 2, "give_start_floor": False, "start": [2.0, 14.0], "heading": 90,
         "description": "floor 2 west to east, down the stairs, floor 1 east to west (start floor not given)",
         "script": [{"visit": "r210"}, {"visit": "r212"}, {"visit": "r214a"}, {"go": [14.0, 14.0]},
                    {"stairs": "A", "dir": "down"}, {"go": [14.0, 1.0]}, {"visit": "r114"}, {"visit": "r112"},
                    {"go": [5.0, 1.0]}]},
        {"id": "f1_revisit", "start_floor": 1, "give_start_floor": False, "start": [11.0, 1.0], "heading": 270,
         "description": "short floor-1 walk east to west ending at the exit (start floor not given)",
         "script": [{"visit": "r114"}, {"visit": "r112"}, {"visit": "r110"}, {"visit": "exit_w"}]},
    ],
    "stair_links": [["stairs_1", "stairs_2"]],
    "routes": [
        {"from": "r110", "to": "r212", "floor_changes": 1},
        {"from": "r214a", "to": "exit_w", "floor_changes": 1},
        {"from": "r112", "to": "r114", "floor_changes": 0},
        {"from": "r210", "to": "r214a", "floor_changes": 0},
    ],
}
