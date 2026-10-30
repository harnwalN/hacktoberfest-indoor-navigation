"""Small single-floor loop: a 2 m corridor around a 8 x 4 m block (12 x 8 m outside).

Rooms 101 / RM 103 on the south side, 105 on the east, '107 MAKER SPACE' on the north wall and 110A on the
block, an EXIT on the west wall and a phone-number poster. Two walks go round in opposite directions, so
every room is seen from both sides (loop closure across walks, same place walked the other way).
"""

GRID = [
    "............",  # y 7-8: north corridor
    "............",  # y 6-7
    "..########..",  # y 5-6: central block x 2..10
    "..########..",
    "..########..",
    "..########..",  # y 2-3
    "............",  # y 1-2: south corridor
    "............",  # y 0-1
]

SPEC = {
    "description": "Single floor loop walked both ways",
    "seed": 7,
    "size": (1280, 720),
    "levels": {1: {"z": 0.0, "grid": GRID, "stripe": (150, 90, 40)}},
    "decals": [
        {"kind": "door", "id": "d101", "level": 1, "wall": [4, 0, "N"], "plaque": "ROOM 101", "room": "101",
         "place": {"id": "r101", "label": "Room 101"}},
        {"kind": "door", "id": "d103", "level": 1, "wall": [8, 0, "N"], "plaque": "RM 103", "room": "103",
         "color": (50, 70, 110), "place": {"id": "r103", "label": "Room 103"}},
        {"kind": "door", "id": "d105", "level": 1, "wall": [12, 4, "W"], "plaque": "ROOM 105", "room": "105",
         "color": (40, 110, 70), "place": {"id": "r105", "label": "Room 105"}},
        {"kind": "door", "id": "d107", "level": 1, "wall": [8, 8, "S"], "plaque": ["107", "MAKER SPACE"],
         "plaque_size": (0.34, 0.18), "room": "107", "color": (60, 60, 140),
         "place": {"id": "r107", "label": "Maker Space 107"}},
        {"kind": "door", "id": "d110a", "level": 1, "wall": [4, 6, "N"], "plaque": "110A", "room": "110A",
         "plaque_style": "plaque_dark", "place": {"id": "r110a", "label": "Room 110A"}},
        {"kind": "door", "id": "dexit", "level": 1, "wall": [0, 4, "E"], "color": (90, 90, 95),
         "sign": {"text": "EXIT", "style": "exit"}, "place": {"id": "exit_w", "label": "Exit", "type": "exit"}},
        {"kind": "poster", "id": "p_guitar", "level": 1, "wall": [6, 2, "S"],
         "lines": ["GUITAR LESSONS", "CALL 555-0147"], "bg": (250, 220, 90)},
    ],
    "videos": [
        {"id": "ccw_loop", "start_floor": 1, "give_start_floor": True, "start": [1.5, 1.0], "heading": 90,
         "description": "counter-clockwise loop from the SW corner back to it",
         "script": [{"visit": "r101"}, {"visit": "r103"}, {"go": [11.0, 1.0]}, {"visit": "r105"},
                    {"go": [11.0, 7.0]}, {"visit": "r107"}, {"visit": "r110a"}, {"go": [1.0, 7.0]},
                    {"visit": "exit_w"}, {"go": [1.0, 1.0]}, {"go": [3.0, 1.0]}]},
        {"id": "cw_reverse", "start_floor": 1, "give_start_floor": False, "start": [6.0, 1.0], "heading": 270,
         "description": "clockwise (reverse) loop from the middle of the south corridor (start floor not given)",
         "script": [{"go": [1.0, 1.0]}, {"visit": "exit_w"}, {"go": [1.0, 7.0]}, {"visit": "r110a"},
                    {"visit": "r107"}, {"go": [11.0, 7.0]}, {"visit": "r105"}, {"go": [11.0, 1.0]},
                    {"visit": "r103"}, {"go": [6.5, 1.0]}]},
    ],
    "routes": [
        {"from": "r101", "to": "r107", "floor_changes": 0},
        {"from": "exit_w", "to": "r105", "floor_changes": 0},
        {"from": "r110a", "to": "r103", "floor_changes": 0},
    ],
}
