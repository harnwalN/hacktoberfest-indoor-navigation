"""Dead ends: corridor ends that are NOT exits, a NO EXIT sign, and one real exit seen from two directions.

One floor. A 20 m east-west corridor (y 0..2) with a 2 m branch going north at x 15..17 (to y 12).
- West end (x 0): a plain wall, no sign. The walker stands there facing it (a dead end, not an exit).
- North end of the branch (y 12): a plain door on the end wall, no plaque, no sign (a door, not an exit).
- South wall opposite the branch (x 16): the real EXIT (door + green EXIT sign). Walk 1 (west to east) sees it
  obliquely from the corridor, walk 2 straight ahead coming south down the branch: one exit, two directions.
- North wall at x 7.5: a door with a green 'NO EXIT' plate (the exit sign style): not an exit.
Rooms 120 / RM 122 (both walks), 124 (walk 1) and 130 (walk 2); walk 2's start floor is not given. Walk 2 reads the
NO EXIT plate ahead before passing it (`sees`): one door seen by both walks.
Any exit/entrance/restroom/elevator on the map other than this exit is a false key place.
"""

BRANCH = "###############..####"  # x 15..17 open (21 m wide)
GRID = [
    *[BRANCH] * 10,  # y 2-12: branch
    ".....................",  # y 1-2: main corridor x 0..21
    ".....................",  # y 0-1
]

SPEC = {
    "description": "Dead-end corridor, door at a corridor end, NO EXIT sign, one exit seen from two directions",
    "seed": 45,
    "size": (1280, 720),
    "levels": {1: {"z": 0.0, "grid": GRID, "stripe": (150, 90, 40)}},
    "decals": [
        {"kind": "door", "id": "d120", "level": 1, "wall": [4, 0, "N"], "plaque": "ROOM 120", "room": "120",
         "place": {"id": "r120", "label": "Room 120"}},
        {"kind": "door", "id": "dnoexit", "level": 1, "wall": [7.5, 2, "S"], "color": (90, 90, 95),
         "sign": {"text": "NO EXIT", "style": "exit", "size": [0.62, 0.18]},
         "place": {"id": "noexit", "label": "NO EXIT door", "type": "landmark"}},
        {"kind": "door", "id": "d122", "level": 1, "wall": [11, 2, "S"], "plaque": "RM 122", "room": "122",
         "color": (50, 70, 110), "place": {"id": "r122", "label": "Room 122"}},
        {"kind": "door", "id": "dexit", "level": 1, "wall": [16, 0, "N"], "color": (90, 90, 95),
         "sign": {"text": "EXIT", "style": "exit"}, "place": {"id": "exit_s", "label": "Exit", "type": "exit"}},
        {"kind": "door", "id": "d124", "level": 1, "wall": [18.6, 2, "S"], "plaque": "ROOM 124", "room": "124",
         "color": (40, 110, 70), "place": {"id": "r124", "label": "Room 124"}},
        {"kind": "door", "id": "d130", "level": 1, "wall": [15, 7, "E"], "plaque": "ROOM 130", "room": "130",
         "color": (60, 60, 140), "place": {"id": "r130", "label": "Room 130"}},
        {"kind": "door", "id": "dend", "level": 1, "wall": [16, 12, "S"], "color": (120, 100, 70),
         "place": {"id": "door_n", "label": "Door at the north end", "type": "landmark"}},
    ],
    "places": [
        {"id": "dead_w", "label": "Hallway end (west)", "type": "hallway", "floor": 1, "stand": [1.0, 1.0]},
    ],
    "videos": [
        {"id": "east", "start_floor": 1, "give_start_floor": True, "start": [3.0, 1.0], "heading": 270,
         "description": "dead end at the west wall, east past NO EXIT, the exit seen obliquely, to room 124",
         "script": [{"go": [1.0, 1.0]}, {"pause": 1.5}, {"face": 90}, {"visit": "r120"}, {"visit": "noexit"},
                    {"visit": "r122"}, {"go": [14.3, 1.0]}, {"look": "exit_s"}, {"visit": "r124"}]},
        {"id": "west", "start_floor": 1, "give_start_floor": False, "start": [16.0, 10.8], "heading": 0,
         "description": "from the door at the branch end down the branch, facing the exit, west to the dead end "
                        "(start floor not given)",
         "script": [{"look": "door_n"}, {"pause": 1.0}, {"face": 180}, {"visit": "r130"}, {"go": [16.0, 1.6]},
                    {"look": "exit_s"}, {"face": 270}, {"visit": "r122"}, {"visit": "r120"}, {"go": [1.0, 1.0]},
                    {"pause": 1.5}],
         "sees": ["noexit"]},  # reads the NO EXIT plate ahead on the right before passing it: one door, two walks
    ],
    "must_not_merge": [["exit_s", "noexit", "a NO EXIT sign is not the exit"],
                       ["exit_s", "dead_w", "a dead end is not the exit"],
                       ["exit_s", "door_n", "a door at a corridor end is not the exit"]],
    "routes": [
        {"from": "r120", "to": "exit_s", "floor_changes": 0},
        {"from": "dead_w", "to": "r124", "floor_changes": 0},
        {"from": "r130", "to": "r120", "floor_changes": 0},
        {"from": "door_n", "to": "noexit", "floor_changes": 0},
    ],
}
