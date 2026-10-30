"""Ground-truth format for evaluation scenarios (truth.json, schema "waypoint-eval/1").

Source-agnostic: the synthetic generator writes it, but a Habitat/HM3D render, a LaMAR sequence or a hand-
annotated phone walk can supply the same file next to its videos. Only `building.places` and per-video
`visits` are required; everything else has a sensible default. See eval/README.md for the field reference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA = "waypoint-eval/1"
PlaceType = Literal["room", "stairs", "elevator", "exit", "entrance", "hallway", "intersection", "restroom",
                    "landmark"]


class Place(BaseModel):
    id: str
    label: str
    room: str | None = None  # canonical room number on its plaque ('110', '214A'); None if it has none
    floor: int
    type: PlaceType = "room"
    x: float | None = None  # approx. metres; any consistent frame per building (used for layout checks only)
    y: float | None = None
    group: str | None = None  # stairwell / elevator bank: the same group on two floors is one shaft
    aliases: list[str] = []  # other ways its sign is written


class Distractor(BaseModel):
    id: str
    text: str
    kind: str = "poster"
    floor: int | None = None
    x: float | None = None
    y: float | None = None


class BuildingTruth(BaseModel):
    floors: list[int] = []
    places: list[Place]
    distractors: list[Distractor] = []


class Visit(BaseModel):
    place: str
    t: float  # seconds into the video when the walker is at (or looking at) the place
    kind: Literal["look", "pass"] = "pass"


class Turn(BaseModel):
    t: float  # middle of the turn, seconds
    delta: float  # degrees, clockwise positive (right turn > 0)
    start: float | None = None
    end: float | None = None


class VideoTruth(BaseModel):
    id: str
    file: str  # relative to the scenario directory
    duration: float | None = None
    start_floor: int
    give_start_floor: bool = False  # tell the pipeline the start floor (as an uploader would), or make it infer it
    sensors: str | None = None  # optional phone sensors.json, relative to the scenario directory
    description: str = ""
    visits: list[Visit]
    turns: list[Turn] = []  # changes of walking direction (optional; scored when present)
    path: list[list[float | int | None]] = []  # optional [t, x, y, floor] samples of where the walker was


class Route(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    from_: str = Field(alias="from")
    to: str
    floor_changes: int | None = None  # expected number of floor changes along the route


class Truth(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    schema_: str = Field(SCHEMA, alias="schema")
    scenario: str
    description: str = ""
    source: dict = {}
    building: BuildingTruth
    videos: list[VideoTruth]
    expect_merged: list[str] = []  # places visited more than once that must end up as ONE map place
    must_not_merge: list[list[str]] = []  # [place_a, place_b, optional reason]: never one map place
    stair_links: list[list[str]] = []  # [lower place, upper place] that must be connected vertically
    routes: list[Route] = []

    @model_validator(mode="after")
    def _check_refs(self):
        ids = {p.id for p in self.building.places}
        if len(ids) != len(self.building.places):
            raise ValueError("duplicate place ids")
        refs = [v.place for vid in self.videos for v in vid.visits] + list(self.expect_merged)
        refs += [x for pair in self.must_not_merge for x in pair[:2]] + [x for pair in self.stair_links for x in pair]
        refs += [x for r in self.routes for x in (r.from_, r.to)]
        if missing := sorted(set(refs) - ids):
            raise ValueError(f"unknown place ids referenced: {missing}")
        return self

    def place(self, pid: str) -> Place:
        return next(p for p in self.building.places if p.id == pid)

    @property
    def rooms(self) -> dict[str, str]:
        """room number (upper case) -> place id"""
        return {p.room.upper(): p.id for p in self.building.places if p.room}


def load(path: Path) -> Truth:
    data = json.loads(Path(path).read_text())
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {data.get('schema')!r}, expected {SCHEMA!r}")
    return Truth.model_validate(data)
