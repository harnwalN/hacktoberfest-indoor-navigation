"""Pydantic models shared by the pipeline, the API and (by convention) the frontend."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, field_validator

from . import config

Vertical = Literal["none", "stairs_up", "stairs_down", "elevator", "ramp_up", "ramp_down"]
PlaceType = Literal[
    "room", "intersection", "hallway", "entrance", "stairs", "elevator", "exit", "restroom", "landmark"
]
Action = Literal[
    "start", "go_straight", "slight_left", "slight_right", "turn_left", "turn_right", "u_turn",
    "stairs_up", "stairs_down", "elevator", "ramp_up", "ramp_down", "destination_reached",
]

FLIP_VERTICAL: dict[str, str] = {
    "none": "none", "stairs_up": "stairs_down", "stairs_down": "stairs_up",
    "elevator": "elevator", "ramp_up": "ramp_down", "ramp_down": "ramp_up",
}


# ---------- Gemma output ----------
class GemmaFrame(BaseModel):
    """Structured JSON Gemma returns for one keyframe (this exact schema is sent to the model).
    Evidence (texts, room_numbers) comes before the has_sign verdict: small models judge better that way."""
    texts: list[str] = []
    room_numbers: list[str] = []
    has_sign: bool = False
    suggested_label: str = ""
    place_type: PlaceType = "hallway"
    stairs: Literal["none", "ahead_up", "ahead_down", "on_stairs_up", "on_stairs_down"] = "none"
    floor_hint: str = ""
    description: str = Field("", description="Short landmark phrase that completes 'Turn left at ___'")


class FrameAnalysis(GemmaFrame):
    """Gemma's answer after OCR grounding, plus the OCR evidence."""
    ocr: list[str] = []
    ocr_rev: int = 0  # ocr.REVISION the `ocr` lines were read with (0: before revisions were recorded)
    grounded: bool = False
    _model_used: str | None = PrivateAttr(default=None)  # Gemma model the answer came from (not serialized)


class SamePlace(BaseModel):
    # Evidence first: generating these before the verdict makes small models far more reliable.
    image1_sign_text: list[str] = []  # each image's signs read on its own, so the model does not mix them up
    image2_sign_text: list[str] = []
    shared_details: list[str] = []
    conflicting_details: list[str] = []
    same_place: bool
    confidence: float
    reason: str


# ---------- Pipeline intermediates ----------
class Keyframe(BaseModel):
    frame_id: str
    video_id: str
    t: float
    kind: str  # start | end | turn_start | turn_end | dwell | glance | scene | heartbeat | manual
    t_arrive: float | None = None  # for turns/glances: when the rotation began / ended
    t_depart: float | None = None
    heading: float
    pitch: float = 0.0
    score: float = 1.0
    soft: bool = False
    dhash: str = ""
    analysis: FrameAnalysis | None = None


# ---------- Map bundle ----------
class Node(BaseModel):
    id: str
    label: str
    type: str = "hallway"
    frame_id: str | None = None
    frame_heading: float | None = None  # building-frame heading the node photo was taken facing
    texts: list[str] = []  # everything read here, including from merged places
    sign_texts: list[str] = []  # wayfinding text from this place's own photo (used for identity matching)
    description: str = ""
    floor: int = 1
    floor_source: Literal["user", "rooms", "linked", "default"] = "default"  # how sure we are of `floor`
    # Stairs/elevator access point, from what the walker did here (climbed a flight, rode the elevator),
    # even when the photo looks like a plain hallway. None = unknown (also every pre-existing map.json).
    vertical: Literal["stairs", "elevator"] | None = None
    verified: bool = False
    soft: bool = False
    video_id: str | None = None
    # Account.id of the contributor who added this place's photo by hand (POST /frames/manual); None for walk places.
    # Server-owned: PUT /map keeps the value already on disk, whatever the editor sends.
    contributor_id: str | None = None
    t: float | None = None
    x: float | None = None  # metres east of the map origin (solved from all connections)
    y: float | None = None  # metres north


class Edge(BaseModel):
    id: str
    from_node: str
    to_node: str
    # Building-frame headings (deg, clockwise, Manhattan-snapped) used for routing.
    heading_out: float  # direction of travel when leaving from_node
    heading_in: float  # direction of travel when arriving at to_node
    heading_out_raw: float | None = None
    heading_in_raw: float | None = None
    vertical: Vertical = "none"
    length_m: float | None = None  # walking distance estimated from video time x walking speed
    frames_fwd: list[str] = []  # ordered photos walking from -> to: [departure ... approach]
    frames_rev: list[str] | None = None  # ordered photos walking to -> from, if ever walked
    source: Literal["walk", "merge", "manual"] = "walk"
    verified: bool = False
    video_id: str | None = None  # walk that recorded this edge (its headings live in that walk's frame)
    inferred_floor_change: bool = False  # floor changed by room numbers but no stairs were seen


class SuggestedMerge(BaseModel):
    id: str
    keep: str
    remove: str
    confidence: float
    reason: str
    signals: dict = {}


class VideoInfo(BaseModel):
    id: str
    filename: str
    duration: float = 0.0
    keyframes: int = 0
    has_sensors: bool = False
    status: str = "queued"
    start_floor: int | None = None  # floor the walk starts on, as entered by the uploader
    floor_offset: int = 0  # floor actually applied to this walk's places (start_floor, or inferred)
    anchored: bool = False  # headings aligned with the building frame (first walk, or matched geometrically)
    contributor_id: str | None = None  # Account.id of whoever uploaded the walk; None = not recorded (every pre-accounts walk)
    # Uploader's sign-in email, kept on the walk. Private: never in a public map response (see VideoPublic).
    # "unknown" for walks uploaded before this was recorded.
    uploader_email: str = "unknown"
    created_at: str | None = None  # ISO 8601 UTC, set at upload; None for walks from before this was recorded
    moderation_status: Literal["pending", "approved", "rejected", "hidden"] = "pending"
    moderation_notes: str | None = None  # reason or note: automatic (jobs.py) or from a moderator (store.set_video_moderation)
    # Owner who made the last moderation decision (owner email, or "owner-token"). None = automatic only.
    # While set, automatic moderation does not change moderation_status (store.set_auto_moderation).
    moderated_by: str | None = None


# Campus a building belongs to. Groundwork for multi-campus support; nothing branches on it yet.
DEFAULT_CAMPUS = "ucsd"


class MapBundle(BaseModel):
    building_id: str
    name: str = ""
    campus: str = Field(DEFAULT_CAMPUS, min_length=1)  # plain string; a map.json without it loads as DEFAULT_CAMPUS
    nodes: list[Node] = []
    edges: list[Edge] = []
    suggested_merges: list[SuggestedMerge] = []
    videos: list[VideoInfo] = []
    dominant_axis_deg: float = 0.0
    processed_with: dict = {}


# ---------- Contributor accounts (store.py, accounts.json) ----------
class Account(BaseModel):
    """Email-identified contributor. No password: sign-in is by emailed link (auth.py)."""
    id: str  # uuid4 hex; the key other records (VideoInfo.contributor_id) refer to
    email: str  # normalized: stripped and lowercased (store.normalize_email)
    created_at: str  # ISO 8601, UTC
    moderator_of: list[str] = []  # building ids this account created (creator = moderator) or was granted
    blocked: bool = False  # set by an owner: no uploads (403 on POST /videos). Sign-in and reads still work.
    last_seen: str | None = None  # ISO 8601 UTC of the last session use (updated at most once an hour)


# ---------- Map as returned by the API ----------
# The photo URLs below are built per response (router._map_out) and never stored: store.save_map drops them.
class NodeOut(Node):
    frame_url: str | None = None  # /bfiles URL of frame_id, or null


class EdgeOut(Edge):
    frames_fwd_urls: list[str | None] = []  # same order as frames_fwd
    frames_rev_urls: list[str | None] | None = None  # same order as frames_rev


class VideoPublic(BaseModel):
    """A walk as a public map response shows it: VideoInfo without the uploader's email and the moderator's email."""
    id: str
    filename: str
    duration: float = 0.0
    keyframes: int = 0
    has_sensors: bool = False
    status: str = "queued"
    start_floor: int | None = None
    floor_offset: int = 0
    anchored: bool = False
    contributor_id: str | None = None
    created_at: str | None = None
    moderation_status: Literal["pending", "approved", "rejected", "hidden"] = "pending"
    moderation_notes: str | None = None


class MapOut(MapBundle):
    nodes: list[NodeOut] = []
    edges: list[EdgeOut] = []
    videos: list[VideoPublic] = []


class FloorMapOut(BaseModel):
    """GET /buildings/{id}/floors/{floor}/map: one floor of the map, with the same photo URLs as MapOut."""
    building_id: str
    floor: int
    nodes: list[NodeOut] = []  # places whose floor is this floor
    edges: list[EdgeOut] = []  # connections with both ends on this floor
    links: list[EdgeOut] = []  # connections with one end on this floor and one on another (stairs, lifts, ramps)


# ---------- API payloads ----------
class MapEdit(BaseModel):
    """PUT /buildings/{id}/map body. nodes and edges are required (no defaults) so a partial body is a 422
    instead of silently wiping the map; any other MapBundle fields the editor sends along are ignored."""
    model_config = ConfigDict(extra="ignore")
    nodes: list[Node]
    edges: list[Edge]


class CreateBuilding(BaseModel):
    name: str


class RouteRequest(BaseModel):
    start_node_id: str
    target_node_id: str
    avoid_stairs: bool = False  # True: only flat walks, ramps and elevators (400 if the only way climbs stairs)


class Step(BaseModel):
    step_index: int
    node_id: str
    label: str
    frame_url: str | None
    action: Action
    vertical: Vertical = "none"
    instruction_text: str
    view_note: str | None = None
    floor: int = 1
    turn_degrees: float = 0.0


class RouteResponse(BaseModel):
    steps: list[Step]
    node_path: list[str]
    step_free: bool  # True when no stairs are used (always computed, whether or not avoid_stairs was asked)


class VideoUpdate(BaseModel):
    start_floor: int = Field(ge=1, le=config.MAX_FLOOR)


class MergeDecision(BaseModel):
    accept: bool


class ManualMerge(BaseModel):
    keep: str
    remove: str


# ---------- problem reports (reports.py, router.py) ----------
ReportKind = Literal["wrong_direction", "place_missing", "sign_changed", "photo_wrong", "blocked", "other"]
ReportStatus = Literal["open", "resolved"]


def _stripped_note(value: str | None, limit: int) -> str | None:
    """Strip surrounding whitespace, then check the length. An empty note is stored as no note."""
    if value is None:
        return None
    value = value.strip()
    if len(value) > limit:
        raise ValueError(f"must be at most {limit} characters (after stripping spaces)")
    return value or None


class ReportCreate(BaseModel):
    kind: ReportKind
    note: str | None = None
    node_id: str | None = Field(None, min_length=1, max_length=64)  # must exist on the map (checked in the route)
    step_index: int | None = Field(None, strict=True, ge=0, le=500)  # position in the route the walker followed
    from_node: str | None = Field(None, min_length=1, max_length=64)  # must exist on the map (checked in the route)
    to_node: str | None = Field(None, min_length=1, max_length=64)  # must exist on the map (checked in the route)

    @field_validator("note")
    @classmethod
    def _note(cls, value: str | None) -> str | None:
        return _stripped_note(value, 500)


class ReportStatusUpdate(BaseModel):
    status: ReportStatus
    resolution_note: str | None = None

    @field_validator("resolution_note")
    @classmethod
    def _resolution_note(cls, value: str | None) -> str | None:
        return _stripped_note(value, 300)


# ---------- owner moderation and contributor views (owner.py, owner_router.py) ----------
class ModerationDecision(BaseModel):
    """POST /api/owner/videos/{building_id}/{video_id}/moderation. approve puts the walk on the map (rebuilding it when
    needed); reject and hide take it off the map (hide is a takedown; reject is a declined walk)."""
    action: Literal["approve", "reject", "hide"]
    note: str | None = None

    @field_validator("note")
    @classmethod
    def _note(cls, value: str | None) -> str | None:
        return _stripped_note(value, 1000)


class UserBlock(BaseModel):
    blocked: bool
