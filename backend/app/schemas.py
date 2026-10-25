"""Pydantic models shared by the pipeline, the API and (by convention) the frontend."""
from typing import Literal

from pydantic import BaseModel, Field

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
    """Structured JSON Gemma returns for one keyframe (this exact schema is sent to the model)."""
    has_sign: bool = False
    texts: list[str] = []
    room_numbers: list[str] = []
    suggested_label: str = ""
    place_type: PlaceType = "hallway"
    stairs: Literal["none", "ahead_up", "ahead_down", "on_stairs_up", "on_stairs_down"] = "none"
    floor_hint: str = ""
    description: str = Field("", description="Short landmark phrase that completes 'Turn left at ___'")


class FrameAnalysis(GemmaFrame):
    """Gemma's answer after OCR grounding, plus the OCR evidence."""
    ocr: list[str] = []
    grounded: bool = False


class SamePlace(BaseModel):
    # Evidence first: generating these before the verdict makes small models far more reliable.
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
    texts: list[str] = []
    description: str = ""
    floor: int = 0
    verified: bool = False
    soft: bool = False
    video_id: str | None = None
    t: float | None = None


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
    frames_fwd: list[str] = []  # ordered photos walking from -> to: [departure ... approach]
    frames_rev: list[str] | None = None  # ordered photos walking to -> from, if ever walked
    source: Literal["walk", "merge", "manual"] = "walk"
    verified: bool = False


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


class MapBundle(BaseModel):
    building_id: str
    name: str = ""
    nodes: list[Node] = []
    edges: list[Edge] = []
    suggested_merges: list[SuggestedMerge] = []
    videos: list[VideoInfo] = []
    dominant_axis_deg: float = 0.0
    processed_with: dict = {}


# ---------- API payloads ----------
class CreateBuilding(BaseModel):
    name: str


class RouteRequest(BaseModel):
    start_node_id: str
    target_node_id: str


class Step(BaseModel):
    step_index: int
    node_id: str
    label: str
    frame_url: str | None
    action: Action
    vertical: Vertical = "none"
    instruction_text: str
    view_note: str | None = None
    floor: int = 0
    turn_degrees: float = 0.0


class RouteResponse(BaseModel):
    steps: list[Step]
    node_path: list[str]


class MergeDecision(BaseModel):
    accept: bool


class ManualMerge(BaseModel):
    keep: str
    remove: str
