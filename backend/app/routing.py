"""Query-time navigation: pure graph math, no model calls.

The turn at node B is derived from headings, not stored per edge:
    delta = wrap(heading_out(B->C) - heading_in(A->B))
Walking an edge backwards uses headings + 180 deg and flips stairs up/down, so one recorded walk
gives correct directions in both directions, and junctions/loops need no special cases.
"""
from dataclasses import dataclass

import networkx as nx

from . import config
from .mapper import flip_vertical, wrap180
from .schemas import Edge, MapBundle, Node, RouteResponse, Step
from .store import frame_url

NO_PATH_MESSAGE = "No valid path found between selected locations. Please check graph connections in Map Editor."

TURN_TEXT = {
    "go_straight": "Continue straight", "slight_left": "Bear slightly left", "slight_right": "Bear slightly right",
    "turn_left": "Turn left", "turn_right": "Turn right", "u_turn": "Turn around",
}
VERTICAL_TEXT = {
    "stairs_up": "take the stairs up", "stairs_down": "take the stairs down", "elevator": "take the elevator",
    "ramp_up": "go up the ramp", "ramp_down": "go down the ramp",
}


@dataclass
class Traversal:
    edge: Edge
    heading_out: float
    heading_in: float
    vertical: str
    frames: list[str] | None  # photos in travel order, None if never walked this way


def quantize_turn(delta: float) -> str:
    d = wrap180(delta)
    a = abs(d)
    if a < 25:
        return "go_straight"
    if a < 60:
        return "slight_right" if d > 0 else "slight_left"
    if a <= 135:
        return "turn_right" if d > 0 else "turn_left"
    return "u_turn"


def traverse(edge: Edge, u: str) -> Traversal:
    if edge.from_node == u:
        return Traversal(edge, edge.heading_out, edge.heading_in, edge.vertical, edge.frames_fwd or None)
    return Traversal(edge, (edge.heading_in + 180) % 360, (edge.heading_out + 180) % 360,
                     flip_vertical(edge.vertical), edge.frames_rev)


def _cost(e: Edge) -> float:
    """Walking metres, plus effort for stairs and a small per-place penalty (prefers simpler routes)."""
    cost = (e.length_m or config.DEFAULT_EDGE_M) + config.TURN_COST_M
    if e.vertical in ("stairs_up", "stairs_down"):
        cost += config.STAIRS_COST_M
    return cost


def build_graph(bundle: MapBundle) -> nx.Graph:
    g = nx.Graph()
    for n in bundle.nodes:
        g.add_node(n.id)
    for e in bundle.edges:
        if e.from_node in g and e.to_node in g:
            g.add_edge(e.from_node, e.to_node, edge=e, cost=_cost(e))
    return g


def _place(node: Node) -> str:
    """Landmark phrase that reads naturally after 'at': 'the glass doors with a plaque'."""
    d = node.description.strip().rstrip(".")
    if not d:
        return node.label
    if node.texts and node.label.lower() not in d.lower():
        return node.label  # a named place reads better than a description of it
    first, _, rest = d.partition(" ")
    if not any(ch.isdigit() for ch in first) and not first.isupper():
        first = first.lower()
    d = f"{first} {rest}".strip()
    return d if d.split(" ", 1)[0] in ("the", "a", "an") else f"the {d}"


def route(bundle: MapBundle, start: str, target: str) -> RouteResponse:
    nodes = {n.id: n for n in bundle.nodes}
    for nid in (start, target):
        if nid not in nodes:
            raise KeyError(nid)
    g = build_graph(bundle)
    path = nx.shortest_path(g, start, target, weight="cost")  # shortest walk; raises nx.NetworkXNoPath
    bid = bundle.building_id

    if len(path) == 1:
        n = nodes[start]
        return RouteResponse(node_path=path, steps=[Step(
            step_index=0, node_id=n.id, label=n.label, frame_url=frame_url(bid, n.frame_id),
            action="destination_reached", instruction_text=f"You are already at {n.label}.", floor=n.floor,
        )])

    trav = [traverse(g.edges[u, v]["edge"], u) for u, v in zip(path, path[1:])]
    steps: list[Step] = []

    def photo(node: Node, frames: list[str] | None, index: int) -> tuple[str | None, str | None]:
        if frames:
            return frame_url(bid, frames[index]), None
        return frame_url(bid, node.frame_id), "Photo was taken facing the other way"

    # Step 0: orient at the start.
    first, t0 = nodes[path[0]], trav[0]
    url, note = photo(first, t0.frames, 0)
    nxt = nodes[path[1]]
    move = VERTICAL_TEXT.get(t0.vertical)
    if move and nxt.floor != first.floor:
        move += f" to floor {nxt.floor}"
    text = f"Start at {first.label}. Face the direction shown and " + (
        f"{move} toward {nxt.label}." if move else f"walk forward toward {nxt.label}.")
    steps.append(Step(step_index=0, node_id=first.id, label=first.label, frame_url=url, view_note=note,
                      action=t0.vertical if move else "start", vertical=t0.vertical,
                      instruction_text=text, floor=first.floor))

    # Intermediate decision points.
    for i in range(1, len(path) - 1):
        node, arrive, leave = nodes[path[i]], trav[i - 1], trav[i]
        delta = wrap180(leave.heading_out - arrive.heading_in)
        turn = quantize_turn(delta)
        url, note = photo(node, arrive.frames, -1)
        # A floor link joined two separate walks: no measured direction across it, so say where to head.
        if arrive.edge.source == "merge" and arrive.vertical != "none":
            word = "top" if arrive.vertical in ("stairs_up", "ramp_up") else "bottom"
            where = f"{'Leaving the elevator' if arrive.vertical == 'elevator' else f'At the {word} of the stairs'}"
            steps.append(Step(
                step_index=i, node_id=node.id, label=node.label, frame_url=url, view_note=note,
                action="go_straight", vertical="none", floor=node.floor,
                instruction_text=f"{where} ({node.label}, floor {node.floor}), head toward {nodes[path[i + 1]].label} "
                                 "as shown in the photo.",
            ))
            continue
        if leave.edge.source == "merge" and leave.vertical != "none":
            turn, delta = "go_straight", 0.0
        move = VERTICAL_TEXT.get(leave.vertical)
        place, upcoming = _place(node), nodes[path[i + 1]]
        if move and upcoming.floor != node.floor:
            move += f" to floor {upcoming.floor}"
        if turn == "go_straight":
            text = f"At {place}, {move}." if move else f"Continue straight past {place}."
        else:
            text = f"{TURN_TEXT[turn]} at {place}" + (f" and {move}." if move else ".")
        if upcoming.id != path[-1] and upcoming.texts and not move:
            text = text.rstrip(".") + f", toward {upcoming.label}."
        steps.append(Step(
            step_index=i, node_id=node.id, label=node.label, frame_url=url, view_note=note,
            action=leave.vertical if (move and turn == "go_straight") else turn, vertical=leave.vertical,
            instruction_text=text, floor=node.floor, turn_degrees=round(delta, 1),
        ))

    last = nodes[path[-1]]
    url, note = photo(last, trav[-1].frames, -1)
    steps.append(Step(step_index=len(path) - 1, node_id=last.id, label=last.label, frame_url=url, view_note=note,
                      action="destination_reached", instruction_text=f"You have arrived at {last.label}.",
                      floor=last.floor))
    return RouteResponse(steps=steps, node_path=path)
