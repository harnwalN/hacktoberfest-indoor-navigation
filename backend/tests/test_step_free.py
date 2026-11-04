"""Step-free routing (avoid_stairs): only flat walks, ramps and elevators.

Hand-written two-floor maps. Lengths make the stairs the cheaper way in normal routing, so avoid_stairs must
pick a different route, and the tests can tell the two apart.
"""
import networkx as nx
import pytest
from fastapi.testclient import TestClient

from app import config, routing, store
from app.main import app
from app.schemas import Edge, MapBundle, Node


def _n(nid, floor, label=None, type="hallway", vertical=None):
    return Node(id=nid, label=label or f"Place {nid}", type=type, floor=floor, vertical=vertical,
                frame_id=f"{nid}.jpg")


def _e(i, a, b, length, vertical="none"):
    return Edge(id=f"e{i}", from_node=a, to_node=b, heading_out=0, heading_in=0, length_m=length,
                vertical=vertical, frames_fwd=[f"{a}_{b}.jpg"], frames_rev=[f"{b}_{a}.jpg"])


def _bundle(nodes, edges):
    return MapBundle(building_id="test", nodes=nodes, edges=edges)


def _actions(r):
    return [s.action for s in r.steps]


# Floor 1: A (start), B, S1 (stairs landing). Floor 2: T1 (top of the stairs), U (target).
# Stairs route A-B-S1-T1-U costs about 44; every other way is much longer.
def _stairs_nodes():
    return [_n("A", 1, "Room 101"), _n("B", 1), _n("S1", 1, "Stairs", "stairs", "stairs"),
            _n("T1", 2, "Stairs (2)", "stairs", "stairs"), _n("U", 2, "Room 208")]


def _stairs_edges():
    return [_e(1, "A", "B", 5), _e(2, "B", "S1", 5), _e(3, "S1", "T1", 6, "stairs_up"), _e(4, "T1", "U", 5)]


@pytest.fixture
def stairs_only():
    return _bundle(_stairs_nodes(), _stairs_edges())


@pytest.fixture
def two_floors():
    """Stairs plus an elevator 40 m away: normal routing takes the stairs, avoid_stairs the elevator."""
    nodes = _stairs_nodes() + [_n("E1", 1, "Elevator", "elevator", "elevator"),
                               _n("E2", 2, "Elevator (2)", "elevator", "elevator")]
    edges = _stairs_edges() + [_e(5, "A", "E1", 40), _e(6, "E1", "E2", 6, "elevator"), _e(7, "E2", "U", 40)]
    return _bundle(nodes, edges)


@pytest.fixture
def stairs_and_ramp():
    """Stairs plus a ramp 40 m away: normal routing takes the stairs, avoid_stairs the ramp."""
    nodes = _stairs_nodes() + [_n("R1", 1, "Ramp foot"), _n("R2", 2, "Ramp top")]
    edges = _stairs_edges() + [_e(5, "A", "R1", 40), _e(6, "R1", "R2", 20, "ramp_up"), _e(7, "R2", "U", 40)]
    return _bundle(nodes, edges)


# ---------- routing ----------
def test_normal_route_takes_the_shorter_stairs(two_floors):
    r = routing.route(two_floors, "A", "U")
    assert r.node_path == ["A", "B", "S1", "T1", "U"]
    assert "stairs_up" in _actions(r)
    assert r.step_free is False


def test_avoid_stairs_takes_the_elevator(two_floors):
    r = routing.route(two_floors, "A", "U", avoid_stairs=True)
    assert r.node_path == ["A", "E1", "E2", "U"]
    assert "elevator" in _actions(r)
    assert not any(a.startswith("stairs") for a in _actions(r))
    assert r.step_free is True


def test_avoid_stairs_only_stairs_raises_step_free_message(stairs_only):
    with pytest.raises(routing.NoStepFreePath) as exc:
        routing.route(stairs_only, "A", "U", avoid_stairs=True)
    assert str(exc.value) == routing.NO_STEP_FREE_MESSAGE
    assert routing.NO_STEP_FREE_MESSAGE == "No step-free route between these places. The only way uses stairs."


def test_normal_routing_still_works_when_only_stairs_connect(stairs_only):
    r = routing.route(stairs_only, "A", "U")
    assert r.node_path == ["A", "B", "S1", "T1", "U"]
    assert _actions(r)[2] == "stairs_up"


def test_reverse_direction_going_down_is_excluded(stairs_only, two_floors):
    down = routing.route(stairs_only, "U", "A")
    assert down.node_path == ["U", "T1", "S1", "B", "A"]
    assert "stairs_down" in _actions(down) and down.step_free is False
    with pytest.raises(routing.NoStepFreePath):
        routing.route(stairs_only, "U", "A", avoid_stairs=True)
    # going down by elevator is still fine
    assert routing.route(two_floors, "U", "A", avoid_stairs=True).node_path == ["U", "E2", "E1", "A"]


def test_stairs_are_not_reachable_even_one_flight_up(stairs_only):
    with pytest.raises(routing.NoStepFreePath):
        routing.route(stairs_only, "A", "T1", avoid_stairs=True)  # T1 is only reachable by the flight
    with pytest.raises(routing.NoStepFreePath):
        routing.route(stairs_only, "U", "S1", avoid_stairs=True)  # ...and so is S1 from floor 2


def test_flat_route_on_one_floor_is_step_free(stairs_only):
    r = routing.route(stairs_only, "A", "B", avoid_stairs=True)
    assert r.node_path == ["A", "B"] and r.step_free is True
    assert routing.route(stairs_only, "A", "B").step_free is True  # normal route, same answer


def test_start_or_end_at_a_stairs_place_is_fine_without_the_stairs(stairs_only):
    # S1 is a stairs-type place on floor 1: leaving it on the flat is allowed
    r = routing.route(stairs_only, "S1", "B", avoid_stairs=True)
    assert r.node_path == ["S1", "B"] and r.step_free is True
    # T1 is the top of the flight: the flat walk from it to U is allowed
    r = routing.route(stairs_only, "T1", "U", avoid_stairs=True)
    assert r.node_path == ["T1", "U"] and r.step_free is True


def test_same_place_is_step_free(stairs_only):
    r = routing.route(stairs_only, "A", "A", avoid_stairs=True)
    assert r.step_free is True and _actions(r) == ["destination_reached"]


def test_ramp_counts_as_step_free(stairs_and_ramp):
    up = routing.route(stairs_and_ramp, "A", "U")
    assert up.node_path == ["A", "B", "S1", "T1", "U"] and up.step_free is False

    r = routing.route(stairs_and_ramp, "A", "U", avoid_stairs=True)
    assert r.node_path == ["A", "R1", "R2", "U"]
    assert "ramp_up" in _actions(r) and r.step_free is True


def test_ramp_going_down_is_step_free_too(stairs_and_ramp):
    r = routing.route(stairs_and_ramp, "U", "A", avoid_stairs=True)
    assert r.node_path == ["U", "R2", "R1", "A"]
    assert "ramp_down" in _actions(r) and r.step_free is True


def test_stairs_and_elevator_between_the_same_two_places():
    """Two edges on one pair: the stairs one must not hide the elevator one from avoid_stairs (nx.Graph keeps
    one edge per pair, so stair edges are dropped before they are added)."""
    for order in ("stairs_first", "elevator_first"):
        stairs, lift = _e(1, "A", "X", 2, "stairs_up"), _e(2, "A", "X", 30, "elevator")
        edges = [stairs, lift] if order == "stairs_first" else [lift, stairs]
        b = _bundle([_n("A", 1), _n("X", 2)], edges)
        r = routing.route(b, "A", "X", avoid_stairs=True)
        assert _actions(r)[0] == "elevator", order
        assert r.step_free is True, order


def test_floor_change_without_a_marked_vertical_is_not_step_free(stairs_only):
    """A flat-marked edge that changes floor is an unlabelled climb: never step-free, in either mode."""
    stairs_only.edges[2].vertical = "none"  # S1 -> T1 changes floor but is marked flat
    r = routing.route(stairs_only, "A", "U")
    assert r.node_path == ["A", "B", "S1", "T1", "U"] and r.step_free is False
    with pytest.raises(routing.NoStepFreePath):
        routing.route(stairs_only, "A", "U", avoid_stairs=True)


def test_no_path_at_all_keeps_the_normal_error(stairs_only):
    stairs_only.nodes.append(_n("Z", 1, "Island"))
    with pytest.raises(nx.NetworkXNoPath) as exc:
        routing.route(stairs_only, "A", "Z", avoid_stairs=True)
    assert not isinstance(exc.value, routing.NoStepFreePath)


def test_unknown_node_is_still_a_key_error(stairs_only):
    with pytest.raises(KeyError):
        routing.route(stairs_only, "A", "nope", avoid_stairs=True)


def test_normal_routes_are_unchanged_by_the_new_option(two_floors):
    """avoid_stairs=False gives exactly the old answer: same path, steps and distances."""
    old = routing.route(two_floors, "A", "U")
    again = routing.route(two_floors, "A", "U", avoid_stairs=False)
    assert old.node_path == again.node_path
    assert [s.model_dump() for s in old.steps] == [s.model_dump() for s in again.steps]


# ---------- API ----------
@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BUILDINGS_DIR", tmp_path / "buildings")
    return TestClient(app)


def _building(name, bundle):
    b = store.create_building(name)
    bundle.building_id = b.building_id
    store.save_map(bundle)
    return b.building_id


def test_route_request_without_the_field_is_a_normal_route(api, two_floors):
    bid = _building("Two Floors", two_floors)
    r = api.post(f"/api/buildings/{bid}/route", json={"start_node_id": "A", "target_node_id": "U"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["node_path"] == ["A", "B", "S1", "T1", "U"]
    assert body["step_free"] is False
    assert set(body) == {"steps", "node_path", "step_free"}


def test_route_with_avoid_stairs_false_matches_the_default(api, two_floors):
    bid = _building("Two Floors", two_floors)
    base = {"start_node_id": "A", "target_node_id": "U"}
    default = api.post(f"/api/buildings/{bid}/route", json=base).json()
    explicit = api.post(f"/api/buildings/{bid}/route", json={**base, "avoid_stairs": False}).json()
    assert default == explicit


def test_route_with_avoid_stairs_true_uses_the_elevator(api, two_floors):
    bid = _building("Two Floors", two_floors)
    r = api.post(f"/api/buildings/{bid}/route",
                 json={"start_node_id": "A", "target_node_id": "U", "avoid_stairs": True})
    assert r.status_code == 200, r.text
    assert r.json()["node_path"] == ["A", "E1", "E2", "U"]
    assert r.json()["step_free"] is True


def test_route_with_avoid_stairs_on_a_stairs_only_building_is_400(api, stairs_only):
    bid = _building("Stairs Only", stairs_only)
    r = api.post(f"/api/buildings/{bid}/route",
                 json={"start_node_id": "A", "target_node_id": "U", "avoid_stairs": True})
    assert r.status_code == 400
    assert r.json()["detail"] == routing.NO_STEP_FREE_MESSAGE
    normal = api.post(f"/api/buildings/{bid}/route", json={"start_node_id": "A", "target_node_id": "U"})
    assert normal.status_code == 200 and normal.json()["step_free"] is False


def test_route_with_no_path_at_all_keeps_the_old_message(api, stairs_only):
    stairs_only.nodes.append(_n("Z", 1, "Island"))
    bid = _building("Island", stairs_only)
    r = api.post(f"/api/buildings/{bid}/route",
                 json={"start_node_id": "A", "target_node_id": "Z", "avoid_stairs": True})
    assert r.status_code == 400 and r.json()["detail"] == routing.NO_PATH_MESSAGE


def test_route_avoid_stairs_must_be_a_boolean_and_nodes_must_exist(api, two_floors):
    bid = _building("Two Floors", two_floors)
    bad = api.post(f"/api/buildings/{bid}/route",
                   json={"start_node_id": "A", "target_node_id": "U", "avoid_stairs": "maybe"})
    assert bad.status_code == 422
    unknown = api.post(f"/api/buildings/{bid}/route",
                       json={"start_node_id": "A", "target_node_id": "zz", "avoid_stairs": True})
    assert unknown.status_code == 404


def test_step_free_route_is_asked_with_avoid_stairs(api, two_floors):
    bid = _building("Two Floors", two_floors)
    ok = api.post(f"/api/buildings/{bid}/route",
                  json={"start_node_id": "A", "target_node_id": "U", "avoid_stairs": True})
    assert ok.status_code == 200, ok.text
    assert ok.json()["node_path"] == ["A", "E1", "E2", "U"] and ok.json()["step_free"] is True


def test_stairs_only_building_gets_the_step_free_message(api, stairs_only):
    sid = _building("Stairs Only", stairs_only)
    r = api.post(f"/api/buildings/{sid}/route",
                 json={"start_node_id": "A", "target_node_id": "U", "avoid_stairs": True})
    assert r.status_code == 400 and r.json()["detail"] == routing.NO_STEP_FREE_MESSAGE
