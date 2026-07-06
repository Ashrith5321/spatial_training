"""Ground-truth scene-graph extraction for the eval loop.

Reuses the habitat-lab Env's already-running simulator (env.sim) instead of
spinning up a second habitat_sim.Simulator (as Navigation_scene_graph/gt_scene_graph
does), since the eval loop's HabitatSimSemanticSensorConfig already renders the
semantic image every step for free.

This is a real scene graph, not just a per-step FOV list: rooms are nodes,
ground-truth objects are children of the room they physically belong to, and
edges between rooms record that the agent has walked directly between them.
The graph only ever contains what's been observed through the semantic sensor
during the episode so far -- it does not leak unobserved ground truth.

Ported/adapted from:
  Navigation_scene_graph/gt_scene_graph/src/habitat/gt_addition.py (HabitatGT.objects_in_fov)
  Navigation_scene_graph/gt_scene_graph/src/graph/scene_graph.py (room/object/edge design)
"""

from typing import Dict, List, Optional

import numpy as np

try:  # habitat-lab is present in the eval env, but not in the bare habitat_sim (mcoconav) env
    from habitat.utils.geometry_utils import quaternion_rotate_vector
except Exception:  # pure-numpy fallback so the graph code runs against a raw habitat_sim too
    def quaternion_rotate_vector(quat, v):
        """Rotate vector `v` by quaternion `quat` (a np.quaternion or [w,x,y,z])."""
        if hasattr(quat, "w"):
            w, x, y, z = quat.w, quat.x, quat.y, quat.z
        else:
            a = np.asarray(quat).ravel()
            w, x, y, z = a[0], a[1], a[2], a[3]
        v = np.asarray(v, dtype=np.float32)
        u = np.array([x, y, z], dtype=np.float32)
        return (2.0 * float(np.dot(u, v))) * u + (w * w - float(np.dot(u, u))) * v + 2.0 * w * np.cross(u, v)

_FORWARD_LOCAL = np.array([0.0, 0.0, -1.0], dtype=np.float32)
_RIGHT_LOCAL = np.array([1.0, 0.0, 0.0], dtype=np.float32)

_BEARINGS = [
    (22.5, "ahead"),
    (67.5, "ahead-right"),
    (112.5, "behind-right"),
    (157.5, "behind"),
]

# ObjectNav-21 canonical categories (the MP3D v1_21cat ObjectNav vocabulary).
# HM3D-Sem v0.2 only *navigates* to 6 of these (bed, chair, plant, sofa, toilet,
# tv_monitor), but its raw semantic annotations contain many of the others under
# free-text labels, so the scene graph maps raw labels onto this fixed vocabulary
# and drops everything else. Edit this set (and _CATEGORY_SYNONYMS) to change the
# vocabulary -- e.g. cut it down to the 6 HM3D goal categories.
_ALLOWED_CATEGORIES = frozenset({
    "bathtub", "bed", "cabinet", "chair", "chest_of_drawers", "clothes", "counter",
    "cushion", "fireplace", "gym_equipment", "picture", "plant", "seating", "shower",
    "sink", "sofa", "stool", "table", "toilet", "towel", "tv_monitor",
})

# Raw HM3D-Sem label (normalized: lowercased, underscores->spaces) -> canonical-21.
# Covers the common HM3D names that don't match a canonical token verbatim; without
# them goal objects like "couch"/"tv" would be silently dropped by the whitelist.
_CATEGORY_SYNONYMS = {
    "couch": "sofa",
    "tv": "tv_monitor", "television": "tv_monitor", "monitor": "tv_monitor",
    "armchair": "chair", "dining chair": "chair", "office chair": "chair",
    "pillow": "cushion",
    "painting": "picture", "photo": "picture", "art frame": "picture",
    "kitchen cabinet": "cabinet", "bathroom cabinet": "cabinet",
    "drawer": "chest_of_drawers", "drawers": "chest_of_drawers",
    "chest of drawers": "chest_of_drawers",
    "desk": "table", "dining table": "table", "coffee table": "table",
    "side table": "table",
    "flowerpot": "plant", "potted plant": "plant", "flower pot": "plant",
    "shower wall": "shower",
}


def _normalize_category(name: str) -> str:
    """Lowercase, drop surrounding whitespace, and collapse underscores/spaces."""
    return " ".join(name.strip().lower().replace("_", " ").split())


def _canonical_category(raw: str) -> Optional[str]:
    """Map a raw HM3D-Sem label onto the canonical ObjectNav-21 vocabulary, or
    None if it isn't one of the 21 (so it's excluded from the scene graph)."""
    norm = _normalize_category(raw)
    canon = _CATEGORY_SYNONYMS.get(norm)
    if canon is not None:
        return canon
    token = norm.replace(" ", "_")
    return token if token in _ALLOWED_CATEGORIES else None


def build_id_to_object(sim) -> Dict[int, object]:
    """int instance-id -> habitat semantic object.

    The semantic sensor image encodes each pixel as an object's integer instance
    id; that integer is the suffix of `obj.id` (e.g. "chair_42" -> 42).
    """
    id2obj: Dict[int, object] = {}
    for obj in sim.semantic_scene.objects:
        if obj is None:
            continue
        try:
            int_id = int(obj.id.split("_")[-1])
        except ValueError:
            continue
        id2obj[int_id] = obj
    return id2obj


def build_room_names(sim) -> Dict[int, str]:
    """int region-id -> human room name, when the scene annotates one.

    HM3D-Sem regions are often unlabeled; callers fall back to "room {id}".
    """
    names: Dict[int, str] = {}
    for region in getattr(sim.semantic_scene, "regions", []) or []:
        if region is None or region.id is None:
            continue
        try:
            rid = int(str(region.id).split("_")[-1])
        except ValueError:
            continue
        cat = getattr(region, "category", None)
        label = cat.name() if cat is not None else ""
        if label and label not in ("", "unknown"):
            names[rid] = label
    return names


def _region_id(obj) -> int:
    """Which room (region) `obj` belongs to, -1 if unknown. Region ids look like
    "_2"/"0_12"; the numeric suffix is the room index."""
    if obj.region is not None and obj.region.id is not None:
        try:
            return int(str(obj.region.id).split("_")[-1])
        except ValueError:
            pass
    return -1


def objects_in_fov(
    semantic_img: np.ndarray,
    id2obj: Dict[int, object],
    min_pixels: int = 50,
    top_k: int = 6,
) -> List[dict]:
    """Ground-truth objects visible in `semantic_img`, most-visible first."""
    ids, counts = np.unique(semantic_img, return_counts=True)

    results = []
    for inst_id, px in zip(ids.tolist(), counts.tolist()):
        if inst_id == 0 or px < min_pixels:  # 0 == unlabeled/background
            continue
        obj = id2obj.get(inst_id)
        if obj is None:
            continue
        raw = obj.category.name() if obj.category is not None else "unknown"
        category = _canonical_category(raw)
        if category is None:  # not one of the ObjectNav-21 -> keep it out of the graph
            continue
        results.append(
            {
                "id": inst_id,
                "category": category,
                "region_id": _region_id(obj),
                "pixel_count": px,
                "aabb_center": np.array(obj.aabb.center, dtype=np.float32),
            }
        )

    results.sort(key=lambda o: o["pixel_count"], reverse=True)
    return results[:top_k]


def _bearing_label(rel: np.ndarray, forward: np.ndarray, right: np.ndarray) -> str:
    """Bearing of `rel` (world offset to object) relative to the agent's own
    forward/right axes, projected onto the horizontal plane."""
    fwd_component = float(np.dot(rel, forward))
    right_component = float(np.dot(rel, right))
    angle = abs(np.degrees(np.arctan2(right_component, fwd_component)))  # 0=ahead, 180=behind
    side = "right" if right_component >= 0 else "left"

    for threshold, label in _BEARINGS:
        if angle <= threshold:
            return label if label in ("ahead", "behind") else f"{label.split('-')[0]}-{side}"
    return "behind"


def _agent_frame(agent_state):
    """(position, forward, right) world vectors for the agent's current pose."""
    position = np.array(agent_state.position, dtype=np.float32)
    forward = quaternion_rotate_vector(agent_state.rotation, _FORWARD_LOCAL)
    right = quaternion_rotate_vector(agent_state.rotation, _RIGHT_LOCAL)
    return position, forward, right


def _relative_position(center, position, forward, right):
    """Agent-relative ('~2.3m ahead-left', distance) for a world-space point."""
    rel = center - position
    dist = float(np.linalg.norm(rel))
    return f"~{dist:.1f}m {_bearing_label(rel, forward, right)}", dist


def describe_fov_text(objects: List[dict], agent_state) -> str:
    """One compact line listing nearby GT objects with distance + bearing."""
    if not objects:
        return "Visible now: none."

    position, forward, right = _agent_frame(agent_state)
    parts = []
    for obj in objects:
        rel_text, _ = _relative_position(obj["aabb_center"], position, forward, right)
        parts.append(f"{obj['category']} {rel_text}")

    return "Visible now: " + "; ".join(parts) + "."


class EpisodeSceneGraph:
    """Room-hierarchy scene graph accumulated over one episode.

    Rooms are nodes; each ground-truth object seen so far is a child of the room
    it belongs to (keyed by instance id, so repeat sightings don't duplicate).
    An edge is added between two rooms the first time the agent's dominant room
    changes from one to the other (traversability), mirroring
    gt_scene_graph/src/graph/scene_graph.py's update_scene_graph.
    """

    def __init__(self, max_objects_per_room: int = 8):
        self.max_objects_per_room = max_objects_per_room
        # room_id -> {instance_id: {"category": str, "center": np.ndarray}}
        self.rooms: Dict[int, Dict[int, dict]] = {}
        self.room_edges = set()  # {frozenset({room_a, room_b})}
        self.room_order: List[int] = []
        self.current_room: Optional[int] = None
        self.start_position: Optional[np.ndarray] = None  # set on first record_pose
        # --- accumulated for the 3D viewer export (see scene_graph_viewer.py) ---
        self.step = 0                                    # advanced once per observed step
        self.room_first_seen: Dict[int, int] = {}        # room_id -> step
        self.object_first_seen: Dict[int, int] = {}      # instance_id -> step
        self.edge_first_seen: Dict[frozenset, int] = {}  # frozenset(rooms) -> step
        self.trajectory: List[dict] = []                 # per step: {"pos":[x,y,z], "heading":[fx,fz]}

    def record_pose(self, agent_state) -> None:
        """Append the agent's current world pose to the trajectory (viewer export)."""
        position, forward, _ = _agent_frame(agent_state)
        if self.start_position is None:
            self.start_position = position.copy()
        self.trajectory.append({
            "pos": [float(position[0]), float(position[1]), float(position[2])],
            "heading": [float(forward[0]), float(forward[2])],
        })

    def advance(self) -> None:
        self.step += 1

    def update(self, objects: List[dict]) -> None:
        # Dominant room this step = region of the most-visible object with a known region
        # (objects are already sorted by pixel_count, most-visible first).
        room_id = next((o["region_id"] for o in objects if o["region_id"] != -1), None)

        if room_id is not None:
            if room_id not in self.room_order:
                self.room_order.append(room_id)
                self.room_first_seen[room_id] = self.step
            if self.current_room is not None and room_id != self.current_room:
                edge = frozenset((self.current_room, room_id))
                self.room_edges.add(edge)
                self.edge_first_seen.setdefault(edge, self.step)
            self.current_room = room_id

        for obj in objects:
            rid = obj["region_id"]
            if rid == -1:
                continue
            # Keyed by instance id, so repeat sightings refresh (not duplicate) the
            # remembered world position. Relative position is derived per-step in
            # describe() from the current agent pose, so it stays up to date.
            self.rooms.setdefault(rid, {})[obj["id"]] = {
                "category": obj["category"],
                "center": obj["aabb_center"],
            }
            self.object_first_seen.setdefault(obj["id"], self.step)

    def _room_objects_text(self, room_id, position, forward, right) -> str:
        """Objects remembered in `room_id`, each with its live agent-relative
        position, nearest first, capped at max_objects_per_room."""
        items = []
        for rec in self.rooms.get(room_id, {}).values():
            rel_text, dist = _relative_position(rec["center"], position, forward, right)
            items.append((dist, f"{rec['category']} {rel_text}"))
        items.sort(key=lambda t: t[0])
        return ", ".join(text for _, text in items[: self.max_objects_per_room])

    def describe(self, agent_state) -> str:
        if self.current_room is None:
            return ""

        # Recompute every object's position relative to the agent's current pose,
        # so each step reports fresh distances/bearings for the whole graph.
        position, forward, right = _agent_frame(agent_state)

        lines = [
            f"Current room (room_{self.current_room}) contains: "
            + self._room_objects_text(self.current_room, position, forward, right) + "."
        ]

        other_rooms = [r for r in self.room_order if r != self.current_room]
        if other_rooms:
            parts = [
                f"room_{r} ({self._room_objects_text(r, position, forward, right)})"
                for r in other_rooms
            ]
            lines.append("Previously visited rooms: " + "; ".join(parts) + ".")

        if self.room_edges:
            edges = ", ".join(
                f"room_{a}<->room_{b}" for a, b in (tuple(e) for e in self.room_edges)
            )
            lines.append("Room connections: " + edges + ".")

        return " ".join(lines)

    def _rel_xyz(self, world_pos) -> List[float]:
        """Position relative to the agent's starting point."""
        origin = self.start_position if self.start_position is not None else np.zeros(3)
        r = world_pos - origin
        return [round(float(r[0]), 2), round(float(r[1]), 2), round(float(r[2]), 2)]

    def to_toon(self, agent_state, visible_ids=None, max_objects: int = 14) -> str:
        position, _, _ = _agent_frame(agent_state)
        agent_xyz = self._rel_xyz(position)

        lines = [f"agent: room_{self.current_room} xyz={agent_xyz}"]
        for rid in self.room_order:
            objs = []
            for rec in self.rooms.get(rid, {}).values():
                xyz = self._rel_xyz(rec["center"])
                objs.append((rec["category"], xyz))
            parts = ", ".join(f"{cat} {xyz}" for cat, xyz in objs)
            lines.append(f"room_{rid}: {parts}")

        if self.room_edges:
            edges = sorted(f"room_{min(a,b)}-room_{max(a,b)}" for a, b in (tuple(e) for e in self.room_edges))
            lines.append(f"edges: {', '.join(edges)}")
        return "\n".join(lines)

    def to_dict(self, meta: dict, room_names: Optional[Dict[int, str]] = None) -> dict:
        """Serialize the graph + trajectory for the 3D viewer (scene_graph_viewer).

        Room node positions are the mean of their observed objects' AABB centers,
        so no extra sim query is needed. Everything is JSON-native (plain floats).
        """
        room_names = room_names or {}
        rooms, objects = [], []
        for rid in self.room_order:
            members = self.rooms.get(rid, {})
            if not members:
                continue
            centers = np.array([m["center"] for m in members.values()], dtype=np.float32)
            centroid = centers.mean(axis=0)
            rooms.append({
                "id": int(rid),
                "name": room_names.get(rid, f"room {rid}"),
                "pos": [float(centroid[0]), float(centroid[1]), float(centroid[2])],
                "disc": int(self.room_first_seen.get(rid, 0)),
            })
            for inst_id, m in members.items():
                c = m["center"]
                objects.append({
                    "cat": m["category"],
                    "room": int(rid),
                    "pos": [float(c[0]), float(c[1]), float(c[2])],
                    "disc": int(self.object_first_seen.get(inst_id, 0)),
                })

        edges = []
        for e in self.room_edges:
            a, b = tuple(e)
            edges.append([int(a), int(b), int(self.edge_first_seen.get(e, 0))])

        traj = [{"p": [t["pos"][0], t["pos"][2]], "h": t["heading"]} for t in self.trajectory]
        return {"meta": meta, "rooms": rooms, "objects": objects, "edges": edges, "traj": traj}


class GTSceneGraphProvider:
    """Per-scene id->object cache plus the per-episode room/object graph.

    Call `reset_episode()` at the start of each episode, then `describe(...)`
    each step to get the text block to append to the VLM's message alongside
    the image (this text becomes part of the growing KV cache each turn).
    """

    def __init__(self, min_pixels: int = 50, top_k: int = 6, text_format: str = "toon"):
        self.min_pixels = min_pixels
        self.top_k = top_k
        self.text_format = text_format  # "toon" (default, fed to the VLM) or "nl"
        self._scene_id: Optional[str] = None
        self._id2obj: Dict[int, object] = {}
        self._room_names: Dict[int, str] = {}
        self.graph = EpisodeSceneGraph()

    def _ensure_scene(self, sim, scene_id: str) -> None:
        if scene_id != self._scene_id:
            self._id2obj = build_id_to_object(sim)
            self._room_names = build_room_names(sim)
            self._scene_id = scene_id

    def reset_episode(self) -> None:
        self.graph = EpisodeSceneGraph()

    def describe(self, sim, scene_id: str, semantic_img: np.ndarray, agent_state) -> str:
        self._ensure_scene(sim, scene_id)
        self.graph.record_pose(agent_state)
        objects = objects_in_fov(semantic_img, self._id2obj, self.min_pixels, self.top_k)
        self.graph.update(objects)

        if self.text_format == "toon":
            visible_ids = {o["id"] for o in objects}
            text = self.graph.to_toon(agent_state, visible_ids=visible_ids)
        else:  # natural-language fallback
            fov_text = describe_fov_text(objects, agent_state)
            text = f"{fov_text} {self.graph.describe(agent_state)}".strip()
        self.graph.advance()
        return text

    def export(self, meta: Optional[dict] = None) -> dict:
        """The accumulated graph + trajectory as a viewer-ready dict."""
        return self.graph.to_dict(meta or {}, room_names=self._room_names)

    def save_html(self, path, meta: Optional[dict] = None) -> None:
        """Write a self-contained 3D scene-graph viewer for the current episode.

        No-op if nothing was observed (an empty graph would render a blank scene).
        """
        from pathlib import Path
        from longnav.utils.scene_graph_viewer import render_html

        if not self.graph.rooms:
            return
        Path(path).write_text(render_html(self.export(meta)), encoding="utf-8")
