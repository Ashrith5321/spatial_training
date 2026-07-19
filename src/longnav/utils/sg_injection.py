"""Load-time scene-graph injection for SFT (train == eval).

The eval loop (run_longnav_eval.py) feeds the model a per-step ground-truth scene
graph rendered by ``GTSceneGraphProvider(text_format="json")`` (see
``longnav.utils.gt_scene_graph``). To train on the *same* signal, this module
rebuilds the identical message format at data-load time and injects the per-step
SG json into each observation turn.

The scene graph itself is produced offline (Habitat replay + semantic sensor) and
stored as one file per episode; this module only reads those files and assembles
messages -- it never touches the simulator. Wire ``make_sg_message_transform`` into
the dataset with ``dataset.set_transform(...)``.

IMPORTANT: the prompt constants below are copied verbatim from run_longnav_eval.py
and MUST stay in sync with it, otherwise train != eval. ``verify_sg_in_sft.py``
asserts they still match the eval script.
"""

import json
import re
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Optional

# --------------------------------------------------------------------------
# Prompt constants -- keep in sync with run_longnav_eval.py (checked by verify).
# --------------------------------------------------------------------------
ACTION_SPACE = ["stop", "forward", "left", "right"]
ACTION_SPACE_STR = "[stop, forward, left, right]"

SYSTEM_PROMPT = (
    'You are a visual navigation agent tasked with finding "$instr_or_goal" '
    "in an unknown environment.\n"
    "You will receive a sequence of observations showing your movement history "
    "up to the current moment.\n\n"
    "**Action Space:**\n"
    "$action_space_str\n\n"
    "**Your Mission:**\n"
    "1. Analyze the observation history to understand your current location and orientation.\n"
    "2. Select the next discrete action to navigate efficiently towards the goal.\n\n"
    "**Critical Constraints:**\n"
    "* **Collision Detection:** If your previous action was **forward** but the visual "
    "observation did not change significantly, you have collided. You MUST turn or move "
    "away immediately. Do not keep pushing forward.\n"
    "* **Success Condition:** Output **stop** ONLY when the target is plainly in view, "
    "centered, and within 1 meter (close enough to touch).\n\n"
    "$scene_graph_section"
    "**Output Format:**\n"
    "Respond with the selected action inside double asterisks.\n"
)

SCENE_GRAPH_SECTION = (
    "**Scene Graph:**\n"
    "Each observation includes a scene graph of objects discovered so far. "
    "Format: the agent node shows your current room and xyz position relative to "
    "your start. Each room lists its objects with name and xyz. Edges show which "
    "rooms connect. Use this spatial memory to avoid revisiting explored rooms "
    "and to navigate toward unexplored areas.\n\n"
)

# --------------------------------------------------------------------------
# Action normalization -> eval's 4-action lowercase vocabulary.
# --------------------------------------------------------------------------
_ACTION_ALIASES = {
    "stop": "stop", "0": "stop",
    "forward": "forward", "move_forward": "forward", "1": "forward",
    "left": "left", "turn_left": "left", "2": "left",
    "right": "right", "turn_right": "right", "3": "right",
}


def normalize_action(action: Any) -> str:
    """Map a raw action (int id or string) onto eval's ACTION_SPACE names."""
    if isinstance(action, bool):  # guard: bool is an int subclass
        raise ValueError(f"Unexpected bool action: {action!r}")
    if isinstance(action, int):
        return ACTION_SPACE[action]
    key = str(action).strip().lower()
    if key in _ACTION_ALIASES:
        return _ACTION_ALIASES[key]
    if key in ACTION_SPACE:
        return key
    raise ValueError(f"Unknown action {action!r}; expected one of {ACTION_SPACE} or a known alias")


# --------------------------------------------------------------------------
# Per-episode SG file parsing.
# --------------------------------------------------------------------------
_STEP_HDR = re.compile(r"^#\s*=*\s*step\s+(\d+)\s*=*\s*$", re.IGNORECASE)
_TELEOP_LINE = re.compile(r"^\[step\s+(\d+)\]\s*model SG:\s*(.*)$", re.IGNORECASE)


def load_episode_sg(path) -> List[str]:
    """Parse a per-episode SG file into a list of per-step strings (index == step).

    Supports the formats the repo emits:
      * ``[step N] model SG: {json}``            (teleop stdout capture)
      * ``# ==== step N ====`` delimited blocks  (run_longnav_eval --dump-toon)
      * ``.jsonl``: one json object per line     (step per line, in order)
      * ``.json``: a json list, or a {step: sg} dict

    Each returned element is the raw text injected verbatim into the message, so it
    must already be in the SAME format eval feeds the model (json).
    """
    p = Path(path)
    text = p.read_text(encoding="utf-8")

    # 1. teleop stdout lines: "[step N] model SG: {...}"
    teleop = {}
    for line in text.splitlines():
        m = _TELEOP_LINE.match(line.strip())
        if m:
            teleop[int(m.group(1))] = m.group(2).strip()
    if teleop:
        return [teleop[i] for i in sorted(teleop)]

    # 2. "# ==== step N ====" delimited blocks
    if any(_STEP_HDR.match(l.strip()) for l in text.splitlines()):
        blocks: Dict[int, List[str]] = {}
        cur: Optional[int] = None
        buf: List[str] = []
        for line in text.splitlines():
            m = _STEP_HDR.match(line.strip())
            if m:
                if cur is not None:
                    blocks[cur] = buf
                cur, buf = int(m.group(1)), []
            elif cur is not None:
                buf.append(line)
        if cur is not None:
            blocks[cur] = buf
        return ["\n".join(b).strip() for _, b in sorted(blocks.items())]

    # 3. .json list / dict
    if p.suffix == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict):
            items = sorted(((int(k), v) for k, v in obj.items()), key=lambda t: t[0])
            return [v if isinstance(v, str) else json.dumps(v) for _, v in items]
        if isinstance(obj, list):
            return [v if isinstance(v, str) else json.dumps(v) for v in obj]

    # 4. jsonl: one object per non-empty line, kept in order
    out: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if line:
            out.append(line)
    return out


class SGStore:
    """Loads every per-episode SG file under ``sg_dir`` and indexes it by a join key.

    The default join key is parsed from the eval/teleop filename convention
    ``{index:04d}_{scene}_{episode_id}_{goal}.<ext>`` -> ``"{scene}/{episode_id}"``.
    Pass a custom ``key_from_filename`` if your files are named differently.
    """

    _FNAME = re.compile(r"^\d+_(?P<scene>.+?)_(?P<ep>\d+)_.*$")

    def __init__(
        self,
        sg_dir,
        patterns=("*.toon.txt", "*.sg.jsonl", "*.sg.json", "*.json"),
        key_from_filename: Optional[Callable[[str], Optional[str]]] = None,
    ):
        self.sg_dir = Path(sg_dir)
        self._index: Dict[str, List[str]] = {}
        key_fn = key_from_filename or self._default_key
        seen = set()
        for pat in patterns:
            for f in sorted(self.sg_dir.glob(pat)):
                if "_scenegraph" in f.name:  # the .html viewer file, not SG text
                    continue
                if f.name in seen:
                    continue
                seen.add(f.name)
                key = key_fn(f.name)
                if key is None:
                    continue
                try:
                    self._index[key] = load_episode_sg(f)
                except Exception as e:  # noqa: BLE001 - skip a corrupt file, keep going
                    print(f"[SGStore] skipping {f.name}: {e}")
        if not self._index:
            raise FileNotFoundError(f"No SG files parsed under {self.sg_dir} (patterns={patterns})")

    @classmethod
    def _default_key(cls, fname: str) -> Optional[str]:
        stem = fname
        for ext in (".toon.txt", ".sg.jsonl", ".sg.json", ".json", ".txt"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        m = cls._FNAME.match(stem)
        if not m:
            return None
        return f"{m.group('scene')}/{m.group('ep')}"

    def get(self, key: str) -> Optional[List[str]]:
        return self._index.get(key)

    def __len__(self) -> int:
        return len(self._index)


# --------------------------------------------------------------------------
# Message construction (identical turn structure to run_longnav_eval.py).
# --------------------------------------------------------------------------
def build_sg_messages(
    goal: str,
    actions: List[Any],
    sg_per_step: Optional[List[str]],
    num_images: int,
    use_scene_graph: bool = True,
    align: str = "strict",
) -> List[Dict[str, Any]]:
    """Build the full multi-turn conversation the collator will tokenize.

    Layout (matches eval's SYSTEM_PROMPT + CONVO_START/TURN, fully unrolled):
        user   : system prompt (with SCENE_GRAPH_SECTION iff use_scene_graph)
        user   : [image_t, sg_t]      # sg_t omitted if use_scene_graph is False
        assistant: **action_t**
        ... one (user, assistant) pair per step ...

    Alignment: image_t <-> action_t <-> sg_t, all length == num_images.
    ``align="truncate"`` trims to the shortest of the three (with a warning) to
    tolerate off-by-one temporal-shift conventions; ``"strict"`` raises instead.
    """
    n = num_images
    if use_scene_graph and sg_per_step is None:
        raise ValueError("use_scene_graph=True but sg_per_step is None")

    lens = [n, len(actions)] + ([len(sg_per_step)] if use_scene_graph else [])
    if len(set(lens)) != 1:
        if align == "truncate":
            n = min(lens)
            actions = actions[:n]
            if use_scene_graph:
                sg_per_step = sg_per_step[:n]
            print(f"[build_sg_messages] length mismatch {lens} -> truncated to {n}")
        else:
            raise ValueError(
                f"Alignment error: images={n}, actions={len(actions)}"
                + (f", sg={len(sg_per_step)}" if use_scene_graph else "")
            )

    system_text = Template(SYSTEM_PROMPT).substitute(
        instr_or_goal=goal,
        action_space_str=ACTION_SPACE_STR,
        scene_graph_section=SCENE_GRAPH_SECTION if use_scene_graph else "",
    )

    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": [{"type": "text", "text": system_text}]}
    ]
    for t in range(n):
        user_content: List[Dict[str, Any]] = [{"type": "image"}]
        if use_scene_graph:
            user_content.append({"type": "text", "text": sg_per_step[t]})
        messages.append({"role": "user", "content": user_content})
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": f"**{normalize_action(actions[t])}**"}]}
        )
    return messages


def default_join_key(batch: Dict[str, List[Any]], i: int) -> str:
    """episode join key from a dataset row: '{scene}/{episode_id}'.

    Reads ``scene_id``/``scene`` and ``episode_id``/``episode``; strips any
    ``.glb``/``.basis.glb`` suffix and directory from the scene so it matches the
    SGStore filename key.
    """
    def _first(*keys):
        for k in keys:
            if k in batch:
                return batch[k][i]
        return None

    scene = _first("scene", "scene_id")
    ep = _first("episode_id", "episode", "episode_idx", "id")
    if scene is None or ep is None:
        raise KeyError(
            "Cannot build join key: need scene_id/scene and episode_id in the dataset. "
            f"Available columns: {list(batch.keys())}. Pass a custom id_fn."
        )
    scene = Path(str(scene)).name.replace(".basis.glb", "").replace(".glb", "")
    return f"{scene}/{ep}"


def make_sg_message_transform(
    sg_store: SGStore,
    *,
    goal_key: str = "goal_text",
    action_key: str = "action_sequence",
    id_fn: Callable[[Dict[str, List[Any]], int], str] = default_join_key,
    resize_transform: Optional[Callable] = None,
    align: str = "strict",
    require_match: bool = True,
) -> Callable[[Dict[str, List[Any]]], Dict[str, List[Any]]]:
    """A ``dataset.set_transform`` callable that (re)builds ``messages`` with SG.

    For each row it looks up the episode's per-step SG in ``sg_store``, builds the
    eval-format conversation, and writes ``batch['messages']``. If a resize
    transform is given (e.g. the repo's ``dynamic_resize_transform``) it is applied
    afterwards so images are still budget-fit.

    ``require_match=True`` raises when an episode has no SG file (so you never
    silently train SG-free rows); set False to fall back to an SG-free prompt.
    """
    def _transform(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        n_ex = len(batch[action_key])
        all_messages: List[List[Dict[str, Any]]] = []
        for i in range(n_ex):
            actions = batch[action_key][i]
            goal = batch[goal_key][i] if goal_key in batch else "the target"
            key = id_fn(batch, i)
            sg_list = sg_store.get(key)
            if sg_list is None:
                if require_match:
                    raise KeyError(f"No scene graph for episode key {key!r} (SGStore has {len(sg_store)} eps)")
                use_sg = False
            else:
                use_sg = True
            all_messages.append(
                build_sg_messages(
                    goal=goal,
                    actions=actions,
                    sg_per_step=sg_list,
                    num_images=len(batch["images"][i]),
                    use_scene_graph=use_sg,
                    align=align,
                )
            )
        batch["messages"] = all_messages
        if resize_transform is not None:
            batch = resize_transform(batch)
        return batch

    return _transform
