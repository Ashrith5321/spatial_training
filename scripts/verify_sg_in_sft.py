#!/usr/bin/env python
"""Verify the scene graph actually reaches the tokens train_sft_sparse trains on.

Runs the same load-time path the trainer uses (SG transform -> ActionMaskingVLM
collator) and checks, on one real example:

  1. SG is present in the built `messages`.
  2. SG survives chat-template + tokenization into `input_ids` (what the model sees).
  3. SG is MASKED in `labels` (context, not a training target).

Also asserts the prompt constants in longnav.utils.sg_injection still match
run_longnav_eval.py, so train == eval.

Usage:
  python scripts/verify_sg_in_sft.py \
      --train_dataset_dir <disk_dataset> \
      --scene_graph_dir   <dir_of_per_episode_sg_files> \
      --model_id          Aasdfip/qwen3_webnav_0.1
"""
import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from longnav.utils import sg_injection as sgi

SG_MARKERS = ('"agent"', "room_", '"edges"', '"rooms"', '"xyz"')


def _has_sg(text: str) -> bool:
    return any(m in text for m in SG_MARKERS)


def check_constants_match_eval() -> bool:
    """Fail loudly if the prompt strings drifted from run_longnav_eval.py."""
    eval_py = _ROOT / "run_longnav_eval.py"
    if not eval_py.exists():
        print(f"[constants] SKIP: {eval_py} not present in this checkout")
        return True
    src = eval_py.read_text(encoding="utf-8")
    # Distinctive phrases that each live inside a SINGLE source string literal, so
    # they survive run_longnav_eval.py's multi-literal string concatenation.
    checks = {
        "SCENE_GRAPH_SECTION": ["scene graph of objects discovered so far"],
        "ACTION_SPACE_STR": ["[stop, forward, left, right]"],
        "SYSTEM_PROMPT": ["navigation agent tasked with finding",
                          "selected action inside double asterisks"],
    }
    ok = True
    for name, needles in checks.items():
        for needle in needles:
            if needle not in src:
                print(f"[constants] MISMATCH: {name} phrase not found in eval script:\n   {needle!r}")
                ok = False
    print("[constants] OK" if ok else "[constants] DRIFT DETECTED -> train != eval")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_dataset_dir", required=True)
    ap.add_argument("--scene_graph_dir", required=True)
    ap.add_argument("--model_id", required=True)
    ap.add_argument("--goal_key", default="goal_text")
    ap.add_argument("--action_key", default="action_sequence")
    ap.add_argument("--index", type=int, default=0)
    args = ap.parse_args()

    from datasets import load_from_disk, Sequence, Image
    from transformers import AutoProcessor
    from longnav.utils.collators import ActionMaskingVLMCollator

    consts_ok = check_constants_match_eval()

    print(f"\n[load] dataset: {args.train_dataset_dir}")
    ds = load_from_disk(args.train_dataset_dir)
    print(f"[load] columns: {ds.column_names}")
    ds = ds.cast_column("images", Sequence(Image(decode=True)))

    store = sgi.SGStore(args.scene_graph_dir)
    print(f"[load] SGStore episodes: {len(store)}")

    transform = sgi.make_sg_message_transform(
        store, goal_key=args.goal_key, action_key=args.action_key, require_match=True
    )
    ds.set_transform(transform)

    ex = ds[args.index]

    # --- Check 1: SG in messages ---
    msg_blob = json.dumps(ex["messages"])
    c1 = _has_sg(msg_blob)
    print(f"\n[check 1] SG present in messages: {c1}")
    if not c1:
        print("  -> messages sample:", msg_blob[:600])

    # --- Check 2: SG survives into input_ids ---
    proc = AutoProcessor.from_pretrained(args.model_id)
    coll = ActionMaskingVLMCollator(processor=proc, dropout=-1, length_warning=10**9)
    batch = coll.torch_call([ex])
    decoded = proc.tokenizer.decode(batch["input_ids"][0])
    c2 = _has_sg(decoded)
    print(f"[check 2] SG present in tokenized input_ids: {c2}")
    print(f"          seq_len = {batch['input_ids'].shape[1]}")

    # --- Check 3: SG is masked out of the labels ---
    ids, labels = batch["input_ids"][0], batch["labels"][0]
    trained_on = proc.tokenizer.decode(ids[labels != -100])
    c3 = not _has_sg(trained_on)
    print(f"[check 3] SG correctly MASKED (absent from labels): {c3}")
    print(f"          trained-on targets: {trained_on[:200]!r}")

    ok = consts_ok and c1 and c2 and c3
    print("\n==== RESULT:", "PASS ✅" if ok else "FAIL ❌", "====")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
