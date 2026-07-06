"""
LongNav R1 VLM evaluation on HM3D v2 (1000 episodes).
Standalone single-process script — no Ray needed.

Produces per-episode videos with side-by-side RGB + top-down map showing:
  - Shortest geodesic path (drawn at episode start)
  - Agent's actual route (drawn progressively as the agent moves)
  - Agent marker with heading indicator
Logs per-episode metrics to JSONL and prints aggregate stats at the end.
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from string import Template

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

import habitat
from habitat import make_dataset
from habitat.config import read_write
from habitat.config.default import get_config
from habitat.config.default_structured_configs import (
    FogOfWarConfig,
    HabitatSimSemanticSensorConfig,
    TopDownMapMeasurementConfig,
)
from habitat.core.dataset import EpisodeIterator
from habitat.utils.visualizations import maps as habitat_maps
import longnav.utils.measures  # noqa: F401
import longnav.utils.ovon.ovon_dataset  # noqa: F401
import longnav.utils.ovon.ovon_nav  # noqa: F401
from longnav.utils.gt_scene_graph import GTSceneGraphProvider

DATASET_PATH = "data/datasets/objectnav/hm3d/v2/val/val.json.gz"
RGB_WIDTH, FRAME_HEIGHT, MAP_WIDTH = 640, 480, 480

BASE_MODEL = "Phyllis1/qwen3_sft_sft_sparse_03drop_single_action_20260103_210803_ckpt10800"
LORA_CHECKPOINT = "Aasdfip/hm3d_rpp_ke_standard-checkpoint_231"

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

CONVO_START_TEMPLATE = [
    {"role": "user", "content": [{"type": "text", "text": "$system_prompt"}]},
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "$scene_graph_text"},
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
]

CONVO_TURN_TEMPLATE = [
    {"role": "assistant", "content": [{"type": "text", "text": "**$action**"}]},
    {"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "$scene_graph_text"},
    ]},
    {"role": "assistant", "content": [{"type": "text", "text": "**forward**"}]},
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LongNav R1 VLM evaluation on HM3D v2 val (1000 episodes)"
    )
    parser.add_argument("--output-dir", default="dump/longnav_r1_eval_1000")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--lora-checkpoint", default=LORA_CHECKPOINT)
    parser.add_argument("--no-lora", action="store_true", help="Run base model without LoRA")
    parser.add_argument("--no-scene-graph", action="store_true",
                        help="Disable scene-graph injection (drops both the prompt section and per-step graph text)")
    parser.add_argument("--dump-toon", action="store_true",
                        help="Write per-step scene-graph TOON to {stem}.toon.txt (off by default)")
    parser.add_argument("--wandb", action="store_true", help="Log live progress to Weights & Biases")
    parser.add_argument("--wandb-project", default="longnav_r1_eval")
    parser.add_argument("--wandb-name", default=None, help="wandb run name (default: output dir name)")
    return parser.parse_args()


# ---------- Conversation template substitution ----------

def substitute_convo_template(template, substitutions):
    new_conversation = []
    for message in template:
        new_message = message.copy()
        new_content = []
        for item in message.get("content", []):
            new_item = item.copy()
            if "text" in new_item and "$" in new_item["text"]:
                new_item["text"] = Template(new_item["text"]).substitute(substitutions)
                if new_item["text"] == "":
                    continue  # scene graph disabled: drop the empty text block entirely
            new_content.append(new_item)
        new_message["content"] = new_content
        new_conversation.append(new_message)
    return new_conversation


# ---------- Habitat config ----------

def build_config():
    config = get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")
    with read_write(config):
        config.habitat.dataset.data_path = DATASET_PATH
        config.habitat.dataset.split = "val"

        agent = config.habitat.simulator.agents.main_agent
        agent.sim_sensors.rgb_sensor.width = 640
        agent.sim_sensors.rgb_sensor.height = 480
        agent.sim_sensors.rgb_sensor.hfov = 79
        agent.sim_sensors.rgb_sensor.position = [0, 0.88, 0]
        agent.sim_sensors.depth_sensor.width = 640
        agent.sim_sensors.depth_sensor.height = 480
        agent.sim_sensors.depth_sensor.hfov = 79
        agent.sim_sensors.depth_sensor.min_depth = 0.0
        agent.sim_sensors.depth_sensor.max_depth = 5.0
        agent.sim_sensors.depth_sensor.position = [0, 0.88, 0]

        agent.sim_sensors.semantic_sensor = HabitatSimSemanticSensorConfig()
        agent.sim_sensors.semantic_sensor.width = 640
        agent.sim_sensors.semantic_sensor.height = 480
        agent.sim_sensors.semantic_sensor.hfov = 79
        agent.sim_sensors.semantic_sensor.position = [0, 0.88, 0]

        agent.height = 0.88
        agent.radius = 0.18

        config.habitat.simulator.turn_angle = 30
        config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
        config.habitat.simulator.habitat_sim_v0.allow_sliding = True
        config.habitat.environment.max_episode_steps = 500
        config.habitat.environment.iterator_options.shuffle = False

        config.habitat.task.measurements.success.success_distance = 1.0
        config.habitat.task.measurements.top_down_map = TopDownMapMeasurementConfig(
            map_padding=3,
            map_resolution=1024,
            draw_goal_positions=True,
            draw_shortest_path=True,
            draw_view_points=True,
            draw_border=True,
            fog_of_war=FogOfWarConfig(draw=True, visibility_dist=20, fov=79),
        )
    return config


# ---------- VLM model loading ----------

def load_vlm(model_id, lora_checkpoint=None, use_sparse=True):
    """Load the VLM worker with optional LoRA checkpoint."""
    from longnav.utils.vlm_worker import VLMWorker

    worker = VLMWorker(
        model_id=model_id,
        attn_impl="sdpa",
        dtype="bfloat16",
        prefix="<|im_start|>assistant\n**",
        postfix="**<|im_end|>\n",
        vocab=ACTION_SPACE,
        save_outputs=False,
        load_model=True,
        offload_cache=False,
        use_sparse=use_sparse,
    )

    if lora_checkpoint:
        from huggingface_hub import snapshot_download
        from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
        from peft.utils import load_peft_weights

        ckpt_path = snapshot_download(
            lora_checkpoint,
            allow_patterns=["*.json", "*.bin", "*.safetensors", "*.pt", "*.yaml"],
        )
        print(f"LoRA checkpoint: {ckpt_path}")

        adapter_config_path = os.path.join(ckpt_path, "adapter_config.json")
        with open(adapter_config_path) as f:
            adapter_cfg = json.load(f)

        lora_config = LoraConfig(
            r=adapter_cfg.get("r", 128),
            lora_alpha=adapter_cfg.get("lora_alpha", 256),
            target_modules=adapter_cfg.get("target_modules", []),
            lora_dropout=adapter_cfg.get("lora_dropout", 0.0),
            bias=adapter_cfg.get("bias", "none"),
            task_type=adapter_cfg.get("task_type", "CAUSAL_LM"),
            modules_to_save=adapter_cfg.get("modules_to_save", None),
        )
        worker.model = get_peft_model(worker.model, lora_config)
        weights = load_peft_weights(ckpt_path)
        set_peft_model_state_dict(worker.model, weights)
        worker._is_lora = True
        worker._is_merged = False
        worker.model.merge_adapter()
        worker._is_merged = True
        print("LoRA adapter loaded and merged.")

    return worker


# ---------- Visualization ----------

def render_map(info):
    top_down = info.get("top_down_map")
    if top_down is None:
        return np.zeros((FRAME_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)
    image = habitat_maps.colorize_topdown_map(
        top_down["map"], top_down["fog_of_war_mask"]
    )
    agent_coord = top_down["agent_map_coord"]
    agent_angle = top_down["agent_angle"]
    if isinstance(agent_coord, list):
        agent_coord = agent_coord[0]
    if isinstance(agent_angle, list):
        agent_angle = agent_angle[0]
    agent_angle = float(agent_angle)
    image = habitat_maps.draw_agent(
        image=image,
        agent_center_coord=agent_coord,
        agent_rotation=agent_angle,
        agent_radius_px=max(1, min(image.shape[:2]) // 32),
    )
    return cv2.resize(image, (MAP_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


def make_frame(obs, info, goal, step, max_steps, action_name=None):
    rgb = cv2.resize(obs["rgb"], (RGB_WIDTH, FRAME_HEIGHT))
    overlay = rgb.copy()

    distance = info.get("distance_to_goal")
    distance_str = f"{float(distance):.2f}m" if distance is not None else "N/A"

    texts = [
        f"Goal: {goal}",
        f"Distance: {distance_str}",
        f"Step: {step}/{max_steps}",
    ]
    if action_name:
        texts.append(f"Action: {action_name}")

    y = 30
    for text in texts:
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, (5, y - th - 5), (10 + tw, y + 5), (0, 0, 0), -1)
        cv2.putText(
            overlay, text, (8, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA,
        )
        y += th + 15

    tdm = render_map(info)
    return np.concatenate([overlay, tdm], axis=1)


# ---------- Helpers ----------

def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def get_goal_name(env, obs):
    episode = env.current_episode
    goal = str(getattr(episode, "object_category", "unknown"))
    return goal


# ---------- Episode runner ----------

def run_episode(env, vlm, index, output_dir, max_steps, fps, temperature,
                skip_video=False, gt_scene_graph=None, dump_toon=False):
    obs = env.reset()
    episode = env.current_episode
    scene = Path(episode.scene_id).name.replace(".basis.glb", "")
    episode_id = str(episode.episode_id)
    goal = get_goal_name(env, obs)
    geodesic = episode.info.get("geodesic_distance", -1)

    if gt_scene_graph is not None:
        gt_scene_graph.reset_episode()

    sg_log = []  # per-step scene-graph TOON blocks fed to the VLM (for inspection)

    def scene_graph_text():
        if gt_scene_graph is None:
            return ""
        agent_state = env.sim.get_agent(0).get_state()
        txt = gt_scene_graph.describe(
            env.sim, episode.scene_id, obs["semantic"], agent_state
        )
        sg_log.append(txt)
        return txt

    clean = lambda v: "".join(c if c.isalnum() or c in "-_" else "_" for c in v)
    stem = f"{index:04d}_{clean(scene)}_{clean(episode_id)}_{clean(goal)}"
    video_path = output_dir / f"{stem}.mp4"
    temp_path = output_dir / f"{stem}.tmp.mp4"

    writer = None
    if not skip_video:
        writer = cv2.VideoWriter(
            str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (RGB_WIDTH + MAP_WIDTH, FRAME_HEIGHT),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {temp_path}")

    # Reset VLM state (KV cache, position tracking)
    vlm.reset()

    # Build initial conversation
    system_prompt = Template(SYSTEM_PROMPT).substitute(
        instr_or_goal=goal, action_space_str=ACTION_SPACE_STR,
        scene_graph_section=SCENE_GRAPH_SECTION if gt_scene_graph is not None else "",
    )
    messages = substitute_convo_template(
        CONVO_START_TEMPLATE,
        {"system_prompt": system_prompt, "scene_graph_text": scene_graph_text()},
    )

    started = time.time()
    steps = 0
    stopped = False
    action_name = None

    try:
        info = env.get_metrics()
        if writer:
            frame = make_frame(obs, info, goal, steps, max_steps, action_name)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        while steps < max_steps and not env.episode_over:
            # VLM inference
            rgb_pil = Image.fromarray(obs["rgb"])
            action_probs, action_logprobs, outputs = vlm.infer_probs(
                images=[rgb_pil], messages=messages, temperature=temperature,
            )

            # Sample action
            action_id = np.random.choice(len(action_probs), p=action_probs)
            action_name = ACTION_SPACE[action_id]
            stopped = action_id == 0

            # Step environment
            obs = env.step(action_id)
            steps += 1
            info = env.get_metrics()

            if writer:
                frame = make_frame(obs, info, goal, steps, max_steps, action_name)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if stopped:
                break

            # Build next turn messages
            messages = substitute_convo_template(
                CONVO_TURN_TEMPLATE,
                {"action": action_name, "scene_graph_text": scene_graph_text()},
            )
    finally:
        if writer:
            writer.release()

    if not skip_video:
        if not temp_path.exists() or temp_path.stat().st_size == 0:
            raise RuntimeError(f"Empty video: {temp_path}")
        temp_path.replace(video_path)

    metrics = env.get_metrics()
    elapsed = round(time.time() - started, 3)

    record = {
        "status": "complete",
        "episode_index": index,
        "scene": scene,
        "episode_id": episode_id,
        "goal": goal,
        "geodesic_distance": geodesic,
        "steps": steps,
        "stopped": stopped,
        "success": metrics.get("success"),
        "spl": metrics.get("spl"),
        "soft_spl": metrics.get("soft_spl"),
        "distance_to_goal": metrics.get("distance_to_goal"),
        "elapsed_seconds": elapsed,
    }
    if not skip_video:
        record["video"] = str(video_path)
        record["video_bytes"] = video_path.stat().st_size

    if gt_scene_graph is not None:
        sg_path = output_dir / f"{stem}_scenegraph.html"
        gt_scene_graph.save_html(sg_path, meta={
            "scene": scene, "episode_id": episode_id, "goal": goal,
            "steps": steps, "success": metrics.get("success"),
        })
        if sg_path.exists():
            record["scene_graph_html"] = str(sg_path)
        if dump_toon and sg_log:  # per-step TOON blocks (last one = end-of-episode graph)
            toon_path = output_dir / f"{stem}.toon.txt"
            blocks = [f"# ==== step {i} ====\n{t}" for i, t in enumerate(sg_log)]
            toon_path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")
            record["scene_graph_toon"] = str(toon_path)

    return record


# ---------- Aggregate stats ----------

def print_aggregate_stats(results_path):
    records = []
    for line in results_path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
            if r.get("status") == "complete":
                records.append(r)
        except (ValueError, KeyError):
            pass

    if not records:
        print("No completed episodes found.")
        return

    n = len(records)
    successes = sum(1 for r in records if r.get("success"))
    avg_spl = np.mean([r["spl"] for r in records if r.get("spl") is not None])
    avg_soft_spl = np.mean([r["soft_spl"] for r in records if r.get("soft_spl") is not None])
    avg_dtg = np.mean([r["distance_to_goal"] for r in records if r.get("distance_to_goal") is not None])
    avg_steps = np.mean([r["steps"] for r in records])
    avg_time = np.mean([r["elapsed_seconds"] for r in records])
    total_time = sum(r["elapsed_seconds"] for r in records)

    print("\n" + "=" * 60)
    print("AGGREGATE EVALUATION RESULTS (LongNav R1)")
    print("=" * 60)
    print(f"  Episodes completed:  {n}")
    print(f"  Success rate:        {successes}/{n} = {successes/n:.4f}")
    print(f"  SPL:                 {avg_spl:.4f}")
    print(f"  Soft SPL:            {avg_soft_spl:.4f}")
    print(f"  Avg distance to goal:{avg_dtg:.4f}m")
    print(f"  Avg steps:           {avg_steps:.1f}")
    print(f"  Avg time/episode:    {avg_time:.2f}s")
    print(f"  Total wall time:     {total_time:.1f}s ({total_time/3600:.2f}h)")
    print("=" * 60)


# ---------- Main ----------

def main():
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    log_path = output_dir / "eval.log"

    # Load VLM
    print(f"Loading VLM: {args.base_model}")
    lora = None if args.no_lora else args.lora_checkpoint
    vlm = load_vlm(args.base_model, lora_checkpoint=lora)
    print("VLM loaded successfully.")

    # Build Habitat config and env
    config = build_config()
    dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
    total = dataset.num_episodes
    end = min(args.end if args.end is not None else total, total)

    if args.start < 0 or args.start >= end:
        raise ValueError(f"Invalid range [{args.start}, {end}) for {total} episodes")

    env = habitat.Env(config=config, dataset=dataset)
    env.episode_iterator = EpisodeIterator(
        dataset.episodes, cycle=False, shuffle=False,
        group_by_scene=False, seed=17,
    )
    gt_scene_graph = None if args.no_scene_graph else GTSceneGraphProvider()

    # Check for already completed episodes (resumability). Also seed running
    # aggregates so the wandb success-rate/SPL curves stay accurate across resumes.
    completed = set()
    agg = {"n": 0, "succ": 0, "spl": 0.0, "soft": 0.0, "steps": 0, "dtg": 0.0}

    def _accumulate(rec):
        agg["n"] += 1
        agg["succ"] += 1 if rec.get("success") else 0
        agg["spl"] += rec.get("spl") or 0.0
        agg["soft"] += rec.get("soft_spl") or 0.0
        agg["steps"] += rec.get("steps") or 0
        agg["dtg"] += rec.get("distance_to_goal") or 0.0

    if results_path.exists() and not args.overwrite:
        for line in results_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                if record.get("status") == "complete":
                    completed.add(int(record["episode_index"]))
                    _accumulate(record)
            except (ValueError, KeyError, TypeError):
                pass

    banner = (
        f"LongNav R1 eval | model={args.base_model.split('/')[-1]} | "
        f"lora={'none' if args.no_lora else args.lora_checkpoint.split('/')[-1]} | "
        f"scene_graph={'off' if args.no_scene_graph else 'on'} | "
        f"episodes={total} | range=[{args.start}, {end}) | "
        f"already_complete={len(completed)} | max_steps={args.max_steps} | "
        f"temp={args.temperature} | video={'off' if args.no_video else 'on'}"
    )
    print(banner, flush=True)

    # Optional Weights & Biases live monitoring.
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=args.wandb_name or Path(args.output_dir).name,
                resume="allow",
                config={
                    "model": args.base_model,
                    "lora": None if args.no_lora else args.lora_checkpoint,
                    "range": [args.start, end], "total_episodes": total,
                    "max_steps": args.max_steps, "temperature": args.temperature,
                },
            )
            print(f"[wandb] logging to {wandb_run.url}", flush=True)
        except Exception as e:
            print(f"[wandb] disabled ({e}); continuing without it.", flush=True)
            wandb_run = None

    def log_wandb(rec, index):
        if wandb_run is None or rec.get("status") != "complete":
            return
        _accumulate(rec)
        n = agg["n"]
        wandb_run.log({
            "episode/success": 1 if rec.get("success") else 0,
            "episode/spl": rec.get("spl"),
            "episode/soft_spl": rec.get("soft_spl"),
            "episode/steps": rec.get("steps"),
            "episode/distance_to_goal": rec.get("distance_to_goal"),
            "episode/time_s": rec.get("elapsed_seconds"),
            "running/success_rate": agg["succ"] / n,
            "running/spl": agg["spl"] / n,
            "running/soft_spl": agg["soft"] / n,
            "running/avg_steps": agg["steps"] / n,
            "running/avg_distance_to_goal": agg["dtg"] / n,
            "running/completed": n,
            "progress/pct": 100.0 * n / max(1, end - args.start),
        }, step=index)

    with log_path.open("a", encoding="utf-8") as logfile:
        logfile.write(
            f"\n{'='*60}\nRun started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"{banner}\n{'='*60}\n"
        )
        logfile.flush()

        try:
            for index in range(end):
                if index < args.start or (index in completed and not args.overwrite):
                    env.reset()
                    continue

                try:
                    record = run_episode(
                        env, vlm, index, output_dir,
                        args.max_steps, args.fps, args.temperature,
                        args.no_video, gt_scene_graph=gt_scene_graph,
                        dump_toon=args.dump_toon,
                    )
                    append_jsonl(results_path, record)
                    log_wandb(record, index)

                    msg = (
                        f"[{index + 1}/{end}] {record['scene']} "
                        f"goal={record['goal']} steps={record['steps']} "
                        f"success={record['success']} "
                        f"spl={record.get('spl', 0):.3f} "
                        f"dist={record['distance_to_goal']:.3f} "
                        f"time={record['elapsed_seconds']:.1f}s"
                    )
                    print(msg, flush=True)
                    logfile.write(msg + "\n")
                    logfile.flush()

                except Exception as error:
                    err_record = {
                        "status": "failed",
                        "episode_index": index,
                        "error": repr(error),
                        "traceback": traceback.format_exc(),
                    }
                    append_jsonl(results_path, err_record)
                    msg = f"[{index + 1}/{end}] FAILED: {error!r}"
                    print(msg, flush=True)
                    logfile.write(msg + "\n" + traceback.format_exc() + "\n")
                    logfile.flush()

        finally:
            env.close()

    if wandb_run is not None and agg["n"] > 0:
        n = agg["n"]
        wandb_run.summary.update({
            "final/completed": n,
            "final/success_rate": agg["succ"] / n,
            "final/spl": agg["spl"] / n,
            "final/soft_spl": agg["soft"] / n,
            "final/avg_steps": agg["steps"] / n,
            "final/avg_distance_to_goal": agg["dtg"] / n,
        })
        wandb_run.finish()

    print_aggregate_stats(results_path)


if __name__ == "__main__":
    main()
