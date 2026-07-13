"""
HM3D v2 full 1000-episode oracle evaluation.
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

import cv2
import numpy as np

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
import longnav.utils.measures  # noqa: F401 -- registers custom measures
import longnav.utils.ovon.ovon_dataset  # noqa: F401 -- registers OVON dataset
import longnav.utils.ovon.ovon_nav  # noqa: F401 -- registers OVON measures

from longnav.env.habitat import ObjectNavOracle

DATASET_PATH = "data/datasets/objectnav/hm3d/v2/val/val.json.gz"
RGB_WIDTH, FRAME_HEIGHT, MAP_WIDTH = 640, 480, 480


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run oracle evaluation on HM3D v2 val (1000 episodes)"
    )
    parser.add_argument(
        "--output-dir", default="dump/hm3d_v2_eval_1000",
        help="Directory for videos, logs, and results",
    )
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-video", action="store_true",
        help="Skip video generation (metrics only)",
    )
    return parser.parse_args()


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


def render_map(info):
    """Render top-down map with agent trail and shortest path."""
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


def make_frame(obs, info, goal, step, total_steps):
    """Create a side-by-side frame: RGB view | top-down map."""
    rgb = cv2.resize(obs["rgb"], (RGB_WIDTH, FRAME_HEIGHT))
    overlay = rgb.copy()

    distance = info.get("distance_to_goal")
    distance_str = f"{float(distance):.2f}m" if distance is not None else "N/A"

    texts = [
        f"Goal: {goal}",
        f"Distance: {distance_str}",
        f"Step: {step}/{total_steps}",
    ]
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


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def run_episode(env, oracle, index, output_dir, max_steps, fps, skip_video=False):
    obs = env.reset()
    episode = env.current_episode
    scene = Path(episode.scene_id).name.replace(".basis.glb", "")
    episode_id = str(episode.episode_id)
    goal = str(getattr(episode, "object_category", "unknown"))
    geodesic = episode.info.get("geodesic_distance", -1)

    clean = lambda v: "".join(c if c.isalnum() or c in "-_" else "_" for c in v)
    stem = f"{index:04d}_{clean(scene)}_{clean(episode_id)}_{clean(goal)}"

    writer = None
    video_path = output_dir / f"{stem}.mp4"
    temp_path = output_dir / f"{stem}.tmp.mp4"

    if not skip_video:
        writer = cv2.VideoWriter(
            str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (RGB_WIDTH + MAP_WIDTH, FRAME_HEIGHT),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {temp_path}")

    started = time.time()
    steps = 0
    stopped = False

    try:
        info = env.get_metrics()
        if writer:
            frame = make_frame(obs, info, goal, steps, max_steps)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        while steps < max_steps and not env.episode_over:
            action = int(oracle.get_best_action())
            stopped = action == 0
            obs = env.step(action)
            steps += 1
            info = env.get_metrics()

            if writer:
                frame = make_frame(obs, info, goal, steps, max_steps)
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            if stopped:
                break
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

    return record


def print_aggregate_stats(results_path):
    """Print aggregate evaluation metrics from the results JSONL."""
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
    print("AGGREGATE EVALUATION RESULTS")
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


def main():
    args = parse_args()
    output_dir = (ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    log_path = output_dir / "eval.log"

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
    oracle = ObjectNavOracle(
        env, success_distance=config.habitat.task.measurements.success.success_distance,
    )

    completed = set()
    if results_path.exists() and not args.overwrite:
        for line in results_path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                if record.get("status") == "complete":
                    completed.add(int(record["episode_index"]))
            except (ValueError, KeyError, TypeError):
                pass

    banner = (
        f"HM3D v2 oracle eval | episodes={total} | range=[{args.start}, {end}) | "
        f"already_complete={len(completed)} | max_steps={args.max_steps} | "
        f"video={'off' if args.no_video else 'on'}"
    )
    print(banner, flush=True)

    with log_path.open("a", encoding="utf-8") as logfile:
        logfile.write(f"\n{'='*60}\nRun started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n{banner}\n{'='*60}\n")
        logfile.flush()

        try:
            for index in range(end):
                if index < args.start or (index in completed and not args.overwrite):
                    env.reset()
                    continue

                try:
                    record = run_episode(
                        env, oracle, index, output_dir,
                        args.max_steps, args.fps, args.no_video,
                    )
                    append_jsonl(results_path, record)

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

    print_aggregate_stats(results_path)


if __name__ == "__main__":
    main()
