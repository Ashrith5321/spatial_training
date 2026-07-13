"""
HM3D v2 full evaluation (1000 episodes) with oracle navigation.
Videos: RGB (left) | top-down map (right) showing both the shortest path
and the agent's route drawn progressively as the agent moves.

Usage:
    conda activate habitat033
    python run_eval_v2.py [--output-dir dump/hm3d_v2_eval] [--max-steps 500] [--fps 10]
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
    TopDownMapMeasurementConfig,
)
from habitat.core.dataset import EpisodeIterator
from habitat.utils.visualizations import maps as habitat_maps

import longnav.utils.measures          # noqa: F401 – registers custom measures
import longnav.utils.ovon.ovon_dataset  # noqa: F401
import longnav.utils.ovon.ovon_nav      # noqa: F401
from longnav.env.habitat import ObjectNavOracle

DATASET_PATH = "data/datasets/objectnav/hm3d/v2/val/val.json.gz"
RGB_WIDTH, FRAME_HEIGHT, MAP_WIDTH = 640, 480, 480


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default="dump/hm3d_v2_eval")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--end", type=int)
    p.add_argument("--max-steps", type=int, default=500)
    p.add_argument("--fps", type=float, default=10.0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def build_config():
    config = get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")
    with read_write(config):
        config.habitat.dataset.data_path = DATASET_PATH
        config.habitat.dataset.split = "val"

        agent = config.habitat.simulator.agents.main_agent
        for sensor_name in ("rgb_sensor", "depth_sensor"):
            sensor = agent.sim_sensors[sensor_name]
            sensor.width = 640
            sensor.height = 480
            sensor.hfov = 79
            sensor.position = [0.0, 0.88, 0.0]
        agent.sim_sensors.depth_sensor.min_depth = 0.0
        agent.sim_sensors.depth_sensor.max_depth = 5.0
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
    top_down = info.get("top_down_map")
    if top_down is None:
        return np.zeros((FRAME_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)
    image = habitat_maps.colorize_topdown_map(
        top_down["map"], top_down["fog_of_war_mask"]
    )
    coord = top_down["agent_map_coord"]
    if isinstance(coord, list):
        coord = coord[0]
    angle = top_down["agent_angle"]
    if isinstance(angle, list):
        angle = angle[0]
    angle = float(np.asarray(angle).item())
    image = habitat_maps.draw_agent(
        image=image,
        agent_center_coord=coord,
        agent_rotation=angle,
        agent_radius_px=max(1, min(image.shape[:2]) // 32),
    )
    return cv2.resize(image, (MAP_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)


def make_frame(obs, info, goal):
    rgb = cv2.resize(obs["rgb"], (RGB_WIDTH, FRAME_HEIGHT))
    overlay = rgb.copy()
    distance = info.get("distance_to_goal")
    distance = f"{float(distance):.2f}m" if distance is not None else "N/A"
    step_count = info.get("steps_taken", "")
    step_str = f"Step: {int(step_count)}" if step_count != "" else ""

    y = 30
    for text in [f"Goal: {goal}", f"Distance: {distance}"] + ([step_str] if step_str else []):
        (width, height), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        cv2.rectangle(overlay, (5, y - height - 5), (10 + width, y + 5), (0, 0, 0), -1)
        cv2.putText(
            overlay, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            (255, 255, 255), 2, cv2.LINE_AA,
        )
        y += height + 15

    return np.concatenate([overlay, render_map(info)], axis=1)


def append_jsonl(path, record):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def run_episode(env, oracle, index, output_dir, max_steps, fps):
    obs = env.reset()
    episode = env.current_episode
    scene = Path(episode.scene_id).name.replace(".basis.glb", "")
    episode_id = str(episode.episode_id)
    goal = str(getattr(episode, "object_category", "unknown"))
    clean = lambda value: "".join(
        c if c.isalnum() or c in "-_" else "_" for c in value
    )
    stem = f"{index:04d}_{clean(scene)}_{clean(episode_id)}_{clean(goal)}"
    video_path = output_dir / f"{stem}.mp4"
    temp_path = output_dir / f"{stem}.tmp.mp4"
    writer = cv2.VideoWriter(
        str(temp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
        (RGB_WIDTH + MAP_WIDTH, FRAME_HEIGHT),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer: {temp_path}")

    started, steps, stopped = time.time(), 0, False
    try:
        info = env.get_metrics()
        writer.write(cv2.cvtColor(make_frame(obs, info, goal), cv2.COLOR_RGB2BGR))
        while steps < max_steps and not env.episode_over:
            action = int(oracle.get_best_action())
            stopped = action == 0
            obs = env.step(action)
            steps += 1
            info = env.get_metrics()
            writer.write(cv2.cvtColor(make_frame(obs, info, goal), cv2.COLOR_RGB2BGR))
            if stopped:
                break
    finally:
        writer.release()

    if not temp_path.exists() or temp_path.stat().st_size == 0:
        raise RuntimeError(f"Empty video: {temp_path}")
    temp_path.replace(video_path)

    metrics = env.get_metrics()
    return {
        "status": "complete",
        "episode_index": index,
        "scene": scene,
        "episode_id": episode_id,
        "goal": goal,
        "steps": steps,
        "stopped": stopped,
        "success": metrics.get("success"),
        "spl": metrics.get("spl"),
        "soft_spl": metrics.get("soft_spl"),
        "distance_to_goal": metrics.get("distance_to_goal"),
        "elapsed_seconds": round(time.time() - started, 3),
        "video": str(video_path),
        "video_bytes": video_path.stat().st_size,
    }


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

    msg = (
        f"HM3D v2 oracle eval | episodes={total} | range=[{args.start}, {end}) | "
        f"already_complete={len(completed)} | habitat-sim=0.3.3"
    )
    print(msg, flush=True)
    with log_path.open("a") as lf:
        lf.write(f"\n{'='*80}\n{time.strftime('%Y-%m-%d %H:%M:%S')} — {msg}\n")

    successes, spls, episode_count = 0, 0.0, 0
    try:
        for index in range(end):
            if index < args.start or (index in completed and not args.overwrite):
                env.reset()
                continue
            try:
                record = run_episode(
                    env, oracle, index, output_dir, args.max_steps, args.fps
                )
                append_jsonl(results_path, record)
                episode_count += 1
                if record["success"]:
                    successes += 1
                spls += record.get("spl", 0) or 0
                running_sr = successes / episode_count * 100
                running_spl = spls / episode_count

                line = (
                    f"[{index + 1}/{end}] {record['scene']} "
                    f"goal={record['goal']} steps={record['steps']} "
                    f"success={record['success']} "
                    f"dist={record['distance_to_goal']:.3f} "
                    f"time={record['elapsed_seconds']:.1f}s "
                    f"| SR={running_sr:.1f}% SPL={running_spl:.3f}"
                )
                print(line, flush=True)
                with log_path.open("a") as lf:
                    lf.write(line + "\n")
            except Exception as error:
                err_record = {
                    "status": "failed", "episode_index": index,
                    "error": repr(error), "traceback": traceback.format_exc(),
                }
                append_jsonl(results_path, err_record)
                err_line = f"[{index + 1}/{end}] FAILED: {error!r}"
                print(err_line, flush=True)
                with log_path.open("a") as lf:
                    lf.write(err_line + "\n")
    finally:
        env.close()

    if episode_count > 0:
        summary = (
            f"\n{'='*60}\n"
            f"EVALUATION COMPLETE\n"
            f"  Episodes: {episode_count}\n"
            f"  Success Rate: {successes}/{episode_count} = {successes/episode_count*100:.1f}%\n"
            f"  Average SPL: {spls/episode_count:.4f}\n"
            f"  Results: {results_path}\n"
            f"  Videos: {output_dir}\n"
            f"{'='*60}"
        )
        print(summary, flush=True)
        with log_path.open("a") as lf:
            lf.write(summary + "\n")


if __name__ == "__main__":
    main()
