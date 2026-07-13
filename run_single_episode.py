"""
Standalone 1-episode evaluation with oracle (shortest-path) navigation.
Produces a side-by-side video: robot perspective | top-down map
Overlay shows only: goal name + distance to goal
"""
import os, sys
import numpy as np
import cv2

os.chdir("/home/ashed/Documents/spatial_training")
sys.path.insert(0, "src")

from habitat.config.default import get_config
from habitat.config import read_write
from habitat.config.default_structured_configs import (
    TopDownMapMeasurementConfig,
    FogOfWarConfig,
)
from habitat import make_dataset
from habitat.core.dataset import EpisodeIterator
from habitat.utils.visualizations import maps as habitat_maps
import longnav.utils.measures
import longnav.utils.ovon.ovon_dataset
import longnav.utils.ovon.ovon_nav

OUTPUT_DIR = "/home/ashed/Documents/spatial_training/dump/single_episode_test"
MAX_STEPS = 500
FPS = 10

config_path = "habitat_configs/objectnav_hm3d_rgbd_semantic.yaml"
dataset_path = "data/datasets/objectnav/hm3d/v2/val/val.json.gz"

config = get_config(config_path)

with read_write(config):
    config.habitat.dataset.data_path = dataset_path
    config.habitat.task.measurements.top_down_map = TopDownMapMeasurementConfig(
        map_padding=3,
        map_resolution=1024,
        draw_goal_positions=True,
        draw_shortest_path=True,
        draw_view_points=True,
        draw_border=True,
        fog_of_war=FogOfWarConfig(draw=True, visibility_dist=20, fov=79),
    )

dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
print(f"Total episodes in dataset: {dataset.num_episodes}")

import habitat
env = habitat.Env(config=config, dataset=dataset)

episode_iterator = EpisodeIterator(
    dataset.episodes,
    cycle=False,
    shuffle=False,
    group_by_scene=False,
    seed=17,
)
env.episode_iterator = episode_iterator

id_to_name = {}
if hasattr(dataset, "category_to_task_category_id"):
    id_to_name = {v: k for k, v in dataset.category_to_task_category_id.items()}

from longnav.env.habitat import ObjectNavOracle
oracle = ObjectNavOracle(
    env,
    success_distance=config.habitat.task.measurements.success.success_distance,
)

obs = env.reset()

current_episode = env.current_episode
scene_id = os.path.basename(current_episode.scene_id).replace(".basis.glb", "")
episode_id = current_episode.episode_id
goal_category = getattr(current_episode, 'object_category', 'unknown')
obj_id = obs.get('objectgoal', None)
if obj_id is not None:
    goal_name = id_to_name.get(
        obj_id.item() if hasattr(obj_id, 'item') else obj_id[0],
        goal_category
    )
else:
    goal_name = goal_category

print(f"Episode: {scene_id}_{episode_id}")
print(f"Goal: {goal_name}")
print(f"Num goals: {len(current_episode.goals)}")
print(f"Geodesic distance: {current_episode.info.get('geodesic_distance', 'N/A')}")

MAP_WIDTH = 480

def render_top_down_map(info, target_height):
    if 'top_down_map' not in info or info['top_down_map'] is None:
        return np.zeros((target_height, MAP_WIDTH, 3), dtype=np.uint8)

    tdm_info = info['top_down_map']
    tdm = habitat_maps.colorize_topdown_map(tdm_info['map'], tdm_info['fog_of_war_mask'])
    tdm = habitat_maps.draw_agent(
        image=tdm,
        agent_center_coord=tdm_info['agent_map_coord'],
        agent_rotation=tdm_info['agent_angle'],
        agent_radius_px=min(tdm.shape[0:2]) // 32,
    )
    # resize to fixed dimensions for consistent video frames
    tdm = cv2.resize(tdm, (MAP_WIDTH, target_height), interpolation=cv2.INTER_AREA)
    return tdm

def make_frame(obs, info, goal_name):
    rgb = obs['rgb']
    h, w = rgb.shape[:2]
    distance = info.get('distance_to_goal', None)
    dist_str = f"{distance:.2f}m" if distance is not None else "N/A"

    tdm = render_top_down_map(info, h)
    overlay = rgb.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    texts = [f"Goal: {goal_name}", f"Distance: {dist_str}"]
    y = 30
    for txt in texts:
        (tw, th), _ = cv2.getTextSize(txt, font, 0.6, 2)
        cv2.rectangle(overlay, (5, y - th - 5), (10 + tw, y + 5), (0, 0, 0), -1)
        cv2.putText(overlay, txt, (8, y), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        y += th + 15

    return np.concatenate([overlay, tdm], axis=1)

action_names = ["STOP", "FORWARD", "LEFT", "RIGHT", "LOOK_UP", "LOOK_DOWN"]
frames = []
actions_taken = []

# take first step to populate metrics before entering loop
action = oracle.get_best_action()
action = action if isinstance(action, int) else int(action)

# record initial frame (before first action)
info = env.get_metrics()
frames.append(make_frame(obs, info, goal_name))
print(f"  Step 0: action={action_names[action]}, dist={info.get('distance_to_goal','?'):.2f}m")

for step_i in range(MAX_STEPS):
    actions_taken.append(action)
    if action == 0:
        print(f"  Step {step_i}: STOP. Distance: {info.get('distance_to_goal','?'):.2f}m")
        break

    obs = env.step(action)
    info = env.get_metrics()
    done = env.episode_over

    frames.append(make_frame(obs, info, goal_name))

    if step_i % 25 == 0:
        print(f"  Step {step_i}: action={action_names[action]}, dist={info.get('distance_to_goal','?'):.2f}m")

    if done:
        print(f"  Episode done at step {step_i + 1}. Distance: {info.get('distance_to_goal','?'):.2f}m")
        break

    action = oracle.get_best_action()
    action = action if isinstance(action, int) else int(action)

final_metrics = env.get_metrics()
print(f"\n=== Episode Metrics ===")
print(f"  Success: {final_metrics.get('success', 'N/A')}")
print(f"  SPL: {final_metrics.get('spl', 'N/A')}")
print(f"  Distance to goal: {final_metrics.get('distance_to_goal', 'N/A')}")
print(f"  Steps taken: {len(actions_taken)}")

os.makedirs(OUTPUT_DIR, exist_ok=True)
video_path = os.path.join(OUTPUT_DIR, f"{scene_id}_{episode_id}_oracle.mp4")
h_frame, w_frame = frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(video_path, fourcc, FPS, (w_frame, h_frame))
for f in frames:
    writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
writer.release()

print(f"\nVideo saved to: {video_path}")
print(f"Frames: {len(frames)}, Resolution: {w_frame}x{h_frame}")

env.close()
