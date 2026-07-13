import json, glob
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'text.usetex': False,
})

COLORS = {
    'success': '#2ecc71',
    'false_stop': '#e74c3c',
    'collision': '#f39c12',
    'exploration': '#3498db',
    'accent': '#9b59b6',
    'bar': '#2c3e50',
    'grid': '#ecf0f1',
}

OBJ_COLORS = {
    'bed': '#e74c3c',
    'chair': '#3498db',
    'plant': '#2ecc71',
    'sofa': '#f39c12',
    'toilet': '#9b59b6',
    'tv_monitor': '#1abc9c',
}

files = glob.glob('/home/ashed/Documents/spatial_training/dump/longnav_eval/hm3d_v2_eval/rollout/**/summary.json', recursive=True)
episodes = {}
for f in files:
    with open(f) as fh:
        d = json.load(fh)
    label = d['episode_label']
    if label not in episodes:
        episodes[label] = d

data = list(episodes.values())
N = len(data)

# Categorize outcomes
outcomes = {'Success': 0, 'False Positive Stop': 0, 'Collision Failure': 0, 'Exploration Timeout': 0}
for d in data:
    if d['success'] == 1.0:
        outcomes['Success'] += 1
    elif d['n_steps'] >= 500:
        outcomes['Exploration Timeout'] += 1
    elif d['collision_rate'] > 0.05:
        outcomes['Collision Failure'] += 1
    else:
        outcomes['False Positive Stop'] += 1

# ========== FIGURE 1: Episode Outcome Distribution ==========
fig, ax = plt.subplots(figsize=(6, 4))
cats = list(outcomes.keys())
vals = [outcomes[c] for c in cats]
pcts = [v / N * 100 for v in vals]
colors_1 = [COLORS['success'], COLORS['false_stop'], COLORS['collision'], COLORS['exploration']]

bars = ax.barh(cats, vals, color=colors_1, edgecolor='white', linewidth=0.5, height=0.6)
for bar, v, p in zip(bars, vals, pcts):
    ax.text(bar.get_width() + 8, bar.get_y() + bar.get_height()/2,
            f'{v} ({p:.1f}%)', va='center', fontsize=10, fontweight='bold')

ax.set_xlabel('Number of Episodes')
ax.set_title('Episode Outcome Distribution')
ax.set_xlim(0, max(vals) * 1.25)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('/home/ashed/Documents/spatial_training/figures/fig1_outcome_distribution.pdf')
plt.savefig('/home/ashed/Documents/spatial_training/figures/fig1_outcome_distribution.png')
plt.close()
print('Fig 1 done')

# ========== FIGURE 2: SR by Object Category ==========
obj_stats = defaultdict(lambda: {'total': 0, 'success': 0})
for d in data:
    g = d['goal']
    obj_stats[g]['total'] += 1
    obj_stats[g]['success'] += d['success']

goals_sorted = sorted(obj_stats.keys(), key=lambda g: obj_stats[g]['success']/obj_stats[g]['total'], reverse=True)

fig, ax = plt.subplots(figsize=(6, 4))
x = np.arange(len(goals_sorted))
srs = [obj_stats[g]['success']/obj_stats[g]['total']*100 for g in goals_sorted]
bar_colors = [OBJ_COLORS.get(g, '#2c3e50') for g in goals_sorted]

bars = ax.bar(x, srs, color=bar_colors, edgecolor='white', linewidth=0.5, width=0.6)
for bar, sr, g in zip(bars, srs, goals_sorted):
    n = obj_stats[g]['total']
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{sr:.1f}%\n(n={n})', ha='center', va='bottom', fontsize=9)

ax.set_xticks(x)
ax.set_xticklabels([g.replace('_', ' ').title() for g in goals_sorted], fontsize=10)
ax.set_ylabel('Success Rate (%)')
ax.set_title('Success Rate by Object Category')
ax.set_ylim(0, 105)
ax.axhline(y=81.7, color='gray', linestyle='--', alpha=0.5, label=f'Overall SR={81.7:.1f}%')
ax.legend(loc='lower right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('/home/ashed/Documents/spatial_training/figures/fig2_sr_by_object.pdf')
plt.savefig('/home/ashed/Documents/spatial_training/figures/fig2_sr_by_object.png')
plt.close()
print('Fig 2 done')

# ========== FIGURE 3: SR and Failure Breakdown by Object ==========
fig, ax = plt.subplots(figsize=(7, 4.5))
width = 0.55

obj_outcomes = defaultdict(lambda: {'Success': 0, 'False Positive Stop': 0, 'Collision Failure': 0, 'Exploration Timeout': 0})
for d in data:
    g = d['goal']
    if d['success'] == 1.0:
        obj_outcomes[g]['Success'] += 1
    elif d['n_steps'] >= 500:
        obj_outcomes[g]['Exploration Timeout'] += 1
    elif d['collision_rate'] > 0.05:
        obj_outcomes[g]['Collision Failure'] += 1
    else:
        obj_outcomes[g]['False Positive Stop'] += 1

x = np.arange(len(goals_sorted))
bottom = np.zeros(len(goals_sorted))
cats_stack = ['Success', 'False Positive Stop', 'Collision Failure', 'Exploration Timeout']
stack_colors = [COLORS['success'], COLORS['false_stop'], COLORS['collision'], COLORS['exploration']]

for cat, color in zip(cats_stack, stack_colors):
    vals = [obj_outcomes[g][cat] / obj_stats[g]['total'] * 100 for g in goals_sorted]
    ax.bar(x, vals, width, bottom=bottom, label=cat, color=color, edgecolor='white', linewidth=0.3)
    bottom += np.array(vals)

ax.set_xticks(x)
ax.set_xticklabels([g.replace('_', ' ').title() for g in goals_sorted], fontsize=10)
ax.set_ylabel('Percentage (%)')
ax.set_title('Success and Failure Breakdown by Object Category')
ax.set_ylim(0, 108)
ax.legend(loc='upper right', fontsize=8, framealpha=0.9)
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('/home/ashed/Documents/spatial_training/figures/fig3_sr_failure_by_object.pdf')
plt.savefig('/home/ashed/Documents/spatial_training/figures/fig3_sr_failure_by_object.png')
plt.close()
print('Fig 3 done')

# ========== FIGURE 4: Per-Scene SR ==========
scene_stats = defaultdict(lambda: {'total': 0, 'success': 0})
for d in data:
    s = d['scene_id']
    scene_stats[s]['total'] += 1
    scene_stats[s]['success'] += d['success']

scenes_sorted = sorted(scene_stats.keys(), key=lambda s: scene_stats[s]['success']/scene_stats[s]['total'], reverse=True)
scene_srs = [scene_stats[s]['success']/scene_stats[s]['total']*100 for s in scenes_sorted]

fig, ax = plt.subplots(figsize=(12, 4.5))
x = np.arange(len(scenes_sorted))
bar_colors_scene = [COLORS['success'] if sr >= 81.7 else '#e74c3c' for sr in scene_srs]
bars = ax.bar(x, scene_srs, color=bar_colors_scene, edgecolor='white', linewidth=0.3, width=0.7)

ax.axhline(y=81.7, color='gray', linestyle='--', alpha=0.6, linewidth=1.2, label=f'Overall SR={81.7:.1f}%')
ax.set_xticks(x)
ax.set_xticklabels(scenes_sorted, rotation=65, ha='right', fontsize=7)
ax.set_ylabel('Success Rate (%)')
ax.set_title('Per-Scene Success Rate (36 HM3D v2 Val Scenes)')
ax.set_ylim(0, 105)
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig('/home/ashed/Documents/spatial_training/figures/fig4_per_scene_sr.pdf')
plt.savefig('/home/ashed/Documents/spatial_training/figures/fig4_per_scene_sr.png')
plt.close()
print('Fig 4 done')

# ========== FIGURE 5: Bottom 10 Scene Analysis ==========
bottom10 = scenes_sorted[-10:]
bottom10_data = defaultdict(lambda: {
    'sr': 0, 'total': 0, 'success': 0,
    'collision_rate': [], 'false_stop': 0, 'collision_fail': 0, 'exploration_timeout': 0
})

for d in data:
    s = d['scene_id']
    if s in bottom10:
        bottom10_data[s]['total'] += 1
        bottom10_data[s]['success'] += d['success']
        bottom10_data[s]['collision_rate'].append(d['collision_rate'])
        if d['success'] == 0:
            if d['n_steps'] >= 500:
                bottom10_data[s]['exploration_timeout'] += 1
            elif d['collision_rate'] > 0.05:
                bottom10_data[s]['collision_fail'] += 1
            else:
                bottom10_data[s]['false_stop'] += 1

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

scenes_b10 = sorted(bottom10, key=lambda s: scene_stats[s]['success']/scene_stats[s]['total'])
x = np.arange(len(scenes_b10))

# Panel A: SR
srs_b10 = [scene_stats[s]['success']/scene_stats[s]['total']*100 for s in scenes_b10]
axes[0].barh(x, srs_b10, color=COLORS['false_stop'], edgecolor='white', height=0.6)
for i, (sr, s) in enumerate(zip(srs_b10, scenes_b10)):
    n = scene_stats[s]['total']
    axes[0].text(sr + 1, i, f'{sr:.0f}% (n={n})', va='center', fontsize=8)
axes[0].set_yticks(x)
axes[0].set_yticklabels(scenes_b10, fontsize=8)
axes[0].set_xlabel('Success Rate (%)')
axes[0].set_title('(a) Success Rate')
axes[0].set_xlim(0, 100)
axes[0].grid(axis='x', alpha=0.3, linestyle='--')

# Panel B: Mean collision rate
coll_means = [np.mean(bottom10_data[s]['collision_rate'])*100 for s in scenes_b10]
axes[1].barh(x, coll_means, color=COLORS['collision'], edgecolor='white', height=0.6)
for i, c in enumerate(coll_means):
    axes[1].text(c + 0.1, i, f'{c:.1f}%', va='center', fontsize=8)
axes[1].set_yticks(x)
axes[1].set_yticklabels(scenes_b10, fontsize=8)
axes[1].set_xlabel('Mean Collision Rate (%)')
axes[1].set_title('(b) Collision Rate')
axes[1].grid(axis='x', alpha=0.3, linestyle='--')

# Panel C: Failure breakdown (stacked)
fs_vals = [bottom10_data[s]['false_stop'] for s in scenes_b10]
cf_vals = [bottom10_data[s]['collision_fail'] for s in scenes_b10]
et_vals = [bottom10_data[s]['exploration_timeout'] for s in scenes_b10]

axes[2].barh(x, fs_vals, height=0.6, label='False Positive Stop', color=COLORS['false_stop'], edgecolor='white', linewidth=0.3)
axes[2].barh(x, cf_vals, height=0.6, left=fs_vals, label='Collision Failure', color=COLORS['collision'], edgecolor='white', linewidth=0.3)
left2 = [a+b for a,b in zip(fs_vals, cf_vals)]
axes[2].barh(x, et_vals, height=0.6, left=left2, label='Exploration Timeout', color=COLORS['exploration'], edgecolor='white', linewidth=0.3)
axes[2].set_yticks(x)
axes[2].set_yticklabels(scenes_b10, fontsize=8)
axes[2].set_xlabel('Number of Failures')
axes[2].set_title('(c) Failure Breakdown')
axes[2].legend(fontsize=7, loc='lower right')
axes[2].grid(axis='x', alpha=0.3, linestyle='--')

plt.suptitle('Bottom 10 Scenes Analysis', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/home/ashed/Documents/spatial_training/figures/fig5_bottom10_analysis.pdf')
plt.savefig('/home/ashed/Documents/spatial_training/figures/fig5_bottom10_analysis.png')
plt.close()
print('Fig 5 done')

# ========== FIGURE 6: SR vs Success Distance Threshold ==========
thresholds = np.arange(0.1, 1.05, 0.05)
srs_by_thresh = []
for t in thresholds:
    successes = sum(1 for d in data if d['distance_to_goal'] <= t)
    srs_by_thresh.append(successes / N * 100)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(thresholds, srs_by_thresh, 'o-', color='#2c3e50', linewidth=2, markersize=4, label='LongNav-R1')
ax.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
ax.annotate(f'SR@1.0m = {srs_by_thresh[-1]:.1f}%',
            xy=(1.0, srs_by_thresh[-1]), xytext=(0.7, srs_by_thresh[-1] - 8),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='gray'),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='wheat', alpha=0.5))

ax.set_xlabel('Success Distance Threshold (m)')
ax.set_ylabel('Success Rate (%)')
ax.set_title('Success Rate vs. Success Distance Threshold')
ax.set_xlim(0.05, 1.1)
ax.set_ylim(0, 100)
ax.grid(alpha=0.3, linestyle='--')
ax.legend(loc='lower right')
plt.tight_layout()
plt.savefig('/home/ashed/Documents/spatial_training/figures/fig6_sr_vs_distance.pdf')
plt.savefig('/home/ashed/Documents/spatial_training/figures/fig6_sr_vs_distance.png')
plt.close()
print('Fig 6 done')

print('\nAll figures saved to /home/ashed/Documents/spatial_training/figures/')
