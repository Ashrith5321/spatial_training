"""Teleoperate a real Habitat sim and watch the GT scene graph build in real time.

Serves a local web app (open the printed URL in a browser):
  - left panel : the agent's live RGB view
  - right panel: the 3D scene graph (scene_graph_viewer, live mode), orbitable

Keys (focus the page):
  W / ArrowUp    forward        A / ArrowLeft  turn left
  D / ArrowRight turn right     Space / F      stop ("found")
  N              next episode

Rendering is headless (EGL -> JPEG -> browser), so no X display is needed. The
HTTP server is single-threaded on purpose: every sim call runs on the same
(main) thread that created the GL context.

    python teleop_scene_graph.py                 # real Habitat, HM3D v2 val
    python teleop_scene_graph.py --start 42 --port 8080
    python teleop_scene_graph.py --demo          # no Habitat; synthetic backend to test the UI

Actions match the eval action space: 0 stop, 1 forward, 2 left, 3 right.
"""

import argparse
import base64
import io
import json
import math
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / "src"))

from longnav.utils.scene_graph_viewer import render_html  # noqa: E402

DATASET_PATH = "data/datasets/objectnav/hm3d/v2/val/val.json.gz"
ACTION_NAMES = ["stop", "forward", "left", "right"]


def _jpeg_data_url(rgb: np.ndarray, quality: int = 80) -> str:
    """HxWx3 uint8 RGB -> 'data:image/jpeg;base64,...'."""
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(rgb)).save(buf, "JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# --------------------------------------------------------------------------- #
# Backends: a real Habitat env, or a synthetic one for testing the web UI.
# Both expose: reset(), step(action_id), frame() -> {rgb, meta, over}, export().
# --------------------------------------------------------------------------- #
class HabitatBackend:
    def __init__(self, start=0):
        import habitat
        from habitat import make_dataset
        from habitat.config import read_write
        from habitat.config.default import get_config
        from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig
        import longnav.utils.measures  # noqa: F401  (registers measures)
        import longnav.utils.ovon.ovon_dataset  # noqa: F401
        import longnav.utils.ovon.ovon_nav  # noqa: F401
        from habitat.core.dataset import EpisodeIterator
        from longnav.utils.gt_scene_graph import GTSceneGraphProvider

        config = get_config("benchmark/nav/objectnav/objectnav_hm3d.yaml")
        with read_write(config):
            config.habitat.dataset.data_path = DATASET_PATH
            config.habitat.dataset.split = "val"
            agent = config.habitat.simulator.agents.main_agent
            for s in (agent.sim_sensors.rgb_sensor, agent.sim_sensors.depth_sensor):
                s.width, s.height, s.hfov, s.position = 640, 480, 79, [0, 0.88, 0]
            agent.sim_sensors.semantic_sensor = HabitatSimSemanticSensorConfig(
                width=640, height=480, hfov=79, position=[0, 0.88, 0])
            agent.height, agent.radius = 0.88, 0.18
            config.habitat.simulator.turn_angle = 30
            config.habitat.simulator.habitat_sim_v0.gpu_device_id = 0
            config.habitat.simulator.habitat_sim_v0.allow_sliding = True
            config.habitat.environment.iterator_options.shuffle = False

        dataset = make_dataset(config.habitat.dataset.type, config=config.habitat.dataset)
        self.env = habitat.Env(config=config, dataset=dataset)
        self.env.episode_iterator = EpisodeIterator(
            dataset.episodes, cycle=True, shuffle=False, group_by_scene=False, seed=17)
        self.provider = GTSceneGraphProvider(text_format="json")
        self._obs = None
        # skip to the requested starting episode
        for _ in range(max(0, start)):
            self.env.reset()
        self.reset()

    def _agent_state(self):
        return self.env.sim.get_agent(0).get_state()

    def _accumulate(self):
        ep = self.env.current_episode
        sg_text = self.provider.describe(self.env.sim, ep.scene_id, self._obs["semantic"], self._agent_state())
        # Echo the exact string the VLM would receive this step (model-facing JSON).
        print(f"[step {self.provider.graph.step}] model SG: {sg_text}", flush=True)

    def reset(self):
        self._obs = self.env.reset()
        self.provider.reset_episode()
        self._accumulate()

    def step(self, action_id):
        if self.env.episode_over:
            return
        self._obs = self.env.step(int(action_id))
        self._accumulate()  # record the resulting view/pose (also on the final 'stop')

    def meta(self):
        ep = self.env.current_episode
        m = self.env.get_metrics()
        return {
            "scene": Path(ep.scene_id).name.replace(".basis.glb", ""),
            "episode_id": str(ep.episode_id),
            "goal": str(getattr(ep, "object_category", "?")),
            "steps": self.provider.graph.step,
            "success": m.get("success"),
            "distance_to_goal": m.get("distance_to_goal"),
            "over": self.env.episode_over,
        }

    def export(self):
        return self.provider.export(self.meta())

    def frame(self):
        return {"rgb": _jpeg_data_url(self._obs["rgb"]), "meta": self.meta()}


class DemoBackend:
    """Self-contained fake env (no Habitat) so the web UI can be exercised anywhere.

    A tiny apartment; the agent turns/steps with discrete actions and 'observes'
    nearby objects, building the same export schema the viewer consumes.
    """
    ROOMS = {0: ("living room", 1.0, -0.4), 1: ("hallway", 0.2, 3.1),
             2: ("kitchen", 3.3, 4.3), 3: ("bedroom", -3.1, 5.0)}
    OBJ = [("sofa", 0, 2.2, -1.0, .35), ("tv_monitor", 0, 0.0, -0.2, 1.1), ("table", 0, 1.1, 0.0, .4),
           ("chair", 0, 1.7, 0.6, .5), ("picture", 0, -0.4, -1.3, 1.6), ("plant", 0, -0.5, 0.5, .5),
           ("cabinet", 1, 0.8, 3.2, 1.2), ("picture", 1, -0.3, 3.3, 1.6),
           ("cabinet", 2, 4.0, 4.9, 1.4), ("sink", 2, 3.0, 5.1, .9), ("counter", 2, 3.7, 4.3, .9),
           ("bed", 3, -2.9, 4.8, .4), ("chest_of_drawers", 3, -4.2, 5.6, .7), ("chair", 3, -2.1, 5.6, .5)]

    def __init__(self):
        self.reset()

    def reset(self):
        self.x, self.z, self.yaw = 1.0, -0.6, 0.0  # yaw: 0 faces -z
        self.step_i = 0
        self.rooms_seen, self.objs_seen, self.edges = {}, {}, {}
        self.cur_room = None
        self.traj = []
        self._observe()

    def _observe(self):
        fwd = (math.sin(self.yaw), -math.cos(self.yaw))
        self.traj.append({"p": [self.x, self.z], "h": [fwd[0], fwd[1]]})
        dom, dom_d = None, 1e9
        for cat, room, ox, oz, oy in self.OBJ:
            d = math.hypot(ox - self.x, oz - self.z)
            if d < 2.6:
                key = (room, cat, ox, oz)
                self.objs_seen.setdefault(key, {"cat": cat, "room": room, "pos": [ox, oy, oz], "disc": self.step_i})
                self.rooms_seen.setdefault(room, self.step_i)
                if d < dom_d:  # dominant room = nearest observed object (mirrors most-visible)
                    dom, dom_d = room, d
        if dom is not None:
            if self.cur_room is not None and dom != self.cur_room:
                self.edges.setdefault(frozenset((self.cur_room, dom)), self.step_i)
            self.cur_room = dom

    def step(self, action_id):
        action_id = int(action_id)
        if action_id == 1:
            self.x += math.sin(self.yaw) * 0.25
            self.z += -math.cos(self.yaw) * 0.25
        elif action_id == 2:
            self.yaw -= math.radians(30)
        elif action_id == 3:
            self.yaw += math.radians(30)
        self.step_i += 1
        self._observe()

    def meta(self):
        return {"scene": "demo-apartment", "episode_id": "0", "goal": "bed",
                "steps": self.step_i, "success": None, "over": False}

    def export(self):
        rooms, room_pos = [], {}
        for rid, disc in self.rooms_seen.items():
            members = [o for o in self.objs_seen.values() if o["room"] == rid]
            c = np.mean([o["pos"] for o in members], axis=0)
            room_pos[rid] = c
            rooms.append({"id": rid, "name": self.ROOMS[rid][0], "pos": [float(c[0]), float(c[1]), float(c[2])], "disc": disc})
        edges = [[int(a), int(b), d] for e, d in self.edges.items() for a, b in [tuple(e)]]
        return {"meta": self.meta(), "rooms": rooms,
                "objects": list(self.objs_seen.values()), "edges": edges, "traj": list(self.traj)}

    def frame(self):
        # simple top-down sketch so the RGB pane shows something
        img = Image.new("RGB", (640, 480), (16, 20, 28))
        d = ImageDraw.Draw(img)
        sx, sy, sc = 320, 240, 42
        for cat, room, ox, oz, oy in self.OBJ:
            seen = any(k[0] == room and k[1] == cat for k in self.objs_seen)
            px, py = sx + (ox - self.x) * sc, sy + (oz - self.z) * sc
            col = (90, 200, 130) if cat == "bed" else ((120, 170, 230) if seen else (70, 80, 95))
            d.ellipse([px - 5, py - 5, px + 5, py + 5], fill=col)
        d.line([sx, sy, sx + math.sin(self.yaw) * 30, sy - math.cos(self.yaw) * 30], fill=(235, 246, 255), width=3)
        d.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], fill=(235, 246, 255))
        d.text((12, 12), "DEMO (no Habitat) — top-down sketch", fill=(150, 160, 175))
        return {"rgb": _jpeg_data_url(np.asarray(img)), "meta": self.meta()}


# --------------------------------------------------------------------------- #
# Web server
# --------------------------------------------------------------------------- #
SHELL = r"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Teleop · Scene Graph</title>
<style>
  :root{--bg:#eef1f7;--panel:#fff;--ink:#10151c;--dim:#5a6472;--hair:rgba(16,21,28,.14);--accent:#0a6ebd;--good:#0ca30c;--bad:#e34948}
  @media(prefers-color-scheme:dark){:root{--bg:#0b0e12;--panel:#141a21;--ink:#eef2f7;--dim:#9aa7b4;--hair:rgba(255,255,255,.12);--accent:#64d2ff;--good:#3ddc84;--bad:#e66767}}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:system-ui,-apple-system,"Segoe UI",sans-serif;height:100vh;display:flex;flex-direction:column}
  .mono{font-family:ui-monospace,"SF Mono",Menlo,monospace}
  header{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:11px 18px;border-bottom:1px solid var(--hair);flex-wrap:wrap}
  header .title{font-weight:650;letter-spacing:-.01em;display:flex;align-items:center;gap:9px}
  header .status{font:600 12px/1 ui-monospace,monospace;color:var(--dim)}
  header .status b{color:var(--ink)} header .status .ok{color:var(--good)} header .status .no{color:var(--bad)}
  main{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--hair);min-height:0}
  @media(max-width:820px){main{grid-template-columns:1fr;grid-template-rows:1fr 1fr}}
  .pane{position:relative;background:var(--bg);min-height:0;overflow:hidden}
  #rgb{width:100%;height:100%;object-fit:contain;background:#000;display:block}
  iframe{width:100%;height:100%;border:0;display:block;background:var(--bg)}
  .tag{position:absolute;top:10px;left:10px;font:600 10px/1 ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase;
    color:var(--dim);background:color-mix(in srgb,var(--bg) 72%,transparent);border:1px solid var(--hair);border-radius:20px;padding:6px 11px;pointer-events:none}
  footer{display:flex;align-items:center;gap:8px;flex-wrap:wrap;padding:10px 18px;border-top:1px solid var(--hair);font-size:13px;color:var(--dim)}
  kbd{font-family:ui-monospace,monospace;font-size:12px;background:var(--panel);border:1px solid var(--hair);border-bottom-width:2px;border-radius:6px;padding:3px 8px;color:var(--ink);min-width:24px;display:inline-block;text-align:center}
  footer .k{display:inline-flex;align-items:center;gap:6px;margin-right:6px}
  .spacer{flex:1}
  .btn{border:1px solid var(--hair);background:var(--panel);color:var(--ink);border-radius:8px;padding:7px 13px;font:600 12px/1 ui-monospace,monospace;cursor:pointer}
  .btn:hover{border-color:var(--accent)}
</style></head><body>
<header>
  <div class="title">🧭 Teleop <span style="color:var(--dim);font-weight:500">· live scene graph</span></div>
  <div class="status" id="status">connecting…</div>
</header>
<main>
  <div class="pane"><div class="tag">Agent RGB</div><img id="rgb" alt="agent view"></div>
  <div class="pane"><div class="tag">Scene graph (drag to orbit)</div><iframe id="viewer" src="/viewer" title="scene graph"></iframe></div>
</main>
<footer>
  <span class="k"><kbd>W</kbd> forward</span><span class="k"><kbd>A</kbd> left</span><span class="k"><kbd>D</kbd> right</span>
  <span class="k"><kbd>Space</kbd> stop</span><span class="k"><kbd>N</kbd> next episode</span>
  <span class="spacer"></span>
  <button class="btn" id="nextBtn">Next episode ↦</button>
</footer>
<script>
  const KEYMAP={ "w":1,"arrowup":1,"a":2,"arrowleft":2,"d":3,"arrowright":3," ":0,"f":0 };
  let busy=false, over=false;
  const statusEl=document.getElementById("status"), rgb=document.getElementById("rgb");
  function paint(m){
    over=!!m.over;
    const succ=m.success==null?"":(m.success>=1?'<span class="ok">✓ found</span>':'<span class="no">✗</span>');
    const overTag=over?'&nbsp;·&nbsp;<b>episode over</b> — press N':'';
    statusEl.innerHTML=`scene <b>${m.scene||"?"}</b>&nbsp;·&nbsp;goal <b>${m.goal||"?"}</b>`+
      `&nbsp;·&nbsp;step <b>${m.steps}</b>`+(m.distance_to_goal!=null?`&nbsp;·&nbsp;dist <b>${(+m.distance_to_goal).toFixed(2)}m</b>`:"")+
      (succ?"&nbsp;·&nbsp;"+succ:"")+overTag;
  }
  async function call(path,body){
    if(busy) return; busy=true;
    try{ const r=await fetch(path,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(body||{})});
      const d=await r.json(); if(d.rgb) rgb.src=d.rgb; if(d.meta) paint(d.meta);
    }catch(e){ statusEl.textContent="server error"; } finally{ busy=false; }
  }
  function act(id){ if(over){ return; } call("/api/step",{action:id}); }
  function nextEp(){ call("/api/reset",{}); }
  function onKey(k){ k=(k||"").toLowerCase(); if(k==="n"){ nextEp(); return; } if(k in KEYMAP){ act(KEYMAP[k]); } }
  addEventListener("keydown",e=>{ const k=e.key.toLowerCase(); if(k==="n"||k in KEYMAP){ e.preventDefault(); onKey(k); } });
  // keys pressed while the 3D iframe has focus are forwarded here
  addEventListener("message",e=>{ if(e.data && e.data.sgKey) onKey(e.data.sgKey); });
  document.getElementById("nextBtn").onclick=nextEp;
  // initial frame
  fetch("/api/frame").then(r=>r.json()).then(d=>{ if(d.rgb) rgb.src=d.rgb; if(d.meta) paint(d.meta); });
</script></body></html>"""


def make_handler(backend):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # quiet
            pass

        def _send(self, code, body, ctype):
            data = body.encode("utf-8") if isinstance(body, str) else body
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj):
            self._send(200, json.dumps(obj), "application/json")

        def _body(self):
            n = int(self.headers.get("Content-Length", 0) or 0)
            if not n:
                return {}
            try:
                return json.loads(self.rfile.read(n) or b"{}")
            except json.JSONDecodeError:
                return {}

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index"):
                self._send(200, SHELL, "text/html; charset=utf-8")
            elif self.path.startswith("/viewer"):
                html = render_html(backend.export(), live_url="/api/state")
                self._send(200, html, "text/html; charset=utf-8")
            elif self.path.startswith("/api/state"):
                self._json(backend.export())
            elif self.path.startswith("/api/frame"):
                self._json(backend.frame())
            else:
                self._send(404, "not found", "text/plain")

        def do_POST(self):
            if self.path.startswith("/api/step"):
                backend.step(self._body().get("action", 0))
                self._json(backend.frame())
            elif self.path.startswith("/api/reset"):
                backend.reset()
                self._json(backend.frame())
            else:
                self._send(404, "not found", "text/plain")

    return Handler


def main():
    ap = argparse.ArgumentParser(description="Teleoperate Habitat and watch the scene graph build live.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--start", type=int, default=0, help="starting episode index (Habitat mode)")
    ap.add_argument("--demo", action="store_true", help="synthetic backend, no Habitat (UI test)")
    args = ap.parse_args()

    if args.demo:
        print("Starting DEMO backend (no Habitat).")
        backend = DemoBackend()
    else:
        print("Loading Habitat env (this can take a bit)…")
        backend = HabitatBackend(start=args.start)
        print("Habitat ready. Current episode:", backend.meta())

    server = HTTPServer((args.host, args.port), make_handler(backend))
    url = f"http://{args.host}:{args.port}/"
    print(f"\n  Teleop server ready →  {url}\n  (over SSH: forward the port, e.g.  ssh -L {args.port}:localhost:{args.port} <host>)\n"
          "  Keys: W forward · A left · D right · Space stop · N next episode.  Ctrl-C to quit.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
    finally:
        server.server_close()
        if hasattr(backend, "env"):
            backend.env.close()


if __name__ == "__main__":
    main()
