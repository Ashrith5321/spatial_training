"""Self-contained 3D scene-graph viewer for LongNav R1 eval episodes.

`render_html(data)` returns a single standalone HTML document (no external
assets, no libraries) that renders one episode's ground-truth scene graph in 3D:
rooms, ObjectNav-21 objects at their world-space centers, room-to-room
traversability edges, and the agent trajectory -- with a Spatial <-> Layered
toggle and an episode timeline that grows the graph and moves the agent.

`data` is the dict produced by `GTSceneGraphProvider.export(...)`:

    {
      "meta":    {"scene","episode_id","goal","steps","success"},
      "rooms":   [{"id":int,"name":str,"pos":[x,y,z],"disc":int}, ...],
      "objects": [{"cat":str,"room":int,"pos":[x,y,z],"disc":int}, ...],
      "edges":   [[room_a, room_b, disc], ...],
      "traj":    [{"p":[x,z],"h":[fx,fz]}, ...],   # one per recorded step
    }

The renderer is a hand-rolled canvas projector so the output opens in any
browser and passes the Artifact CSP (no CDN / WebGL library needed).
"""

import json

# The viewer body (CSS + markup + engine). Data is injected separately as
# window.__SG_DATA__ so this string never goes through str.format (it is full of
# literal { } from CSS and JS template literals).
_VIEWER = r"""<style>
  :root{
    --bg:#eef1f7;--panel:#fff;--panel-2:#f5f7fb;--viewport:#f4f6fb;
    --ink:#10151c;--ink-dim:#5a6472;--ink-mute:#8b95a3;
    --hair:rgba(16,21,28,.10);--hair-strong:rgba(16,21,28,.16);
    --accent:#0a6ebd;--good:#0ca30c;--seg-active:#fff;--on-accent:#fff;
    --shadow:0 1px 2px rgba(16,21,28,.04),0 8px 30px rgba(16,21,28,.07);
  }
  @media (prefers-color-scheme:dark){:root{
    --bg:#0b0e12;--panel:#141a21;--panel-2:#0f151b;--viewport:#0e1116;
    --ink:#eef2f7;--ink-dim:#9aa7b4;--ink-mute:#6b7684;
    --hair:rgba(255,255,255,.08);--hair-strong:rgba(255,255,255,.14);
    --accent:#64d2ff;--good:#3ddc84;--seg-active:#232c36;--on-accent:#08121a;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 12px 40px rgba(0,0,0,.45);
  }}
  :root[data-theme="light"]{
    --bg:#eef1f7;--panel:#fff;--panel-2:#f5f7fb;--viewport:#f4f6fb;
    --ink:#10151c;--ink-dim:#5a6472;--ink-mute:#8b95a3;
    --hair:rgba(16,21,28,.10);--hair-strong:rgba(16,21,28,.16);
    --accent:#0a6ebd;--good:#0ca30c;--seg-active:#fff;--on-accent:#fff;
  }
  :root[data-theme="dark"]{
    --bg:#0b0e12;--panel:#141a21;--panel-2:#0f151b;--viewport:#0e1116;
    --ink:#eef2f7;--ink-dim:#9aa7b4;--ink-mute:#6b7684;
    --hair:rgba(255,255,255,.08);--hair-strong:rgba(255,255,255,.14);
    --accent:#64d2ff;--good:#3ddc84;--seg-active:#232c36;--on-accent:#08121a;
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
    font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;line-height:1.5;-webkit-font-smoothing:antialiased}
  .mono{font-family:ui-monospace,"SF Mono","JetBrains Mono",Menlo,Consolas,monospace}
  .wrap{max-width:1080px;margin:0 auto;padding:0 20px}
  header.top{border-bottom:1px solid var(--hair);background:var(--panel-2)}
  .top-inner{display:flex;align-items:center;justify-content:space-between;gap:14px;padding:13px 0;flex-wrap:wrap}
  .brand{display:flex;align-items:center;gap:10px;font-weight:650;letter-spacing:-.01em}
  .brand .g{width:22px;height:22px}
  .meta{display:flex;gap:16px;flex-wrap:wrap;font:600 12px/1 ui-monospace,monospace;color:var(--ink-dim)}
  .meta b{color:var(--ink);font-weight:600}
  .meta .ok{color:var(--good)}.meta .no{color:#e34948}
  .theme-btn{border:1px solid var(--hair-strong);background:var(--panel);color:var(--ink-dim);
    border-radius:8px;padding:6px 11px;font:600 12px/1 ui-monospace,monospace;cursor:pointer}
  .theme-btn:hover{color:var(--ink);border-color:var(--accent)}
  .stage{margin:20px 0 8px;border:1px solid var(--hair-strong);border-radius:14px;overflow:hidden;
    background:var(--viewport);box-shadow:var(--shadow);position:relative}
  .holder{position:relative;width:100%;aspect-ratio:16/10;min-height:340px}
  @media(max-width:640px){.holder{aspect-ratio:4/5}}
  canvas{display:block;width:100%;height:100%;cursor:grab;touch-action:none}
  canvas:active{cursor:grabbing}
  .hud{position:absolute;top:13px;left:13px;pointer-events:none;font-family:ui-monospace,monospace;font-size:12px;
    background:color-mix(in srgb,var(--viewport) 78%,transparent);border:1px solid var(--hair);border-radius:10px;
    padding:10px 12px;backdrop-filter:blur(6px);max-width:250px}
  .hud .goal{display:flex;align-items:center;gap:7px;font-weight:650;font-size:13px;margin-bottom:6px}
  .hud .goal .dot{width:9px;height:9px;border-radius:50%;background:var(--good);box-shadow:0 0 0 3px color-mix(in srgb,var(--good) 26%,transparent)}
  .hud .row{display:flex;justify-content:space-between;gap:16px;color:var(--ink-dim);padding:1px 0}
  .hud .row b{color:var(--ink);font-weight:600;font-variant-numeric:tabular-nums}
  .badge{position:absolute;top:13px;right:13px;pointer-events:none;font:600 10px/1 ui-monospace,monospace;
    letter-spacing:.14em;text-transform:uppercase;color:var(--ink-mute);
    background:color-mix(in srgb,var(--viewport) 70%,transparent);border:1px solid var(--hair);border-radius:20px;padding:6px 11px}
  .tooltip{position:absolute;pointer-events:none;z-index:5;opacity:0;transition:opacity .09s;transform:translate(-50%,-116%);
    background:var(--panel);border:1px solid var(--hair-strong);border-radius:9px;padding:8px 10px;box-shadow:var(--shadow);font-size:12px;min-width:130px}
  .tooltip .c{font-weight:650;display:flex;align-items:center;gap:6px}
  .tooltip .c .sw{width:9px;height:9px;border-radius:2px}
  .tooltip .r{display:flex;justify-content:space-between;gap:14px;color:var(--ink-dim);margin-top:3px;font-family:ui-monospace,monospace;font-size:11px}
  .tooltip .r b{color:var(--ink);font-weight:600}
  .console{display:grid;grid-template-columns:1.3fr 1fr;gap:12px;margin:12px 0 6px}
  @media(max-width:820px){.console{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--hair);border-radius:13px;padding:14px 15px}
  .card h3{margin:0 0 12px;font:600 11px/1 ui-monospace,monospace;letter-spacing:.14em;text-transform:uppercase;color:var(--ink-mute)}
  .seg{display:inline-flex;background:var(--panel-2);border:1px solid var(--hair);border-radius:10px;padding:3px;gap:2px}
  .seg button{border:0;background:transparent;color:var(--ink-dim);font:600 13px/1 system-ui;padding:8px 15px;border-radius:7px;cursor:pointer}
  .seg button[aria-pressed="true"]{background:var(--seg-active);color:var(--ink);box-shadow:0 1px 2px rgba(0,0,0,.14)}
  .transport{display:flex;align-items:center;gap:12px;margin-top:13px}
  .play{width:40px;height:40px;flex:0 0 auto;border-radius:50%;border:1px solid var(--hair-strong);
    background:var(--accent);color:var(--on-accent);cursor:pointer;display:grid;place-items:center}
  .play:hover{filter:brightness(1.06)}
  input[type=range]{-webkit-appearance:none;appearance:none;flex:1;height:5px;border-radius:5px;background:var(--hair-strong);outline:none;cursor:pointer}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:16px;height:16px;border-radius:50%;background:var(--accent);border:2px solid var(--panel)}
  input[type=range]::-moz-range-thumb{width:16px;height:16px;border-radius:50%;background:var(--accent);border:2px solid var(--panel)}
  .stepread{font-family:ui-monospace,monospace;font-size:12px;color:var(--ink-dim);white-space:nowrap;font-variant-numeric:tabular-nums}
  .stepread b{color:var(--ink)}
  body.live .transport{display:none}
  .toggles{display:flex;flex-wrap:wrap;gap:8px}
  .toggle{display:inline-flex;align-items:center;gap:8px;border:1px solid var(--hair-strong);background:var(--panel-2);
    color:var(--ink-dim);border-radius:20px;padding:7px 12px;font:500 12.5px/1 system-ui;cursor:pointer;user-select:none}
  .toggle input{display:none}
  .toggle .tick{width:14px;height:14px;border-radius:4px;border:1.5px solid var(--ink-mute);position:relative;transition:.12s}
  .toggle input:checked + .tick{background:var(--accent);border-color:var(--accent)}
  .toggle input:checked + .tick::after{content:"";position:absolute;left:4px;top:1px;width:4px;height:8px;border:solid var(--on-accent);border-width:0 2px 2px 0;transform:rotate(45deg)}
  .toggle:has(input:checked){color:var(--ink);border-color:color-mix(in srgb,var(--accent) 50%,var(--hair-strong))}
  .ghost{border:1px solid var(--hair-strong);background:transparent;color:var(--ink-dim);border-radius:8px;padding:7px 12px;font:600 12px/1 ui-monospace,monospace;cursor:pointer;margin-top:11px}
  .ghost:hover{color:var(--ink);border-color:var(--accent)}
  .legend{display:flex;flex-wrap:wrap;gap:7px 15px}
  .lg{display:flex;align-items:center;gap:8px;font-size:13px;color:var(--ink-dim)}
  .lg .chip{width:13px;height:13px;border-radius:4px;border:1px solid rgba(0,0,0,.15)}
  .lg .chip.round{border-radius:50%}
  .lg small{color:var(--ink-mute);font-size:11.5px}
  .note{margin-top:11px;font-size:12.5px;color:var(--ink-mute)}
  footer{border-top:1px solid var(--hair);padding:18px 0 34px;color:var(--ink-mute);font-size:12px;margin-top:8px}
  footer code{font-family:ui-monospace,monospace;background:var(--panel);border:1px solid var(--hair);padding:1px 6px;border-radius:5px}
  @media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>

<header class="top"><div class="wrap top-inner">
  <div class="brand">
    <svg class="g" viewBox="0 0 32 32" fill="none"><circle cx="16" cy="7" r="3.4" fill="var(--accent)"/>
      <circle cx="7" cy="24" r="2.6" fill="currentColor"/><circle cx="25" cy="24" r="2.6" fill="currentColor"/>
      <path d="M16 10.4 9 21.6M16 10.4l7 11.2M9.4 24h13" stroke="currentColor" stroke-width="1.3" opacity=".5"/></svg>
    <span>Scene Graph</span>
  </div>
  <div class="meta" id="meta"></div>
  <button class="theme-btn" id="themeBtn">◐ Theme</button>
</div></header>

<div class="wrap">
  <div class="stage"><div class="holder">
    <canvas id="cv"></canvas>
    <div class="hud">
      <div class="goal"><span class="dot"></span><span>Goal:&nbsp;<span id="hudGoal">—</span></span></div>
      <div class="row"><span>step</span><b class="mono" id="hudStep">0</b></div>
      <div class="row"><span>rooms seen</span><b class="mono" id="hudRooms">0</b></div>
      <div class="row"><span>objects seen</span><b class="mono" id="hudObjs">0</b></div>
      <div class="row"><span>goal dist</span><b class="mono" id="hudGoalDist">—</b></div>
    </div>
    <div class="badge" id="modeBadge">Spatial</div>
    <div class="tooltip" id="tip"></div>
  </div></div>

  <div class="console">
    <div class="card"><h3>View</h3>
      <div class="seg" role="group" aria-label="View mode">
        <button id="mSpatial" aria-pressed="true">Spatial</button>
        <button id="mLayered" aria-pressed="false">Layered</button>
      </div>
      <div class="transport">
        <button class="play" id="playBtn" aria-label="Play"><svg id="playIcon" width="15" height="15" viewBox="0 0 16 16" fill="currentColor"><path d="M4 2.5v11l9-5.5z"/></svg></button>
        <input type="range" id="scrub" min="0" max="1" value="1" step="1" aria-label="Step">
        <span class="stepread">step <b id="stepNum">0</b>/<span id="stepMax">0</span></span>
      </div>
      <button class="ghost" id="resetBtn">⟲ Reset camera</button>
    </div>
    <div class="card"><h3>Layers</h3>
      <div class="toggles">
        <label class="toggle"><input type="checkbox" id="tRooms" checked><span class="tick"></span>Room bounds</label>
        <label class="toggle"><input type="checkbox" id="tHier" checked><span class="tick"></span>Object links</label>
        <label class="toggle"><input type="checkbox" id="tTrav" checked><span class="tick"></span>Traversability</label>
        <label class="toggle"><input type="checkbox" id="tTraj" checked><span class="tick"></span>Trajectory</label>
        <label class="toggle"><input type="checkbox" id="tLabels"><span class="tick"></span>All labels</label>
        <label class="toggle"><input type="checkbox" id="tSpin"><span class="tick"></span>Auto-rotate</label>
      </div>
    </div>
  </div>

  <div class="card"><h3>Legend</h3>
    <div class="legend" id="legend"></div>
    <div class="note">Nodes are colored <b style="color:var(--ink)">by room</b>; the object category is the text label. Diamonds are rooms, dots are objects, the reticle is the <b style="color:var(--ink)">goal</b> category, the glow is the agent. Faint lines = object→room; heavy lines = traversability the agent walked.</div>
  </div>
</div>
<footer><div class="wrap">Generated by <code>run_longnav_eval.py</code> · read-out of <code>EpisodeSceneGraph</code> · self-contained, no external assets.</div></footer>

<script>
(() => {
  "use strict";
  const root = document.documentElement;
  const themeMode = () => root.dataset.theme || (matchMedia("(prefers-color-scheme:dark)").matches?"dark":"light");
  document.getElementById("themeBtn").onclick = () => { root.dataset.theme = themeMode()==="dark"?"light":"dark"; };

  const PAL = {
    dark:{ view:"#0e1116",grid:"#1b2a38",gridMajor:"#24384a",ink:"#eef2f7",dim:"#9aa7b4",
      agent:"#eaf6ff",agentGlow:"rgba(120,205,255,.9)",good:"#3ddc84",roomOver:"#6b7684",
      rooms:["#3987e5","#199e70","#c98500","#008300","#9085e9","#e66767","#d55181","#d95926"] },
    light:{ view:"#f4f6fb",grid:"#dbe3f0",gridMajor:"#c5d2e6",ink:"#10151c",dim:"#5a6472",
      agent:"#0a3a5c",agentGlow:"rgba(10,110,189,.85)",good:"#0ca30c",roomOver:"#8b95a3",
      rooms:["#2a78d6","#1baf7a","#eda100","#008300","#4a3aa7","#e34948","#e87ba4","#eb6834"] }
  };
  const pal = () => PAL[themeMode()];

  // ---- geometry rebuilt from DATA (re-callable so live mode can poll) ------
  let DATA, ROOMS, roomById, OBJS, EDGES, TRAJ, NSTEP, GOALCAT, allX, allY, allZ, target, extent, ROOT, framed=false;
  const mid=a=>(Math.min(...a)+Math.max(...a))/2, span=a=>Math.max(...a)-Math.min(...a);
  function rebuild(D){
    DATA=D;
    ROOMS=(D.rooms||[]).map((r,i)=>({id:r.id,name:r.name||("room "+r.id),x:r.pos[0],z:r.pos[2],disc:r.disc,ci:i,hx:1.2,hz:1.2}));
    roomById={}; ROOMS.forEach(r=>roomById[r.id]=r);
    OBJS=(D.objects||[]).map((o,i)=>({id:i,room:o.room,cat:o.cat,x:o.pos[0],y:o.pos[1],z:o.pos[2],disc:o.disc}));
    ROOMS.forEach(r=>{ const xs=[],zs=[]; OBJS.forEach(o=>{ if(o.room===r.id){xs.push(o.x);zs.push(o.z);} });
      if(xs.length){ r.hx=Math.max(0.8,(Math.max(...xs)-Math.min(...xs))/2+0.4); r.hz=Math.max(0.8,(Math.max(...zs)-Math.min(...zs))/2+0.4); } });
    EDGES=(D.edges||[]).map(e=>({a:e[0],b:e[1],disc:e[2]}));
    TRAJ=(D.traj||[]).map(t=>({x:t.p[0],z:t.p[1],hx:(t.h&&t.h[0])||0,hz:(t.h&&t.h[1])||-1}));
    if(!TRAJ.length) TRAJ.push({x:0,z:0,hx:0,hz:-1});
    NSTEP=TRAJ.length-1;
    GOALCAT=((D.meta&&D.meta.goal)||"").trim();
    allX=[];allY=[];allZ=[];
    OBJS.forEach(o=>{allX.push(o.x);allY.push(o.y);allZ.push(o.z);});
    ROOMS.forEach(r=>{allX.push(r.x);allZ.push(r.z);});
    TRAJ.forEach(t=>{allX.push(t.x);allZ.push(t.z);});
    if(!allX.length){allX.push(0);allZ.push(0);} if(!allY.length)allY.push(0);
    if(!framed){ target={x:mid(allX),y:0.7,z:mid(allZ)}; extent=Math.max(4,span(allX),span(allZ)); framed=true; }
    ROOT={x:target.x,z:target.z};
  }
  rebuild(window.__SG_DATA__ || {meta:{},rooms:[],objects:[],edges:[],traj:[]});
  const rcolor = rid => { const r=roomById[rid], C=pal(); const i=r?r.ci:0; return i<C.rooms.length?C.rooms[i]:C.roomOver; };

  // ---- camera / projection ------------------------------------------------
  const cv=document.getElementById("cv"), ctx=cv.getContext("2d"), tip=document.getElementById("tip");
  const cam={ az:-0.62, el:0.92, R:extent*1.55 };
  const DEF={...cam};
  let W=0,H=0,DPR=1,focal=1;
  function resize(){ const r=cv.getBoundingClientRect(); DPR=Math.min(devicePixelRatio||1,2);
    W=r.width;H=r.height; cv.width=Math.round(W*DPR); cv.height=Math.round(H*DPR); ctx.setTransform(DPR,0,0,DPR,0,0);
    focal=(H/2)/Math.tan(38*Math.PI/180/2); }
  new ResizeObserver(resize).observe(cv.parentElement);
  let basis=null;
  const sub=(a,b)=>({x:a.x-b.x,y:a.y-b.y,z:a.z-b.z});
  const cross=(a,b)=>({x:a.y*b.z-a.z*b.y,y:a.z*b.x-a.x*b.z,z:a.x*b.y-a.y*b.x});
  const dot=(a,b)=>a.x*b.x+a.y*b.y+a.z*b.z;
  const norm=a=>{const l=Math.hypot(a.x,a.y,a.z)||1;return{x:a.x/l,y:a.y/l,z:a.z/l};};
  function computeBasis(){ const {az,el,R}=cam;
    const eye={x:target.x+R*Math.cos(el)*Math.sin(az),y:target.y+R*Math.sin(el),z:target.z+R*Math.cos(el)*Math.cos(az)};
    const f=norm(sub(target,eye)), rt=norm(cross(f,{x:0,y:1,z:0})); basis={eye,f,rt,up:cross(rt,f)}; }
  function project(p){ const d=sub(p,basis.eye), cz=dot(d,basis.f);
    if(cz<0.05) return {behind:true};
    return {x:W/2+(dot(d,basis.rt)/cz)*focal, y:H/2-(dot(d,basis.up)/cz)*focal, z:cz, behind:false}; }

  const ui={ mode:"spatial",morph:0,step:NSTEP,playing:false,spin:false,rooms:true,hier:true,trav:true,traj:true,labels:false };
  let hover=null;
  const layeredY=k=>k==="root"?4.6:k==="room"?2.6:0.15;
  const nodeY=(base,k)=>base*(1-ui.morph)+layeredY(k)*ui.morph;
  const agentAt=s=>{ const i=Math.floor(s),f=s-i,a=TRAJ[Math.min(i,NSTEP)],b=TRAJ[Math.min(i+1,NSTEP)];
    return {x:a.x+(b.x-a.x)*f,z:a.z+(b.z-a.z)*f}; };
  const heading=s=>{ const i=Math.max(0,Math.min(Math.floor(s),NSTEP)); return {x:TRAJ[i].hx,z:TRAJ[i].hz}; };
  function bearingLabel(dx,dz,s){ const h=heading(s),rx=h.z,rz=-h.x;
    const ang=Math.abs(Math.atan2(dx*rx+dz*rz,dx*h.x+dz*h.z))*180/Math.PI, side=(dx*rx+dz*rz)>=0?"right":"left";
    if(ang<=22.5)return"ahead"; if(ang<=67.5)return"ahead-"+side; if(ang<=112.5)return"beside-"+side; if(ang<=157.5)return"behind-"+side; return"behind"; }
  const dist2=(ax,az,bx,bz)=>Math.hypot(ax-bx,az-bz);
  const reveal=disc=>{ const s=ui.step; return s<disc?0:Math.min(1,(s-disc+1)/5); };

  function seg(a,b){ ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke(); }
  function strokeLine(pa,pb,color,alpha){ const a=project(pa),b=project(pb); if(a.behind||b.behind)return;
    ctx.globalAlpha=alpha; ctx.strokeStyle=color; seg(a,b); ctx.globalAlpha=1; }

  function draw(){
    const C=pal();
    ctx.clearRect(0,0,W,H); ctx.fillStyle=C.view; ctx.fillRect(0,0,W,H); computeBasis();
    const gA=1-ui.morph;
    if(gA>0.02){ const g0x=Math.floor(Math.min(...allX))-1,g1x=Math.ceil(Math.max(...allX))+1,
      g0z=Math.floor(Math.min(...allZ))-1,g1z=Math.ceil(Math.max(...allZ))+1; ctx.lineWidth=1;
      for(let gx=g0x;gx<=g1x;gx++){const m=gx%2===0; strokeLine({x:gx,y:0,z:g0z},{x:gx,y:0,z:g1z},m?C.gridMajor:C.grid,gA*(m?.9:.6));}
      for(let gz=g0z;gz<=g1z;gz++){const m=gz%2===0; strokeLine({x:g0x,y:0,z:gz},{x:g1x,y:0,z:gz},m?C.gridMajor:C.grid,gA*(m?.9:.6));}
    }
    const prims=[]; const push=(depth,fn)=>prims.push({depth,fn});
    const ag=agentAt(ui.step);

    if(ui.rooms) ROOMS.forEach(r=>{ const a=reveal(r.disc); if(a<=0)return; const col=rcolor(r.id);
      const yb=nodeY(0,"roomfloor"), yt=nodeY(2.1,"roomtop");
      const cs=[[r.x-r.hx,r.z-r.hz],[r.x+r.hx,r.z-r.hz],[r.x+r.hx,r.z+r.hz],[r.x-r.hx,r.z+r.hz]];
      const bot=cs.map(c=>({x:c[0],y:yb*(1-ui.morph)+2.4*ui.morph,z:c[1]})).map(project);
      const top=cs.map(c=>({x:c[0],y:yt*(1-ui.morph)+2.8*ui.morph,z:c[1]})).map(project);
      if(bot.some(p=>p.behind)||top.some(p=>p.behind))return;
      push((bot[0].z+top[2].z)/2,()=>{ ctx.lineWidth=1.2; ctx.globalAlpha=a*.55; ctx.strokeStyle=col;
        for(let i=0;i<4;i++){const j=(i+1)%4; seg(bot[i],bot[j]); seg(top[i],top[j]); seg(bot[i],top[i]);} ctx.globalAlpha=1; }); });

    if(ui.hier) OBJS.forEach(o=>{ const a=Math.min(reveal(o.disc),reveal((roomById[o.room]||{disc:0}).disc)); if(a<=0)return;
      const rr=roomById[o.room]; if(!rr)return;
      const rp=project({x:rr.x,y:nodeY(0.9,"room"),z:rr.z}), op=project({x:o.x,y:nodeY(o.y,"obj"),z:o.z});
      if(rp.behind||op.behind)return;
      push((rp.z+op.z)/2,()=>{ ctx.lineWidth=1; ctx.globalAlpha=a*.4; ctx.strokeStyle=rcolor(o.room); seg(rp,op); ctx.globalAlpha=1; }); });
    if(ui.morph>0.02) ROOMS.forEach(r=>{ const a=reveal(r.disc)*ui.morph; if(a<=0)return;
      const rt=project({x:ROOT.x,y:nodeY(0,"root"),z:ROOT.z}), rp=project({x:r.x,y:nodeY(0.9,"room"),z:r.z});
      if(rt.behind||rp.behind)return;
      push((rt.z+rp.z)/2,()=>{ ctx.lineWidth=1; ctx.globalAlpha=a*.5; ctx.strokeStyle=C.dim; seg(rt,rp); ctx.globalAlpha=1; }); });

    if(ui.trav) EDGES.forEach(e=>{ const a=reveal(e.disc); if(a<=0)return; const A=roomById[e.a],B=roomById[e.b]; if(!A||!B)return;
      const pa=project({x:A.x,y:nodeY(0.9,"room"),z:A.z}), pb=project({x:B.x,y:nodeY(0.9,"room"),z:B.z}); if(pa.behind||pb.behind)return;
      push((pa.z+pb.z)/2,()=>{ ctx.lineWidth=2.4; ctx.globalAlpha=a*.85; ctx.strokeStyle=C.ink; ctx.lineCap="round"; seg(pa,pb); ctx.lineCap="butt"; ctx.globalAlpha=1; }); });

    if(ui.traj && ui.morph<0.98){ const sMax=Math.floor(ui.step);
      for(let i=0;i<sMax;i++){ const A=project({x:TRAJ[i].x,y:0.05,z:TRAJ[i].z}), B=project({x:TRAJ[i+1].x,y:0.05,z:TRAJ[i+1].z});
        if(A.behind||B.behind)continue; const age=i/Math.max(sMax,1);
        push((A.z+B.z)/2-0.01,()=>{ ctx.lineWidth=2.6*(1-ui.morph); ctx.globalAlpha=(.15+.75*age)*(1-ui.morph); ctx.strokeStyle=C.agentGlow; ctx.lineCap="round"; seg(A,B); ctx.lineCap="butt"; ctx.globalAlpha=1; }); } }

    const labels=[];
    OBJS.forEach(o=>{ const a=reveal(o.disc); if(a<=0)return; const p=project({x:o.x,y:nodeY(o.y,"obj"),z:o.z}); if(p.behind)return;
      const col=rcolor(o.room), isGoal=GOALCAT&&o.cat===GOALCAT, isHover=hover&&hover.kind==="obj"&&hover.id===o.id;
      const r=Math.max(3.2,Math.min(9,focal/p.z*0.85))*(.6+.4*a)*(isHover?1.35:1);
      push(p.z,()=>{ if(isGoal){ ctx.globalAlpha=a; ctx.strokeStyle=C.good; ctx.lineWidth=2;
          ctx.beginPath(); ctx.arc(p.x,p.y,r+5,0,7); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(p.x-r-9,p.y);ctx.lineTo(p.x-r-2,p.y); ctx.moveTo(p.x+r+2,p.y);ctx.lineTo(p.x+r+9,p.y);
          ctx.moveTo(p.x,p.y-r-9);ctx.lineTo(p.x,p.y-r-2); ctx.moveTo(p.x,p.y+r+2);ctx.lineTo(p.x,p.y+r+9); ctx.stroke(); }
        ctx.globalAlpha=a; ctx.beginPath(); ctx.arc(p.x,p.y,r,0,7); ctx.fillStyle=col; ctx.fill();
        ctx.lineWidth=1.5; ctx.strokeStyle=themeMode()==="light"?"rgba(0,0,0,.35)":"rgba(255,255,255,.7)"; ctx.stroke();
        if(isHover){ ctx.strokeStyle=C.ink; ctx.lineWidth=2; ctx.beginPath(); ctx.arc(p.x,p.y,r+3,0,7); ctx.stroke(); } ctx.globalAlpha=1; });
      o._sx=p.x;o._sy=p.y;o._sr=r;
      if(ui.labels||isGoal||isHover) labels.push({x:p.x,y:p.y-r-6,z:p.z,t:o.cat,a,strong:isGoal||isHover}); });

    ROOMS.forEach(r=>{ const a=reveal(r.disc); if(a<=0)return; const p=project({x:r.x,y:nodeY(0.9,"room"),z:r.z}); if(p.behind)return;
      const col=rcolor(r.id), isHover=hover&&hover.kind==="room"&&hover.id===r.id, s=Math.max(6,Math.min(13,focal/p.z*1.15))*(isHover?1.25:1);
      push(p.z+0.001,()=>{ ctx.globalAlpha=a; ctx.fillStyle=col; ctx.strokeStyle=themeMode()==="light"?"rgba(0,0,0,.4)":"rgba(255,255,255,.85)"; ctx.lineWidth=1.5;
        ctx.beginPath(); ctx.moveTo(p.x,p.y-s);ctx.lineTo(p.x+s,p.y);ctx.lineTo(p.x,p.y+s);ctx.lineTo(p.x-s,p.y);ctx.closePath(); ctx.fill(); ctx.stroke(); ctx.globalAlpha=1; });
      r._sx=p.x;r._sy=p.y;r._sr=s; labels.push({x:p.x,y:p.y-s-7,z:p.z,t:r.name,a,strong:true,room:true}); });

    if(ui.morph>0.3){ const p=project({x:ROOT.x,y:nodeY(0,"root"),z:ROOT.z});
      if(!p.behind){ const a=ui.morph; push(p.z,()=>{ ctx.globalAlpha=a; ctx.fillStyle=C.dim; ctx.beginPath(); ctx.arc(p.x,p.y,9,0,7); ctx.fill(); ctx.globalAlpha=1; });
        labels.push({x:p.x,y:p.y-16,z:p.z,t:"scene",a,strong:true}); } }

    prims.sort((a,b)=>b.depth-a.depth).forEach(pr=>pr.fn());

    const ap=project({x:ag.x,y:nodeY(0.15,"obj"),z:ag.z});
    if(!ap.behind){ const h=heading(ui.step);
      const g=ctx.createRadialGradient(ap.x,ap.y,0,ap.x,ap.y,18); g.addColorStop(0,C.agentGlow); g.addColorStop(1,"transparent");
      ctx.fillStyle=g; ctx.beginPath(); ctx.arc(ap.x,ap.y,18,0,7); ctx.fill();
      const hp=project({x:ag.x+h.x*0.7,y:nodeY(0.15,"obj"),z:ag.z+h.z*0.7});
      if(!hp.behind){ ctx.strokeStyle=C.agent; ctx.lineWidth=2; seg(ap,hp); }
      ctx.fillStyle=C.agent; ctx.beginPath(); ctx.arc(ap.x,ap.y,5,0,7); ctx.fill();
      ctx.strokeStyle=themeMode()==="light"?"#fff":"#0e1116"; ctx.lineWidth=1.5; ctx.stroke(); }

    ctx.textAlign="center"; ctx.textBaseline="bottom";
    labels.sort((a,b)=>b.z-a.z).forEach(l=>{ ctx.font=(l.strong?"600 ":"500 ")+(l.room?12:11)+"px ui-monospace,monospace";
      const w=ctx.measureText(l.t).width; ctx.globalAlpha=Math.min(1,l.a)*0.72; ctx.fillStyle=C.view; ctx.fillRect(l.x-w/2-4,l.y-14,w+8,15);
      ctx.globalAlpha=Math.min(1,l.a); ctx.fillStyle=C.ink; ctx.fillText(l.t,l.x,l.y); }); ctx.globalAlpha=1;

    updateHud(ag);
  }

  function updateHud(ag){ const s=ui.step;
    document.getElementById("hudStep").textContent=Math.round(s)+" / "+NSTEP;
    document.getElementById("stepNum").textContent=Math.round(s);
    document.getElementById("hudRooms").textContent=ROOMS.filter(r=>s>=r.disc).length;
    document.getElementById("hudObjs").textContent=OBJS.filter(o=>s>=o.disc).length;
    const goals=OBJS.filter(o=>GOALCAT&&o.cat===GOALCAT&&s>=o.disc);
    if(goals.length){ let bd=1e9,bo=null; goals.forEach(o=>{const d=dist2(ag.x,ag.z,o.x,o.z); if(d<bd){bd=d;bo=o;}});
      document.getElementById("hudGoalDist").textContent="~"+bd.toFixed(1)+"m "+bearingLabel(bo.x-ag.x,bo.z-ag.z,s);
    } else document.getElementById("hudGoalDist").textContent="not seen yet"; }

  // ---- interaction --------------------------------------------------------
  let drag=null;
  cv.addEventListener("pointerdown",e=>{drag={x:e.clientX,y:e.clientY};cv.setPointerCapture(e.pointerId);});
  cv.addEventListener("pointerup",()=>drag=null);
  cv.addEventListener("pointermove",e=>{ if(drag){ cam.az-=(e.clientX-drag.x)*0.008; cam.el=Math.max(0.12,Math.min(1.45,cam.el+(e.clientY-drag.y)*0.006));
      drag={x:e.clientX,y:e.clientY}; ui.spin=false; document.getElementById("tSpin").checked=false;
    } else { const r=cv.getBoundingClientRect(); pick(e.clientX-r.left,e.clientY-r.top,e.clientX,e.clientY); } });
  cv.addEventListener("pointerleave",()=>{hover=null;tip.style.opacity=0;});
  cv.addEventListener("wheel",e=>{e.preventDefault(); cam.R=Math.max(extent*0.6,Math.min(extent*3.5,cam.R*(1+Math.sign(e.deltaY)*0.08)));},{passive:false});
  function pick(mx,my,px,py){ let best=null,bd=15;
    OBJS.forEach(o=>{ if(o._sx==null||ui.step<o.disc)return; const d=Math.hypot(mx-o._sx,my-o._sy); if(d<bd){bd=d;best={kind:"obj",id:o.id,o};} });
    ROOMS.forEach(r=>{ if(r._sx==null||ui.step<r.disc)return; const d=Math.hypot(mx-r._sx,my-r._sy); if(d<Math.max(15,r._sr+4)&&d<bd){bd=d;best={kind:"room",id:r.id,r};} });
    hover=best;
    if(best){ const ag=agentAt(ui.step);
      if(best.kind==="obj"){ const o=best.o,dx=o.x-ag.x,dz=o.z-ag.z,d=Math.hypot(dx,dz);
        tip.innerHTML=`<div class="c"><span class="sw" style="background:${rcolor(o.room)}"></span>${o.cat}</div>`+
          `<div class="r"><span>room</span><b>${(roomById[o.room]||{name:"?"}).name}</b></div>`+
          `<div class="r"><span>rel. to agent</span><b>~${d.toFixed(1)}m ${bearingLabel(dx,dz,ui.step)}</b></div>`+
          `<div class="r"><span>first seen</span><b>step ${o.disc}</b></div>`;
      } else { const r=best.r,n=OBJS.filter(o=>o.room===r.id&&ui.step>=o.disc).length;
        tip.innerHTML=`<div class="c"><span class="sw" style="background:${rcolor(r.id)}"></span>${r.name}</div>`+
          `<div class="r"><span>objects seen</span><b>${n}</b></div><div class="r"><span>entered</span><b>step ${r.disc}</b></div>`; }
      const hb=cv.getBoundingClientRect(); tip.style.left=(px-hb.left)+"px"; tip.style.top=(py-hb.top)+"px"; tip.style.opacity=1;
    } else tip.style.opacity=0; }

  // ---- controls -----------------------------------------------------------
  const $=id=>document.getElementById(id);
  function setMode(m){ ui.mode=m; $("mSpatial").setAttribute("aria-pressed",m==="spatial"); $("mLayered").setAttribute("aria-pressed",m==="layered"); $("modeBadge").textContent=m==="spatial"?"Spatial":"Layered"; }
  $("mSpatial").onclick=()=>setMode("spatial"); $("mLayered").onclick=()=>setMode("layered");
  const scrub=$("scrub"); scrub.max=NSTEP; scrub.value=NSTEP; $("stepMax").textContent=NSTEP;
  scrub.oninput=()=>{ ui.step=+scrub.value; if(ui.playing)togglePlay(); };
  function togglePlay(){ ui.playing=!ui.playing;
    $("playIcon").innerHTML=ui.playing?'<rect x="3" y="2.5" width="3.4" height="11"/><rect x="9.6" y="2.5" width="3.4" height="11"/>':'<path d="M4 2.5v11l9-5.5z"/>';
    if(ui.playing&&ui.step>=NSTEP)ui.step=0; }
  $("playBtn").onclick=togglePlay; $("resetBtn").onclick=()=>Object.assign(cam,DEF);
  const bind=(id,k)=>$(id).onchange=e=>ui[k]=e.target.checked;
  bind("tRooms","rooms");bind("tHier","hier");bind("tTrav","trav");bind("tTraj","traj");bind("tLabels","labels");
  $("tSpin").onchange=e=>ui.spin=e.target.checked;

  // ---- meta + legend ------------------------------------------------------
  function refreshMeta(){ const m=DATA.meta||{}; const el=document.getElementById("meta");
    const succ=m.success==null?"":(m.success>=1?'<span class="ok">✓ success</span>':'<span class="no">✗ fail</span>');
    el.innerHTML=`scene <b>${m.scene||"?"}</b>`+`&nbsp;·&nbsp;ep <b>${m.episode_id!=null?m.episode_id:"?"}</b>`+
      `&nbsp;·&nbsp;goal <b>${m.goal||"?"}</b>`+`&nbsp;·&nbsp;<b>${m.steps!=null?m.steps:NSTEP}</b> steps`+(succ?"&nbsp;·&nbsp;"+succ:"");
    document.getElementById("hudGoal").textContent=m.goal||"—";
    paintLegend(); }
  refreshMeta();
  function paintLegend(){ const el=document.getElementById("legend"); const C=pal(); el.innerHTML="";
    ROOMS.forEach(r=>{ const d=document.createElement("div"); d.className="lg";
      d.innerHTML=`<span class="chip" style="background:${rcolor(r.id)}"></span>${r.name}`; el.appendChild(d); });
    const g=document.createElement("div"); g.className="lg"; g.innerHTML=`<span class="chip round" style="background:transparent;border:2px solid var(--good)"></span>goal <small>(${GOALCAT||"?"})</small>`; el.appendChild(g);
    const a=document.createElement("div"); a.className="lg"; a.innerHTML=`<span class="chip round" style="background:var(--accent)"></span>agent`; el.appendChild(a); }

  // ---- loop ---------------------------------------------------------------
  const reduce=matchMedia("(prefers-reduced-motion:reduce)").matches; let last=performance.now();
  function tick(now){ const dt=Math.min(0.05,(now-last)/1000); last=now;
    const mt=ui.mode==="layered"?1:0; if(!reduce) ui.morph+=(mt-ui.morph)*Math.min(1,dt*5); else ui.morph=mt;
    if(ui.spin&&!drag) cam.az+=dt*0.25;
    if(ui.playing){ ui.step+=dt*12; if(ui.step>=NSTEP){ui.step=NSTEP;togglePlay();} scrub.value=ui.step; }
    draw(); requestAnimationFrame(tick); }
  new MutationObserver(paintLegend).observe(root,{attributes:true,attributeFilter:["data-theme"]});
  matchMedia("(prefers-color-scheme:dark)").addEventListener("change",paintLegend);

  // ---- live mode: poll the graph endpoint and pin to the latest step ------
  const LIVE = window.__SG_LIVE__;
  if(LIVE){
    document.body.classList.add("live");
    (async function poll(){
      try{ const r=await fetch(LIVE,{cache:"no-store"}); if(r.ok){ rebuild(await r.json()); ui.step=NSTEP; refreshMeta(); } }
      catch(e){}
      setTimeout(poll, 180);
    })();
    // forward movement keys up to the teleop shell even when the canvas has focus
    addEventListener("keydown",e=>{ if(window.parent && window.parent!==window) window.parent.postMessage({sgKey:e.key},"*"); });
  }

  resize(); setMode("spatial"); requestAnimationFrame(tick);
})();
</script>"""


def render_html(data: dict, live_url: str = None) -> str:
    """Full standalone HTML document for one episode's scene graph.

    `live_url` (used by the teleop server) turns on live mode: the page polls
    that URL for fresh graph snapshots, pins to the latest step, and hides the
    replay transport. `data` is then just the initial snapshot.
    """
    meta = data.get("meta", {})
    title = "Scene Graph · {} · {}".format(meta.get("scene", "?"), meta.get("goal", "?"))
    payload = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    live = "window.__SG_LIVE__={};".format(json.dumps(live_url)) if live_url else ""
    return (
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
        "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
        "<title>{}</title></head><body>".format(title)
        + "<script>window.__SG_DATA__=" + payload + ";" + live + "</script>"
        + _VIEWER
        + "</body></html>"
    )
