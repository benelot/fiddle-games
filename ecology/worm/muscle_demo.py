#!/usr/bin/env python3
"""
Hanging mass experiment — Thelen 2003 Hill-type muscle.

A single muscle suspends a 1 kg box against gravity from a fixed ceiling.
A sinusoidal neural drive causes the muscle to contract and relax, lifting
and lowering the box.  The CE is rendered as a Geijtenbeek-style cylinder
coloured blue→red by activation level.

Output: ecology/worm/muscle_demo.html
"""

import json
import math
from pathlib import Path

# ── Thelen 2003 parameters ────────────────────────────────────────────────────
F_MAX  = 20.0    # N    max isometric force
L_OPT  = 0.12    # m    optimal CE fibre length
L_SLACK= 0.05    # m    tendon slack length (rigid tendon)
V_MAX  = 1.2     # m/s  max shortening speed (= 10 × l_opt)
AF     = 0.25    # Hill concentric shape constant
FLEN   = 1.4     # eccentric force ceiling
KSHAPE = 0.45    # Gaussian σ for fl  (Thelen KshapeActive)
KPE    = 5.0     # PEE exponential curvature
E0     = 0.6     # PEE strain at F_iso
TAU_A  = 0.015   # s   activation rise  (Thelen)
TAU_D  = 0.050   # s   activation decay (Thelen)

# ── System ────────────────────────────────────────────────────────────────────
MASS   = 1.0     # kg
G      = 9.81    # m/s²
Y_FLOOR= 0.55    # m below ceiling (hard floor)
Y_INIT = 0.29    # m initial box position
DT     = 0.001   # s integration timestep
T_END  = 10.0    # s simulation duration
FREQ   = 0.35    # Hz neural drive frequency


# ── Thelen 2003 curve functions ───────────────────────────────────────────────

def fl(l_ce: float) -> float:
    """Active force-length: Gaussian (Thelen 2003)."""
    lceN = l_ce / L_OPT
    return math.exp(-((lceN - 1.0) / KSHAPE) ** 2)


def fv(v_ce: float) -> float:
    """Force-velocity: Hill concentric + Thelen 2003 eccentric rational form."""
    v_bar = v_ce / V_MAX
    if v_bar <= 0.0:
        raw = (1.0 + v_bar) / (1.0 - v_bar / AF)
        return max(0.0, min(1.0, raw))
    else:
        c = (2.0 + 2.0 / AF) / (FLEN - 1.0)
        raw = (1.0 + c * FLEN * v_bar) / (1.0 + c * v_bar)
        return max(1.0, min(FLEN, raw))


def fpe(l_ce: float) -> float:
    """Passive force-length: Thelen 2003 exponential."""
    lceN = l_ce / L_OPT
    if lceN <= 1.0:
        return 0.0
    return (math.exp(KPE * (lceN - 1.0) / E0) - 1.0) / (math.exp(KPE) - 1.0)


def activation_rate(u: float, a: float) -> float:
    """Thelen 2003 variable-τ: da/dt = (u−a)/τ(u,a)."""
    a_c = max(0.01, min(1.0, a))
    tau = TAU_A * (0.5 + 1.5 * a_c) if u > a_c else TAU_D / (0.5 + 1.5 * a_c)
    return (u - a_c) / tau


# ── Simulation ────────────────────────────────────────────────────────────────

def simulate() -> list[dict]:
    n = int(T_END / DT)
    y, vy, a = Y_INIT, 0.0, 0.5
    frames = []
    stride = 5  # record every 5 ms → 200 Hz playback

    for i in range(n):
        t = i * DT

        # Neural drive: sinusoid offset so muscle is always partially active
        u = 0.50 + 0.46 * math.sin(2.0 * math.pi * FREQ * t)

        # Rigid-tendon MTU kinematics
        l_mtu = y
        v_mtu = vy
        l_ce = max(l_mtu - L_SLACK, 0.005)
        v_ce = v_mtu

        # Muscle force (upward)
        fl_v  = fl(l_ce)
        fv_v  = fv(v_ce)
        fpe_v = fpe(l_ce)
        F = max(0.0, (a * fl_v * fv_v + fpe_v) * F_MAX)

        # Record before integration
        if i % stride == 0:
            frames.append({
                't':    round(t, 4),
                'y':    round(y, 5),
                'a':    round(a, 4),
                'u':    round(u, 4),
                'F':    round(F, 3),
                'l_ce': round(l_ce, 5),
                'fl':   round(fl_v, 4),
                'fv':   round(fv_v, 4),
                'fpe':  round(fpe_v, 4),
            })

        # Equation of motion (positive y = downward)
        acc = G - F / MASS

        # Constraints
        if y >= Y_FLOOR and vy > 0.0:
            vy *= -0.35            # soft bounce off floor
            y   = Y_FLOOR - 1e-4
        min_y = L_SLACK + 0.005
        if y <= min_y and vy < 0.0:
            vy = 0.0
            y  = min_y + 1e-4

        # Euler integration
        y  += vy * DT
        vy += acc * DT
        a   = max(0.0, min(1.0, a + activation_rate(u, a) * DT))

    return frames


# ── HTML generation ───────────────────────────────────────────────────────────

_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Muscle Demo</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap">
<style>
:root {
  --bg: #f2f1ed;
  --surface: #e8e6e0;
  --border: #ccc9bf;
  --text: #1c1b18;
  --muted: #6b6960;
  --teal: #3aaa8f;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'JetBrains Mono', 'Courier New', monospace;
  padding: 28px 32px;
}
header { margin-bottom: 22px; }
header h1 { font-size: 12px; font-weight: 600; letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
header p  { font-size: 10px; color: var(--muted); margin-top: 3px; }
.layout { display: flex; gap: 36px; align-items: flex-start; }
.left  { flex: 0 0 200px; }
.right { flex: 1; min-width: 0; }

/* canvas */
#scene {
  display: block;
  background: #faf9f5;
  border: 1px solid var(--border);
  border-radius: 4px;
}

/* stats */
.stats { margin-top: 14px; font-size: 10px; line-height: 2.0; }
.stat-row { display: flex; justify-content: space-between; }
.stat-key { color: var(--muted); }
.stat-val { font-weight: 600; min-width: 60px; text-align: right; }

/* controls */
.controls { display: flex; gap: 8px; margin-top: 14px; }
button {
  font-family: inherit; font-size: 10px; letter-spacing: .06em;
  padding: 5px 12px; border: 1px solid var(--border);
  background: var(--surface); color: var(--text);
  border-radius: 3px; cursor: pointer;
}
button:hover { background: var(--border); }
button.on { background: var(--teal); color: #fff; border-color: var(--teal); }

/* plots */
.plot-block { margin-bottom: 18px; }
.plot-label {
  font-size: 9px; letter-spacing: .08em; text-transform: uppercase;
  color: var(--muted); margin-bottom: 5px;
}
svg.plot {
  display: block;
  background: #faf9f5;
  border: 1px solid var(--border);
  border-radius: 4px;
  overflow: visible;
}
.legend { display: flex; gap: 18px; margin-top: 5px; font-size: 9px; color: var(--muted); }
.ld { display: inline-block; width: 16px; height: 2px; border-radius: 1px; vertical-align: middle; margin-right: 4px; }
</style>
</head>
<body>

<header>
  <h1>Hill-Type Muscle · Hanging Mass Experiment</h1>
  <p>Thelen 2003 — F_max=__FMAX__N · m=__MASS__kg · mg=__MG__N · ƒ=__FREQ__Hz</p>
</header>

<div class="layout">
  <div class="left">
    <canvas id="scene" width="200" height="460"></canvas>
    <div class="stats">
      <div class="stat-row"><span class="stat-key">t</span>     <span class="stat-val" id="sv-t">0.000 s</span></div>
      <div class="stat-row"><span class="stat-key">u</span>     <span class="stat-val" id="sv-u">0.000</span></div>
      <div class="stat-row"><span class="stat-key">a</span>     <span class="stat-val" id="sv-a">0.000</span></div>
      <div class="stat-row"><span class="stat-key">F</span>     <span class="stat-val" id="sv-F">0.00 N</span></div>
      <div class="stat-row"><span class="stat-key">l_ce</span> <span class="stat-val" id="sv-lce">0.000 m</span></div>
      <div class="stat-row"><span class="stat-key">fl</span>   <span class="stat-val" id="sv-fl">0.000</span></div>
      <div class="stat-row"><span class="stat-key">fv</span>   <span class="stat-val" id="sv-fv">0.000</span></div>
    </div>
    <div class="controls">
      <button id="btn-pp" class="on">PAUSE</button>
      <button id="btn-rst">RESTART</button>
    </div>
  </div>

  <div class="right">
    <div class="plot-block">
      <div class="plot-label">Neural drive · Activation</div>
      <svg id="plot-a" class="plot" width="580" height="110" viewBox="0 0 580 110"></svg>
      <div class="legend">
        <span><span class="ld" style="background:#aaa;border-top:1px dashed #aaa"></span>u (neural drive)</span>
        <span><span class="ld" style="background:var(--teal)"></span>a (activation)</span>
      </div>
    </div>
    <div class="plot-block">
      <div class="plot-label">Muscle force  ·  mg reference</div>
      <svg id="plot-F" class="plot" width="580" height="110" viewBox="0 0 580 110"></svg>
      <div class="legend">
        <span><span class="ld" style="background:#5a8fd0"></span>F (N)</span>
        <span><span class="ld" style="background:#d07050;border-top:1px dashed #d07050"></span>mg</span>
      </div>
    </div>
    <div class="plot-block">
      <div class="plot-label">Box position  (m below ceiling)</div>
      <svg id="plot-y" class="plot" width="580" height="110" viewBox="0 0 580 110"></svg>
    </div>
  </div>
</div>

<script>
const FRAMES   = __DATA__;
const F_MAX    = __FMAX__;
const L_OPT   = __LOPT__;
const L_SLACK  = __LSLACK__;
const Y_FLOOR  = __YFLOOR__;
const MG       = __MG__;
const T_END    = __TEND__;

// ── Jet colormap ──────────────────────────────────────────────────────────────
function jet(t) {
  t = Math.max(0, Math.min(1, t));
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4*t - 1)));
  return [Math.round(255*r), Math.round(255*g), Math.round(255*b)];
}
function jetCSS(t) { const [r,g,b]=jet(t); return `rgb(${r},${g},${b})`; }

// ── Canvas scene ──────────────────────────────────────────────────────────────
const cv  = document.getElementById('scene');
const ctx = cv.getContext('2d');
const W = cv.width, H = cv.height;

const PAD_TOP = 36;
const VIS_M   = 0.62;                          // metres visible
const PPM     = (H - PAD_TOP - 20) / VIS_M;   // pixels per metre

function toY(m) { return PAD_TOP + m * PPM; }

function drawScene(fr) {
  ctx.clearRect(0, 0, W, H);
  const { y, a, l_ce, F } = fr;
  const cx = W / 2;

  // ── ceiling bar ──
  ctx.fillStyle = '#4a4a46';
  ctx.fillRect(0, PAD_TOP - 14, W, 14);
  // hatch
  ctx.save();
  ctx.strokeStyle = '#686864';
  ctx.lineWidth = 1.5;
  for (let xh = -8; xh < W + 8; xh += 14) {
    ctx.beginPath();
    ctx.moveTo(xh, PAD_TOP - 14);
    ctx.lineTo(xh + 10, PAD_TOP - 26);
    ctx.stroke();
  }
  ctx.restore();

  // positions
  const y_top_tendon = PAD_TOP;
  const l_tendon_px  = L_SLACK * PPM;
  const l_ce_px      = Math.max(2, l_ce * PPM);
  const y_box_top    = toY(y);

  const y_ce_top = y_top_tendon + l_tendon_px;
  const y_ce_bot = y_box_top    - l_tendon_px;
  const ceH      = Math.max(4, y_ce_bot - y_ce_top);

  // ── top tendon ──
  const tw = 9;
  ctx.fillStyle = '#b0a898';
  ctx.fillRect(cx - tw/2, y_top_tendon, tw, l_tendon_px);

  // ── CE cylinder (coloured by activation) ──
  const ceW = 42 + 14 * Math.min(1, F / F_MAX);  // slightly wider at high force
  const ceX = cx - ceW / 2;
  const [r,g,b] = jet(a);

  // lateral shading gradient for pseudo-3-D look
  const grd = ctx.createLinearGradient(ceX, 0, ceX + ceW, 0);
  grd.addColorStop(0,    `rgba(${r},${g},${b},0.6)`);
  grd.addColorStop(0.35, `rgba(${Math.min(255,r+55)},${Math.min(255,g+55)},${Math.min(255,b+55)},1)`);
  grd.addColorStop(1,    `rgba(${r},${g},${b},0.65)`);

  // pill shape: rounded rect
  const rr = Math.min(ceW / 2, 10);
  ctx.beginPath();
  ctx.roundRect(ceX, y_ce_top, ceW, ceH, rr);
  ctx.fillStyle = grd;
  ctx.shadowColor = jetCSS(a);
  ctx.shadowBlur  = 10;
  ctx.fill();
  ctx.shadowBlur  = 0;

  // border
  ctx.strokeStyle = `rgba(${Math.max(0,r-50)},${Math.max(0,g-50)},${Math.max(0,b-50)},0.7)`;
  ctx.lineWidth   = 1.2;
  ctx.stroke();

  // muscle-fibre striations
  if (ceH > 18) {
    ctx.save();
    ctx.clip();   // clip to pill shape already in path
    ctx.strokeStyle = 'rgba(255,255,255,0.13)';
    ctx.lineWidth   = 1;
    for (let yy = y_ce_top + 8; yy < y_ce_bot - 2; yy += 7) {
      ctx.beginPath();
      ctx.moveTo(ceX + 5,        yy);
      ctx.lineTo(ceX + ceW - 5,  yy);
      ctx.stroke();
    }
    ctx.restore();
    // re-path for later strokes
    ctx.beginPath();
    ctx.roundRect(ceX, y_ce_top, ceW, ceH, rr);
  }

  // ── bottom tendon ──
  ctx.fillStyle = '#b0a898';
  ctx.fillRect(cx - tw/2, y_box_top - l_tendon_px, tw, l_tendon_px);

  // ── box ──
  const bW = 68, bH = 38;
  const bX = cx - bW / 2;
  ctx.shadowColor = 'rgba(0,0,0,0.18)';
  ctx.shadowBlur  = 7;
  ctx.shadowOffsetY = 3;
  ctx.fillStyle = '#858078';
  ctx.beginPath();
  ctx.roundRect(bX, y_box_top, bW, bH, 4);
  ctx.fill();
  ctx.shadowBlur = 0;
  ctx.shadowOffsetY = 0;
  ctx.strokeStyle = '#5a5850';
  ctx.lineWidth = 1;
  ctx.stroke();
  // label
  ctx.fillStyle = '#f2f1ed';
  ctx.font = '600 10px JetBrains Mono, monospace';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText('1 kg', cx, y_box_top + bH / 2);

  // ── floor dashed line ──
  ctx.save();
  ctx.strokeStyle = '#c0bdb4';
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 4]);
  const yfl = toY(Y_FLOOR);
  ctx.beginPath(); ctx.moveTo(0, yfl); ctx.lineTo(W, yfl); ctx.stroke();
  ctx.restore();

  // ── activation colour bar (right edge) ──
  const cbX = W - 14, cbY0 = PAD_TOP + 16;
  const cbH = H - cbY0 - 24;
  for (let row = 0; row < cbH; row++) {
    ctx.fillStyle = jetCSS(1 - row / cbH);
    ctx.fillRect(cbX, cbY0 + row, 10, 1);
  }
  ctx.strokeStyle = '#c0bdb4';
  ctx.lineWidth = 0.5;
  ctx.strokeRect(cbX, cbY0, 10, cbH);
  // labels
  ctx.fillStyle = '#888';
  ctx.font = '9px JetBrains Mono, monospace';
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  ctx.fillText('1', cbX + 13, cbY0);
  ctx.textBaseline = 'bottom';
  ctx.fillText('0', cbX + 13, cbY0 + cbH);
  // indicator arrow
  const arrY = cbY0 + (1 - a) * cbH;
  ctx.fillStyle = '#333';
  ctx.beginPath();
  ctx.moveTo(cbX - 4, arrY);
  ctx.lineTo(cbX - 9, arrY - 4);
  ctx.lineTo(cbX - 9, arrY + 4);
  ctx.closePath();
  ctx.fill();
}

// ── SVG plot helpers ──────────────────────────────────────────────────────────
const PW = 580, PH = 110;
const PL = 38, PT = 8, PR = 12, PB = 18;
const PA_W = PW - PL - PR, PA_H = PH - PT - PB;
const N = FRAMES.length;

function px(i)          { return PL + (i / (N-1)) * PA_W; }
function py(v,mn,mx)    { return PT + (1 - (v-mn)/(mx-mn)) * PA_H; }
function dataPaths(vals, mn, mx) {
  return vals.map((v,i) => (i===0?'M':'L') + px(i).toFixed(1) + ',' + py(v,mn,mx).toFixed(1)).join(' ');
}
function svgEl(tag, attrs) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}
function addGrid(svg, mn, mx, ticks) {
  ticks.forEach(v => {
    const y = py(v, mn, mx);
    svg.appendChild(svgEl('line', {x1:PL, x2:PL+PA_W, y1:y, y2:y, stroke:'#d8d5cc', 'stroke-width':'0.5'}));
    const t = svgEl('text', {x:PL-4, y:y+3.5, 'text-anchor':'end', 'font-size':'8', fill:'#aaa'});
    t.textContent = v.toFixed(v<1?2:1);
    svg.appendChild(t);
  });
}
function addTimeTicks(svg, mn, mx) {
  [0, 2, 4, 6, 8, 10].filter(t=>t<=T_END).forEach(t => {
    const x = PL + (t / T_END) * PA_W;
    const tick = svgEl('text', {x, y:PT+PA_H+13, 'text-anchor':'middle', 'font-size':'8', fill:'#aaa'});
    tick.textContent = t + 's';
    svg.appendChild(tick);
  });
}
function makeCursor(svg) {
  const cur = svgEl('line', {x1:PL, x2:PL, y1:PT, y2:PT+PA_H, stroke:'#333', 'stroke-width':'1', opacity:'0.4'});
  svg.appendChild(cur);
  return cur;
}

function buildPlots() {
  // ── activation ──
  const svgA = document.getElementById('plot-a');
  addGrid(svgA, 0, 1, [0, 0.25, 0.5, 0.75, 1.0]);
  const puA = svgEl('path', {d: dataPaths(FRAMES.map(f=>f.u), 0, 1),
    fill:'none', stroke:'#aaa', 'stroke-width':'1', 'stroke-dasharray':'3,2'});
  const paA = svgEl('path', {d: dataPaths(FRAMES.map(f=>f.a), 0, 1),
    fill:'none', stroke:'#3aaa8f', 'stroke-width':'1.8'});
  svgA.appendChild(puA); svgA.appendChild(paA);
  window._curA = makeCursor(svgA);

  // ── force ──
  const svgF = document.getElementById('plot-F');
  const fMax = Math.max(...FRAMES.map(f=>f.F)) * 1.05;
  const fTicks = [0, MG, Math.round(fMax)];
  addGrid(svgF, 0, fMax, fTicks);
  // mg dashed reference
  const mgLine = svgEl('line', {x1:PL, x2:PL+PA_W, y1:py(MG,0,fMax), y2:py(MG,0,fMax),
    stroke:'#d07050', 'stroke-width':'1', 'stroke-dasharray':'4,3'});
  svgF.appendChild(mgLine);
  const pfF = svgEl('path', {d: dataPaths(FRAMES.map(f=>f.F), 0, fMax),
    fill:'none', stroke:'#5a8fd0', 'stroke-width':'1.8'});
  svgF.appendChild(pfF);
  window._curF = makeCursor(svgF);

  // ── position ──
  const svgY = document.getElementById('plot-y');
  const yVals = FRAMES.map(f=>f.y);
  const yMin = Math.min(...yVals) - 0.01, yMax = Y_FLOOR + 0.02;
  const yTick = [yMin, (yMin+yMax)/2, yMax].map(v=>Math.round(v*100)/100);
  addGrid(svgY, yMin, yMax, yTick);
  const pfY = svgEl('path', {d: dataPaths(yVals, yMin, yMax),
    fill:'none', stroke:'#8a6a3a', 'stroke-width':'1.8'});
  svgY.appendChild(pfY);
  addTimeTicks(svgY, yMin, yMax);
  window._curY = makeCursor(svgY);
}

// ── Animation loop ────────────────────────────────────────────────────────────
let fidx = 0, playing = true, lastWall = null;
const REALTIME = 1.0;

function tick(ts) {
  if (playing) {
    if (lastWall !== null) {
      const dtW = (ts - lastWall) / 1000;
      fidx = Math.min(N - 1, fidx + Math.round(dtW * REALTIME * 200));
    }
    lastWall = ts;
  }

  const fr = FRAMES[fidx];
  drawScene(fr);

  document.getElementById('sv-t').textContent   = fr.t.toFixed(3)  + ' s';
  document.getElementById('sv-u').textContent   = fr.u.toFixed(3);
  document.getElementById('sv-a').textContent   = fr.a.toFixed(3);
  document.getElementById('sv-F').textContent   = fr.F.toFixed(2)  + ' N';
  document.getElementById('sv-lce').textContent = fr.l_ce.toFixed(4) + ' m';
  document.getElementById('sv-fl').textContent  = fr.fl.toFixed(3);
  document.getElementById('sv-fv').textContent  = fr.fv.toFixed(3);

  const cx = px(fidx);
  [window._curA, window._curF, window._curY].forEach(c => {
    if (c) { c.setAttribute('x1', cx); c.setAttribute('x2', cx); }
  });

  if (fidx >= N - 1 && playing) { fidx = 0; lastWall = null; }
  requestAnimationFrame(tick);
}

document.getElementById('btn-pp').addEventListener('click', function() {
  playing = !playing; lastWall = null;
  this.textContent = playing ? 'PAUSE' : 'PLAY';
  this.classList.toggle('on', playing);
});
document.getElementById('btn-rst').addEventListener('click', () => { fidx = 0; lastWall = null; });

buildPlots();
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def write_html(frames: list[dict], out_path: Path) -> None:
    mg   = round(MASS * G, 3)
    data = json.dumps(frames)
    html = (_HTML_TEMPLATE
        .replace('__DATA__',   data)
        .replace('__FMAX__',   str(F_MAX))
        .replace('__MASS__',   str(MASS))
        .replace('__MG__',     str(mg))
        .replace('__FREQ__',   str(FREQ))
        .replace('__LOPT__',   str(L_OPT))
        .replace('__LSLACK__', str(L_SLACK))
        .replace('__YFLOOR__', str(Y_FLOOR))
        .replace('__TEND__',   str(T_END))
    )
    out_path.write_text(html, encoding='utf-8')
    print(f"  Written → {out_path}  ({len(html)//1024} KB)")


if __name__ == '__main__':
    print("Simulating hanging mass …")
    frames = simulate()
    print(f"  {len(frames)} frames recorded ({frames[-1]['t']:.1f} s)")

    out = Path(__file__).parent / 'muscle_demo.html'
    write_html(frames, out)
    print("Done.\n  Open muscle_demo.html in your browser.")
