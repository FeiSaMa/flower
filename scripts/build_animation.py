"""生成自包含的 30 秒训练过程动画 HTML。

数据来自 demo/batch_log.csv、demo/pred_ep0..5.json、demo/sample_ids.json、
data/class_names.csv 与 data/raw/jpg 的 16 张缩略图。HTML 不依赖外部资源。
"""

import base64
import csv
import io
import json
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "report" / "training_animation.html"
DATA_FILE = ROOT / "demo" / "batch_log.csv"
SAMPLE_FILE = ROOT / "demo" / "sample_ids.json"
PRED_FILES = [ROOT / "demo" / f"pred_ep{epoch}.json" for epoch in range(6)]
NAMES_FILE = ROOT / "data" / "class_names.csv"
IMAGE_DIR = ROOT / "data" / "raw" / "jpg"


def load_batch_log():
    rows = []
    with DATA_FILE.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "batch_idx": int(row["batch_idx"]),
                    "loss": float(row["loss"]),
                    "batch_acc": float(row["batch_acc"]),
                }
            )
    return rows


def load_predictions():
    predictions = {}
    for epoch, path in enumerate(PRED_FILES):
        with path.open(encoding="utf-8") as handle:
            predictions[str(epoch)] = json.load(handle)
    return predictions


def load_names():
    with NAMES_FILE.open(encoding="utf-8") as handle:
        return {str(int(row["class_idx"])): row["name"] for row in csv.DictReader(handle)}


def load_images(sample_ids):
    images = {}
    for image_id in sample_ids:
        image = Image.open(IMAGE_DIR / image_id).convert("RGB")
        image.thumbnail((128, 128))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=70, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        images[image_id] = f"data:image/jpeg;base64,{encoded}"
    return images


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<title>一次训练到底发生了什么</title>
<style>
  html, body { margin: 0; width: 1280px; height: 720px; overflow: hidden; background: #020617; }
  body {
    font-family: "JetBrains Mono", "Noto Sans Mono CJK SC", "Source Han Sans CN", ui-monospace,
      SFMono-Regular, Menlo, Consolas, monospace;
    color: #FFFFFF;
  }
  #app { position: relative; width: 1280px; height: 720px; background: #020617; }
  header {
    height: 64px; background: #020617; border-bottom: 1px solid #1E293B;
    display: flex; align-items: center; justify-content: space-between; padding: 0 24px;
    box-sizing: border-box;
  }
  header h1 {
    font-size: 1.5rem; line-height: 1; margin: 0; font-weight: 700;
    letter-spacing: -0.025em; color: #FFFFFF;
  }
  #stage {
    font-size: 0.625rem; font-weight: 700; letter-spacing: 0.12em;
    text-transform: uppercase; color: #FB923C;
  }
  #canvas { display: block; width: 1280px; height: 560px; background: #020617; }
  footer {
    height: 96px; box-sizing: border-box; padding: 0 24px; background: #020617;
    border-top: 1px solid #1E293B; display: flex; align-items: center; gap: 16px;
  }
  #play {
    width: 96px; height: 36px; border: 1px solid #1E293B; border-radius: 0.5rem;
    background: #0F172A; color: #FFFFFF; font: 600 0.75rem/1 inherit; cursor: pointer;
    letter-spacing: 0.04em;
  }
  #seek {
    flex: 1; height: 4px; appearance: none; border-radius: 2px;
    background: linear-gradient(to right, #FB923C 0%, #1E293B 0%);
    outline: none;
  }
  #seek::-webkit-slider-thumb {
    appearance: none; width: 12px; height: 12px; border-radius: 50%;
    background: #FB923C; border: 1px solid #020617;
  }
  #time { width: 130px; color: #94A3B8; font-size: 0.75rem; text-align: right; font-variant-numeric: tabular-nums; }
  #tip { color: #475569; font-size: 0.625rem; width: 220px; letter-spacing: 0.02em; }
</style>
</head>
<body>
<div id="app">
  <header>
    <h1>一次训练到底发生了什么</h1>
    <div id="stage">分批</div>
  </header>
  <canvas id="canvas" width="1280" height="560"></canvas>
  <footer>
    <button id="play">暂停</button>
    <input id="seek" type="range" min="0" max="30" step="0.01" value="0">
    <div id="time">0.0s / 30.0s</div>
    <div id="tip">拖动时间轴可回看</div>
  </footer>
</div>
<script>
const DATA = __DATA__;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const W = 1280, H = 560;
const BG = '#020617', PANEL = '#0F172A', BORDER = '#1E293B';
const INK = '#FFFFFF', MUTED = '#94A3B8', DIM = '#475569';
const CYAN = '#22D3EE', GREEN = '#34D399', VIOLET = '#A78BFA';
const AMBER = '#FBBF24', ROSE = '#FB7185', ORANGE = '#FB923C';
const NAVY = INK, GRAY = MUTED, LIGHT = PANEL, RED = ROSE;
const BATCH = DATA.batch.filter(item => item.epoch === 1);
const IMAGES = {};
let loaded = 0;
let currentTime = 0;
let playing = true;
let lastTimestamp = null;
window.__ready = false;

function font(size, bold) {
  return (bold ? 'bold ' : '') + size + 'px "JetBrains Mono", "Noto Sans Mono CJK SC", "Source Han Sans CN", ui-monospace, monospace';
}
function text(value, x, y, size = 16, color = INK, align = 'left', bold = false) {
  ctx.fillStyle = color;
  ctx.font = font(size, bold);
  ctx.textAlign = align;
  ctx.textBaseline = 'alphabetic';
  ctx.fillText(value, x, y);
}
function roundRect(x, y, w, h, r = 10) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}
function box(x, y, w, h, fill, stroke, active = false) {
  ctx.save();
  if (active) {
    ctx.shadowColor = stroke;
    ctx.shadowBlur = 7;
  }
  roundRect(x, y, w, h, 8);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = active ? stroke : BORDER;
  ctx.stroke();
  ctx.restore();
}
function arrow(x1, y1, x2, y2, color = ORANGE) {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(x1, y1);
  ctx.lineTo(x2, y2);
  ctx.stroke();
  const angle = Math.atan2(y2 - y1, x2 - x1);
  ctx.beginPath();
  ctx.moveTo(x2, y2);
  ctx.lineTo(x2 - 9 * Math.cos(angle - 0.45), y2 - 9 * Math.sin(angle - 0.45));
  ctx.lineTo(x2 - 9 * Math.cos(angle + 0.45), y2 - 9 * Math.sin(angle + 0.45));
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}
function clear() {
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);
  roundRect(16, 8, 1248, 544, 16);
  ctx.fillStyle = PANEL;
  ctx.fill();
  ctx.lineWidth = 1;
  ctx.strokeStyle = BORDER;
  ctx.stroke();
}
function shortName(name, limit) {
  return name.length <= limit ? name : name.slice(0, limit - 1) + '…';
}
function oneDecimal(value) {
  const scaled = value * 10;
  const floor = Math.floor(scaled);
  const fraction = scaled - floor;
  const rounded = Math.abs(fraction - 0.5) < 1e-9
    ? (floor % 2 === 0 ? floor : floor + 1)
    : Math.round(scaled);
  return (rounded / 10).toFixed(1);
}
function predictionAccuracy(epoch) {
  let hits = 0;
  for (const imageId of DATA.sampleIds) {
    const entry = DATA.preds[String(epoch)][imageId];
    if (entry.top5[0].class_idx === entry.true_idx) hits += 1;
  }
  return hits / DATA.sampleIds.length;
}

function drawBatch(t) {
  const N = 1734, cols = 34, rows = 51, batchSize = 64;
  const gx = 44, gy = 92, gw = 690, gh = 418;
  const dx = gw / cols, dy = gh / rows;
  const b = Math.min(28, Math.floor(t / 0.28) + 1);
  const local = Math.max(0, Math.min(1, (t - (b - 1) * 0.28) / 0.28));
  const entry = BATCH[Math.min(b, 28) - 1];
  text('TRAIN IMAGES / 1734', gx, 58, 10, MUTED, 'left', true);
  text('64 images = 1 batch · 28 batches = 1 epoch', gx, 532, 10, DIM, 'left');

  box(826, 150, 406, 270, PANEL, GREEN, true);
  text('MODEL / BACKBONE', 1029, 186, 10, GREEN, 'center', true);
  text('ResNet18 + fc(102)', 1029, 210, 12, MUTED, 'center');
  text('EPOCH 1 · BATCH ' + b + '/28', 1029, 255, 14, INK, 'center', true);
  text('LOSS', 1029, 300, 10, MUTED, 'center', true);
  text(entry.loss.toFixed(2), 1029, 342, 30, ROSE, 'center', true);

  for (let i = 0; i < N; i += 1) {
    const col = i % cols, row = Math.floor(i / cols);
    let x = gx + col * dx + dx / 2;
    let y = gy + row * dy + dy / 2;
    let color = BORDER, radius = 2.2;
    if (i < (b - 1) * batchSize) color = DIM;
    if (i >= (b - 1) * batchSize && i < b * batchSize) {
      const j = i - (b - 1) * batchSize;
      const tx = 846 + (j % 8) * 7;
      const ty = 214 + Math.floor(j / 8) * 17;
      x += (tx - x) * local;
      y += (ty - y) * local;
      color = CYAN;
      radius = 3.0;
    }
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  if (local < 1) {
    arrow(gx + gw + 12, 292, 816, 292, 'rgba(251,146,60,0.65)');
  }
  text('1 batch = 64 images', 1029, 458, 10, MUTED, 'center', true);
}

function drawNetwork(t) {
  const p = Math.max(0, Math.min(1, (t - 8) / 7));
  const step = Math.min(4, Math.floor(p * 5));
  const labels = [
    ['INPUT', '(64,3,224,224)', '64 images'],
    ['BACKBONE', '(64,512)', '11.2M params'],
    ['FC', '(64,102)', '52,326 trainable'],
    ['SOFTMAX', '(64,102)', 'probabilities'],
    ['LOSS', 'scalar', 'cross-entropy'],
  ];
  const colors = [CYAN, GREEN, AMBER, CYAN, ROSE];
  const xs = [248, 460, 680, 860, 1072];
  const widths = [150, 160, 120, 150, 160];
  text('FORWARD PASS / ONE BATCH', 44, 58, 10, MUTED, 'left', true);
  text('shapes.json', 44, 532, 10, VIOLET, 'left', true);
  const tile = 28;
  for (let i = 0; i < 16; i += 1) {
    const imageId = DATA.sampleIds[i % DATA.sampleIds.length];
    const image = IMAGES[imageId];
    const x = 44 + (i % 4) * tile;
    const y = 118 + Math.floor(i / 4) * tile;
    if (image && image.complete) {
      ctx.drawImage(image, x, y, tile - 2, tile - 2);
    } else {
      ctx.fillStyle = BORDER;
      ctx.fillRect(x, y, tile - 2, tile - 2);
    }
  }
  text('16 thumbnails', 44, 258, 10, DIM, 'left');
  arrow(162, 206, 240, 206);
  for (let i = 0; i < labels.length; i += 1) {
    const active = i === step;
    box(xs[i], 165, widths[i], 125, PANEL, colors[i], active);
    text(labels[i][0], xs[i] + widths[i] / 2, 202, 10, colors[i], 'center', true);
    text(labels[i][1], xs[i] + widths[i] / 2, 230, 12, INK, 'center');
    text(labels[i][2], xs[i] + widths[i] / 2, 258, 10, DIM, 'center');
    if (i < labels.length - 1) arrow(xs[i] + widths[i] + 6, 228, xs[i + 1] - 6, 228);
  }
  text('forward → backward → update fc', 640, 384, 12, MUTED, 'center');
  text('STEP ' + (step + 1) + ' / 5', 640, 416, 10, ORANGE, 'center', true);
}

function drawLoss(t) {
  const p = Math.max(0, Math.min(1, (t - 15) / 8));
  const n = Math.max(1, Math.min(DATA.batch.length, Math.floor(p * DATA.batch.length) + 1));
  const x0 = 85, x1 = 1220, y0 = 470, y1 = 85;
  const maxLoss = Math.max(...DATA.batch.map(item => item.loss)) + 0.2;
  const x = index => x0 + (x1 - x0) * index / (DATA.batch.length - 1);
  const y = value => y0 - (y0 - y1) * value / maxLoss;
  const colors = [ROSE, ROSE, ROSE, ROSE, ROSE];
  text('LOSS CURVE / 140 UPDATES', 44, 58, 10, MUTED, 'left', true);
  ctx.strokeStyle = BORDER;
  ctx.lineWidth = 1;
  for (let g = 0; g <= 5; g += 1) {
    const yy = y0 - (y0 - y1) * g / 5;
    ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x1, yy); ctx.stroke();
    text((maxLoss * g / 5).toFixed(1), x0 - 12, yy + 4, 11, GRAY, 'right');
  }
  for (let index = 1; index < n; index += 1) {
    const current = DATA.batch[index];
    ctx.strokeStyle = colors[current.epoch - 1];
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(x(index - 1), y(DATA.batch[index - 1].loss));
    ctx.lineTo(x(index), y(current.loss));
    ctx.stroke();
  }
  for (let index = 0; index < n; index += 1) {
    const item = DATA.batch[index];
    ctx.fillStyle = colors[item.epoch - 1];
    ctx.beginPath();
    ctx.arc(x(index), y(item.loss), 2.2, 0, Math.PI * 2);
    ctx.fill();
  }
  const current = DATA.batch[n - 1];
  ctx.fillStyle = ROSE;
  ctx.beginPath(); ctx.arc(x(n - 1), y(current.loss), 5, 0, Math.PI * 2); ctx.fill();
  text('LOSS ' + current.loss.toFixed(2), x(n - 1) + 12, y(current.loss) - 10, 12, ROSE, 'left', true);
  text('BATCH INDEX', (x0 + x1) / 2, 520, 10, MUTED, 'center', true);
  for (const index of [1, 28, 56, 84, 112, 140]) {
    text(String(index), x(index - 1), 495, 10, GRAY, 'center');
  }
  for (let epoch = 1; epoch <= 5; epoch += 1) {
    text('EPOCH ' + epoch, x0 + (epoch - 1) * 72, 78, 10, DIM, 'left', true);
  }
}

function drawPredictions(t) {
  const elapsed = Math.max(0, t - 23);
  let epoch = Math.min(4, Math.floor(elapsed / 1.4));
  if (t >= 29.0) epoch = 5;
  const accuracy = predictionAccuracy(epoch);
  text('PREDICTION EVOLUTION', 44, 58, 10, MUTED, 'left', true);
  text('同一批图：预测如何从随机变成认对', 44, 82, 14, INK, 'left', true);
  text('EP' + epoch, 1180, 58, 10, CYAN, 'right', true);
  text('TOP-1 ACCURACY', 1180, 82, 10, MUTED, 'right', true);
  text(oneDecimal(accuracy * 100) + '%', 1180, 112, 24, INK, 'right', true);
  const cols = 8, cellW = 150, cellH = 212, x0 = 32, y0 = 120;
  for (let index = 0; index < DATA.sampleIds.length; index += 1) {
    const imageId = DATA.sampleIds[index];
    const entry = DATA.preds[String(epoch)][imageId];
    const top = entry.top5[0];
    const correct = top.class_idx === entry.true_idx;
    const col = index % cols, row = Math.floor(index / cols);
    const cx = x0 + col * cellW, cy = y0 + row * cellH;
    const image = IMAGES[imageId];
    ctx.strokeStyle = BORDER;
    ctx.lineWidth = 1;
    roundRect(cx + 4, cy - 4, cellW - 8, 180, 8);
    ctx.stroke();
    if (image && image.complete) {
      const scale = Math.min(120 / image.width, 88 / image.height);
      const iw = image.width * scale, ih = image.height * scale;
      ctx.save();
      roundRect(cx + (cellW - iw) / 2, cy, iw, ih, 3);
      ctx.clip();
      ctx.drawImage(image, cx + (cellW - iw) / 2, cy, iw, ih);
      ctx.restore();
    } else {
      ctx.fillStyle = BORDER;
      ctx.fillRect(cx + 15, cy, 120, 88);
    }
    const predName = shortName(DATA.names[String(top.class_idx)] || ('class ' + top.class_idx), 13);
    const trueName = shortName(DATA.names[String(entry.true_idx)] || ('class ' + entry.true_idx), 13);
    text(predName + ' ' + oneDecimal(top.prob * 100) + '%', cx + cellW / 2, cy + 108, 10, correct ? GREEN : ROSE, 'center', true);
    text('TRUE ' + trueName, cx + cellW / 2, cy + 123, 8, MUTED, 'center');
    ctx.fillStyle = BORDER;
    roundRect(cx + 14, cy + 137, 110, 5, 2); ctx.fill();
    ctx.fillStyle = correct ? GREEN : ROSE;
    roundRect(cx + 14, cy + 137, Math.max(3, 110 * top.prob), 5, 2); ctx.fill();
  }
}

function draw(t) {
  clear();
  const start = t < 8 ? 0 : (t < 15 ? 8 : (t < 23 ? 15 : 23));
  const alpha = start === 0 ? 1 : Math.min(1, (t - start) / 0.18);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.translate(0, 4 * (1 - alpha));
  if (t < 8) drawBatch(t);
  else if (t < 15) drawNetwork(t);
  else if (t < 23) drawLoss(t);
  else drawPredictions(t);
  ctx.restore();
}

function stageName(t) {
  if (t < 8) return '01 · BATCHING';
  if (t < 15) return '02 · FORWARD';
  if (t < 23) return '03 · LOSS';
  return '04 · PREDICTION';
}

function renderTime(t) {
  currentTime = Math.max(0, Math.min(30, Number(t)));
  draw(currentTime);
  const seek = document.getElementById('seek');
  seek.value = String(currentTime);
  seek.style.background = 'linear-gradient(to right, #FB923C ' + (currentTime / 30 * 100) + '%, #1E293B ' + (currentTime / 30 * 100) + '%)';
  document.getElementById('time').textContent = currentTime.toFixed(1) + 's / 30.0s';
  document.getElementById('stage').textContent = stageName(currentTime);
}
function setTime(t) {
  playing = false;
  renderTime(t);
}
window.setTime = setTime;

function tick(timestamp) {
  if (!playing) return;
  if (lastTimestamp === null) lastTimestamp = timestamp;
  const dt = (timestamp - lastTimestamp) / 1000;
  lastTimestamp = timestamp;
  let next = currentTime + dt;
  if (next >= 30) {
    next = 30;
    playing = false;
    document.getElementById('play').textContent = '播放';
  }
  renderTime(next);
  if (playing) requestAnimationFrame(tick);
}

function togglePlay() {
  if (playing) {
    playing = false;
    document.getElementById('play').textContent = '播放';
    return;
  }
  if (currentTime >= 30) renderTime(0);
  playing = true;
  lastTimestamp = null;
  document.getElementById('play').textContent = '暂停';
  requestAnimationFrame(tick);
}

document.getElementById('play').addEventListener('click', togglePlay);
document.getElementById('seek').addEventListener('input', event => {
  playing = false;
  document.getElementById('play').textContent = '播放';
  setTime(parseFloat(event.target.value));
});

for (const imageId of DATA.sampleIds) {
  const image = new Image();
  image.onload = () => {
    loaded += 1;
    if (loaded === DATA.sampleIds.length) {
      window.__ready = true;
      draw(currentTime);
    }
  };
  image.src = DATA.images[imageId];
  IMAGES[imageId] = image;
}
renderTime(0);
if (window.__ready !== true) {
  window.__ready = loaded === DATA.sampleIds.length;
}
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


def main():
    batch = load_batch_log()
    predictions = load_predictions()
    sample_ids = json.loads(SAMPLE_FILE.read_text(encoding="utf-8"))
    if len(sample_ids) != 16:
        raise SystemExit(f"sample_ids 应为 16 个，实际 {len(sample_ids)}")
    payload = {
        "batch": batch,
        "preds": predictions,
        "sampleIds": sample_ids,
        "images": load_images(sample_ids),
        "names": load_names(),
    }
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(f"size {OUTPUT.stat().st_size} bytes")
    print(f"embedded images {len(payload['images'])}")
    print(f"batch rows {len(batch)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
