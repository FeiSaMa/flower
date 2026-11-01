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
  html, body { margin: 0; width: 1280px; height: 720px; overflow: hidden; background: #ffffff; }
  body { font-family: "Source Han Sans CN", "Noto Sans CJK SC", sans-serif; color: #262626; }
  #app { position: relative; width: 1280px; height: 720px; }
  header {
    height: 70px; background: #1F3864; color: white; display: flex;
    align-items: center; justify-content: space-between; padding: 0 28px;
    box-sizing: border-box;
  }
  header h1 { font-size: 26px; margin: 0; font-weight: 700; letter-spacing: 1px; }
  #stage { font-size: 17px; color: #F4B183; font-weight: 700; }
  #canvas { display: block; width: 1280px; height: 560px; }
  footer {
    height: 90px; box-sizing: border-box; padding: 12px 28px;
    background: #F2F2F2; display: flex; align-items: center; gap: 16px;
  }
  #play {
    width: 86px; height: 34px; border: 0; border-radius: 6px;
    background: #C55A11; color: white; font-size: 15px; font-weight: 700;
    cursor: pointer;
  }
  #seek { flex: 1; accent-color: #C55A11; height: 6px; }
  #time { width: 78px; color: #595959; font-size: 14px; text-align: right; }
  #tip { color: #595959; font-size: 13px; width: 300px; }
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
    <div id="time">0.0 s</div>
    <div id="tip">拖动时间轴可回看任意时刻</div>
  </footer>
</div>
<script>
const DATA = __DATA__;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const W = 1280, H = 560;
const NAVY = '#1F3864', ORANGE = '#C55A11', GRAY = '#595959';
const LIGHT = '#F2F2F2', INK = '#262626', GREEN = '#2E7D32', RED = '#C62828';
const BATCH = DATA.batch.filter(item => item.epoch === 1);
const IMAGES = {};
let loaded = 0;
let currentTime = 0;
let playing = true;
let lastTimestamp = null;
window.__ready = false;

function font(size, bold) {
  return (bold ? 'bold ' : '') + size + 'px "Source Han Sans CN", "Noto Sans CJK SC", sans-serif';
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
    ctx.shadowColor = 'rgba(197,90,17,0.35)';
    ctx.shadowBlur = 16;
  }
  roundRect(x, y, w, h, 12);
  ctx.fillStyle = fill;
  ctx.fill();
  ctx.lineWidth = active ? 3 : 1.5;
  ctx.strokeStyle = active ? ORANGE : stroke;
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
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, W, H);
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
  const gx = 38, gy = 78, gw = 690, gh = 430;
  const dx = gw / cols, dy = gh / rows;
  const b = Math.min(28, Math.floor(t / 0.28) + 1);
  const local = Math.max(0, Math.min(1, (t - (b - 1) * 0.28) / 0.28));
  const entry = BATCH[Math.min(b, 28) - 1];
  text('1734 张训练图', gx, 52, 18, NAVY, 'left', true);
  text('每批 64 张：亮起 → 送入模型', gx, 535, 14, GRAY);

  box(830, 145, 400, 270, '#F8FAFD', NAVY, true);
  text('模型', 1030, 190, 25, NAVY, 'center', true);
  text('ResNet18 + fc(102)', 1030, 220, 15, GRAY, 'center');
  text('epoch 1 · batch ' + b + '/28', 1030, 270, 18, INK, 'center', true);
  text('batch loss', 1030, 306, 14, GRAY, 'center');
  text(entry.loss.toFixed(2), 1030, 350, 30, ORANGE, 'center', true);

  for (let i = 0; i < N; i += 1) {
    const col = i % cols, row = Math.floor(i / cols);
    let x = gx + col * dx + dx / 2;
    let y = gy + row * dy + dy / 2;
    let color = '#D9D9D9', radius = 2.2;
    if (i < (b - 1) * batchSize) color = '#9DB2CE';
    if (i >= (b - 1) * batchSize && i < b * batchSize) {
      const j = i - (b - 1) * batchSize;
      const tx = 850 + (j % 8) * 7;
      const ty = 205 + Math.floor(j / 8) * 17;
      x += (tx - x) * local;
      y += (ty - y) * local;
      color = ORANGE;
      radius = 3.0;
    }
    ctx.beginPath();
    ctx.fillStyle = color;
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  }
  if (local < 1) {
    arrow(gx + gw + 12, 290, 820, 290, 'rgba(197,90,17,0.7)');
  }
  text('1 个 batch = 64 张图', 1030, 465, 15, GRAY, 'center');
}

function drawNetwork(t) {
  const p = Math.max(0, Math.min(1, (t - 8) / 7));
  const step = Math.min(4, Math.floor(p * 5));
  const labels = [
    ['64 张图', '(64,3,224,224)', '输入'],
    ['ResNet18 骨干', '(64,512)', '11.2M 参数'],
    ['fc', '(64,102)', '52,326 可训'],
    ['softmax', '(64,102)', '概率'],
    ['交叉熵', 'loss 标量', '一个数'],
  ];
  const xs = [230, 470, 710, 950, 1130];
  const widths = [180, 190, 160, 180, 130];
  text('一个 batch 穿过网络', 40, 52, 20, NAVY, 'left', true);
  text('形状链来自 demo/shapes.json', 40, 535, 14, GRAY);
  const tile = 22;
  for (let i = 0; i < 64; i += 1) {
    const imageId = DATA.sampleIds[i % DATA.sampleIds.length];
    const image = IMAGES[imageId];
    const x = 40 + (i % 8) * tile;
    const y = 115 + Math.floor(i / 8) * tile;
    if (image && image.complete) {
      ctx.drawImage(image, x, y, tile - 2, tile - 2);
    } else {
      ctx.fillStyle = '#D9D9D9';
      ctx.fillRect(x, y, tile - 2, tile - 2);
    }
  }
  text('16 张缩略图循环排列', 40, 315, 13, GRAY);
  for (let i = 0; i < labels.length; i += 1) {
    const active = i === step;
    box(xs[i], 165, widths[i], 125, active ? '#FFF4EC' : LIGHT, active ? ORANGE : '#B7C4D6', active);
    text(labels[i][0], xs[i] + widths[i] / 2, 205, 17, active ? ORANGE : NAVY, 'center', true);
    text(labels[i][1], xs[i] + widths[i] / 2, 235, 14, GRAY, 'center');
    text(labels[i][2], xs[i] + widths[i] / 2, 265, 13, GRAY, 'center');
    if (i < labels.length - 1) arrow(xs[i] + widths[i] + 4, 228, xs[i + 1] - 4, 228);
  }
  text('前向：图片 → 特征 → 分数 → 概率 → loss；然后反向传播更新 fc', 640, 390, 18, INK, 'center');
  text('第 ' + (step + 1) + ' / 5 步', 640, 430, 15, ORANGE, 'center', true);
}

function drawLoss(t) {
  const p = Math.max(0, Math.min(1, (t - 15) / 8));
  const n = Math.max(1, Math.min(DATA.batch.length, Math.floor(p * DATA.batch.length) + 1));
  const x0 = 85, x1 = 1220, y0 = 470, y1 = 85;
  const maxLoss = Math.max(...DATA.batch.map(item => item.loss)) + 0.2;
  const x = index => x0 + (x1 - x0) * index / (DATA.batch.length - 1);
  const y = value => y0 - (y0 - y1) * value / maxLoss;
  const colors = ['#1F3864', '#C55A11', '#595959', '#8C564B', '#2CA02C'];
  text('逐 batch loss：每个点 = 一次参数更新', 40, 52, 20, NAVY, 'left', true);
  ctx.strokeStyle = '#D9D9D9';
  ctx.lineWidth = 1;
  for (let g = 0; g <= 5; g += 1) {
    const yy = y0 - (y0 - y1) * g / 5;
    ctx.beginPath(); ctx.moveTo(x0, yy); ctx.lineTo(x1, yy); ctx.stroke();
    text((maxLoss * g / 5).toFixed(1), x0 - 12, yy + 4, 11, GRAY, 'right');
  }
  for (let index = 1; index < n; index += 1) {
    const current = DATA.batch[index];
    ctx.strokeStyle = colors[current.epoch - 1];
    ctx.lineWidth = 1.8;
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
  ctx.fillStyle = ORANGE;
  ctx.beginPath(); ctx.arc(x(n - 1), y(current.loss), 6, 0, Math.PI * 2); ctx.fill();
  text('当前 loss ' + current.loss.toFixed(2), x(n - 1) + 12, y(current.loss) - 10, 16, ORANGE, 'left', true);
  text('batch 序号', (x0 + x1) / 2, 520, 14, GRAY, 'center');
  for (const index of [1, 28, 56, 84, 112, 140]) {
    text(String(index), x(index - 1), 495, 10, GRAY, 'center');
  }
  for (let epoch = 1; epoch <= 5; epoch += 1) {
    text('epoch ' + epoch, x0 + (epoch - 1) * 60, 78, 11, colors[epoch - 1]);
  }
}

function drawPredictions(t) {
  const elapsed = Math.max(0, t - 23);
  let epoch = Math.min(4, Math.floor(elapsed / 1.4));
  if (t >= 29.0) epoch = 5;
  const accuracy = predictionAccuracy(epoch);
  text('同一批图：预测如何从随机变成认对', 40, 48, 20, NAVY, 'left', true);
  text('ep' + epoch, 1180, 48, 22, ORANGE, 'right', true);
  text('16 张 top-1 准确率 ' + oneDecimal(accuracy * 100) + '%', 1180, 78, 16, NAVY, 'right', true);
  const cols = 8, cellW = 150, cellH = 215, x0 = 32, y0 = 95;
  for (let index = 0; index < DATA.sampleIds.length; index += 1) {
    const imageId = DATA.sampleIds[index];
    const entry = DATA.preds[String(epoch)][imageId];
    const top = entry.top5[0];
    const correct = top.class_idx === entry.true_idx;
    const col = index % cols, row = Math.floor(index / cols);
    const cx = x0 + col * cellW, cy = y0 + row * cellH;
    const image = IMAGES[imageId];
    if (image && image.complete) {
      const scale = Math.min(120 / image.width, 90 / image.height);
      const iw = image.width * scale, ih = image.height * scale;
      ctx.save();
      roundRect(cx + (cellW - iw) / 2, cy, iw, ih, 5);
      ctx.clip();
      ctx.drawImage(image, cx + (cellW - iw) / 2, cy, iw, ih);
      ctx.restore();
    } else {
      ctx.fillStyle = '#D9D9D9';
      ctx.fillRect(cx + 15, cy, 120, 90);
    }
    const predName = shortName(DATA.names[String(top.class_idx)] || ('class ' + top.class_idx), 13);
    const trueName = shortName(DATA.names[String(entry.true_idx)] || ('class ' + entry.true_idx), 13);
    text(predName + ' ' + (top.prob * 100).toFixed(1) + '%', cx + cellW / 2, cy + 108, 11, correct ? GREEN : RED, 'center', true);
    text('真实 ' + trueName, cx + cellW / 2, cy + 124, 9, GRAY, 'center');
    ctx.fillStyle = '#E6E6E6';
    roundRect(cx + 14, cy + 138, 110, 7, 3); ctx.fill();
    ctx.fillStyle = correct ? GREEN : RED;
    roundRect(cx + 14, cy + 138, Math.max(3, 110 * top.prob), 7, 3); ctx.fill();
  }
}

function draw(t) {
  clear();
  if (t < 8) drawBatch(t);
  else if (t < 15) drawNetwork(t);
  else if (t < 23) drawLoss(t);
  else drawPredictions(t);
}

function stageName(t) {
  if (t < 8) return '分批';
  if (t < 15) return '一个 batch 穿过网络';
  if (t < 23) return 'loss 曲线';
  return '预测演化';
}

function setTime(t) {
  currentTime = Math.max(0, Math.min(30, Number(t)));
  draw(currentTime);
  document.getElementById('seek').value = String(currentTime);
  document.getElementById('time').textContent = currentTime.toFixed(1) + ' s';
  document.getElementById('stage').textContent = stageName(currentTime);
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
  setTime(next);
  if (playing) requestAnimationFrame(tick);
}

function togglePlay() {
  if (playing) {
    playing = false;
    document.getElementById('play').textContent = '播放';
    return;
  }
  if (currentTime >= 30) setTime(0);
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
setTime(0);
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
