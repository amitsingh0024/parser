import json
import os

log_file = os.path.join("outputs", "unlimited-ocr2-indic-lora", "training_progress.jsonl")
records = []
with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass

# Clean and extract sequential points
# Calculate moving average (window = 10)
points = []
window = 10
rolling_window = []

for idx, r in enumerate(records):
    loss = float(r.get("loss", 0.0))
    rolling_window.append(loss)
    if len(rolling_window) > window:
        rolling_window.pop(0)
    avg_loss = sum(rolling_window) / len(rolling_window)
    points.append({
        "index": idx + 1,
        "step": int(r.get("step", idx + 1)),
        "epoch": round(float(r.get("epoch", 0.0)), 2),
        "loss": round(loss, 4),
        "ma_loss": round(avg_loss, 4),
        "lr": float(r.get("learning_rate", 0.0)),
        "grad_norm": round(float(r.get("grad_norm", 0.0)), 2),
    })

# Compute summary stats
losses = [p["loss"] for p in points]
min_loss = min(losses)
max_loss = max(losses)
first_loss = losses[0]
recent_10_avg = sum(losses[-10:]) / 10
total_reduction = ((first_loss - recent_10_avg) / first_loss) * 100

data_json = json.dumps(points)

# Generate an SVG file
svg_w, svg_h = 900, 380
pad_l, pad_r, pad_t, pad_b = 60, 40, 40, 40
plot_w = svg_w - pad_l - pad_r
plot_h = svg_h - pad_t - pad_b

# Scaling functions
max_y = 7.0
min_y = 0.0

def scale_x(idx, total):
    return pad_l + (idx / max(1, total - 1)) * plot_w

def scale_y(val):
    clamped = max(min_y, min(max_y, val))
    return pad_t + (1.0 - (clamped - min_y) / (max_y - min_y)) * plot_h

# Build paths
raw_path = []
ma_path = []
for i, p in enumerate(points):
    x = scale_x(i, len(points))
    y_raw = scale_y(p["loss"])
    y_ma = scale_y(p["ma_loss"])
    raw_path.append(f"{'M' if i == 0 else 'L'}{x:.1f},{y_raw:.1f}")
    ma_path.append(f"{'M' if i == 0 else 'L'}{x:.1f},{y_ma:.1f}")

raw_d = " ".join(raw_path)
ma_d = " ".join(ma_path)

# Area fill for MA
ma_area = f"{ma_d} L{scale_x(len(points)-1, len(points)):.1f},{scale_y(0.0):.1f} L{scale_x(0, len(points)):.1f},{scale_y(0.0):.1f} Z"

svg_content = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {svg_w} {svg_h}" width="100%" height="100%" style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: #0f172a; border-radius: 12px;">
  <!-- Grid Lines -->
  <line x1="{pad_l}" y1="{scale_y(6.0)}" x2="{svg_w - pad_r}" y2="{scale_y(6.0)}" stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>
  <text x="{pad_l - 10}" y="{scale_y(6.0) + 4}" fill="#94a3b8" font-size="11" text-anchor="end">6.0</text>
  
  <line x1="{pad_l}" y1="{scale_y(4.0)}" x2="{svg_w - pad_r}" y2="{scale_y(4.0)}" stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>
  <text x="{pad_l - 10}" y="{scale_y(4.0) + 4}" fill="#94a3b8" font-size="11" text-anchor="end">4.0</text>
  
  <line x1="{pad_l}" y1="{scale_y(2.0)}" x2="{svg_w - pad_r}" y2="{scale_y(2.0)}" stroke="#334155" stroke-dasharray="4,4" stroke-width="1"/>
  <text x="{pad_l - 10}" y="{scale_y(2.0) + 4}" fill="#94a3b8" font-size="11" text-anchor="end">2.0</text>

  <line x1="{pad_l}" y1="{scale_y(0.0)}" x2="{svg_w - pad_r}" y2="{scale_y(0.0)}" stroke="#475569" stroke-width="1.5"/>
  <text x="{pad_l - 10}" y="{scale_y(0.0) + 4}" fill="#94a3b8" font-size="11" text-anchor="end">0.0</text>

  <!-- Gradients -->
  <defs>
    <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stop-color="#38bdf8" stop-opacity="0.35"/>
      <stop offset="100%" stop-color="#38bdf8" stop-opacity="0.0"/>
    </linearGradient>
  </defs>

  <!-- Area Under Trendline -->
  <path d="{ma_area}" fill="url(#areaGrad)"/>

  <!-- Raw Batch Loss Curve -->
  <path d="{raw_d}" fill="none" stroke="#64748b" stroke-width="1" stroke-opacity="0.6"/>

  <!-- Smoothed Moving Average Line -->
  <path d="{ma_d}" fill="none" stroke="#38bdf8" stroke-width="2.8" stroke-linecap="round"/>

  <!-- Key Milestone Markers -->
  <!-- Start -->
  <circle cx="{scale_x(0, len(points))}" cy="{scale_y(first_loss)}" r="5" fill="#f43f5e"/>
  <text x="{scale_x(0, len(points)) + 8}" y="{scale_y(first_loss) - 8}" fill="#f43f5e" font-size="12" font-weight="bold">Start: {first_loss:.2f}</text>

  <!-- Latest -->
  <circle cx="{scale_x(len(points)-1, len(points))}" cy="{scale_y(points[-1]['loss'])}" r="5" fill="#10b981"/>
  <text x="{scale_x(len(points)-1, len(points)) - 10}" y="{scale_y(points[-1]['loss']) - 10}" fill="#10b981" font-size="12" font-weight="bold" text-anchor="end">Now: {points[-1]['loss']:.2f}</text>

  <!-- Title & Legend -->
  <text x="{pad_l}" y="24" fill="#f8fafc" font-size="15" font-weight="600">Unlimited-OCR-2 Training Loss Trajectory (Indic LoRA)</text>
  <circle cx="{svg_w - 240}" cy="20" r="4" fill="#64748b"/>
  <text x="{svg_w - 230}" y="24" fill="#94a3b8" font-size="11">Batch Loss</text>
  <line x1="{svg_w - 150}" y1="20" x2="{svg_w - 130}" y2="20" stroke="#38bdf8" stroke-width="2.5"/>
  <text x="{svg_w - 120}" y="24" fill="#38bdf8" font-size="11" font-weight="bold">10-Step Trendline</text>

  <!-- X-Axis Labels -->
  <text x="{scale_x(0, len(points))}" y="{svg_h - 15}" fill="#64748b" font-size="11" text-anchor="start">Step 1 (Epoch 0.0)</text>
  <text x="{scale_x(int(len(points)/2), len(points))}" y="{svg_h - 15}" fill="#64748b" font-size="11" text-anchor="middle">Step {points[int(len(points)/2)]['step']} (Epoch {points[int(len(points)/2)]['epoch']})</text>
  <text x="{scale_x(len(points)-1, len(points))}" y="{svg_h - 15}" fill="#64748b" font-size="11" text-anchor="end">Step {points[-1]['step']} (Epoch {points[-1]['epoch']})</text>
</svg>"""

svg_out_path = os.path.join("outputs", "training_metrics.svg")
with open(svg_out_path, "w", encoding="utf-8") as f:
    f.write(svg_content)
print(f"Generated SVG chart -> {svg_out_path}")

# Write self-contained HTML artifact for generative_ui
artifact_dir = r"C:\Users\xande\.gemini\antigravity\brain\391edf02-5261-4bcc-a020-b6c8c15c43d9"
os.makedirs(artifact_dir, exist_ok=True)
html_artifact_path = os.path.join(artifact_dir, "training_metrics_graph.html")

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <script src="https://www.gstatic.com/antigravity/web/dev/tailwindcss.min.js"></script>
  <style>
    .metric-card {{
      background: var(--card);
      border: 1px solid var(--border);
    }}
  </style>
</head>
<body class="bg-transparent text-[var(--foreground)] antialiased p-3">
  <div class="bg-[var(--card)] text-[var(--foreground)] border border-[var(--border)] rounded-2xl p-5 shadow-lg max-w-4xl mx-auto">
    
    <!-- Header -->
    <div class="flex items-center justify-between border-b border-[var(--border)] pb-4 mb-4">
      <div>
        <h2 class="text-xl font-bold tracking-tight text-[var(--foreground)] flex items-center gap-2">
          <span>📉</span> Training Metrics & Convergence Curve
        </h2>
        <p class="text-xs text-[var(--muted-foreground)] mt-1">Baidu Unlimited-OCR-2 &bull; 4-bit QLoRA Indic Fine-Tuning</p>
      </div>
      <div class="flex items-center gap-2">
        <span class="inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <span class="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse mr-1.5"></span>
          Actively Training
        </span>
      </div>
    </div>

    <!-- Quick Stats Grid -->
    <div class="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-5">
      <div class="metric-card rounded-xl p-3">
        <span class="text-xs text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Start Loss</span>
        <div class="text-lg font-bold text-rose-400 mt-0.5">{first_loss:.2f}</div>
        <span class="text-xs text-[var(--muted-foreground)]">Step 1</span>
      </div>
      <div class="metric-card rounded-xl p-3">
        <span class="text-xs text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Current Loss</span>
        <div class="text-lg font-bold text-emerald-400 mt-0.5">{points[-1]['loss']:.2f}</div>
        <span class="text-xs text-[var(--muted-foreground)]">Step {points[-1]['step']}</span>
      </div>
      <div class="metric-card rounded-xl p-3">
        <span class="text-xs text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Loss Reduction</span>
        <div class="text-lg font-bold text-sky-400 mt-0.5">-{total_reduction:.1f}%</div>
        <span class="text-xs text-[var(--muted-foreground)]">Overall error drop</span>
      </div>
      <div class="metric-card rounded-xl p-3">
        <span class="text-xs text-[var(--muted-foreground)] uppercase tracking-wider font-semibold">Progress</span>
        <div class="text-lg font-bold text-amber-400 mt-0.5">{points[-1]['epoch']:.2f} / 3.0</div>
        <span class="text-xs text-[var(--muted-foreground)]">Epoch completed</span>
      </div>
    </div>

    <!-- Main Chart Container -->
    <div class="bg-slate-950/60 rounded-xl p-3 border border-[var(--border)] overflow-hidden">
      {svg_content}
    </div>

    <!-- Legend & Insights -->
    <div class="mt-4 pt-3 border-t border-[var(--border)] flex flex-wrap items-center justify-between text-xs text-[var(--muted-foreground)]">
      <div class="flex items-center gap-4">
        <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-sky-400 inline-block rounded"></span> 10-Step Moving Average (Trend)</span>
        <span class="flex items-center gap-1.5"><span class="w-3 h-0.5 bg-slate-500 inline-block rounded"></span> Individual Batch Loss</span>
      </div>
      <div>
        <span>Checkpoints on Disk: <strong class="text-[var(--foreground)]">checkpoint-150, checkpoint-200</strong></span>
      </div>
    </div>

  </div>
</body>
</html>
"""

with open(html_artifact_path, "w", encoding="utf-8") as f:
    f.write(html_content)

print(f"Generated HTML artifact -> {html_artifact_path}")
