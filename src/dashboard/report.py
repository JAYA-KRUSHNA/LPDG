"""Dashboard report generator — creates a premium interactive HTML dashboard.

Generates a stunning, dark-themed dashboard with glassmorphism, animations,
interactive Plotly charts, and filterable data views.

Usage:
    python -m src.dashboard.report
    make dashboard
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import subprocess
import sys
import webbrowser

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.model.evaluate import load_engineer_truth, evaluate_predictions
from src.utils.config import load_config, get_data_dir, get_model_dir
from src.utils.gateway_ids import normalize_gateway_id
from src.utils.logging_setup import setup_logging, get_logger

logger = get_logger(__name__)

# ── Premium color palette ──────────────────────────────────────────────────
P = {
    "bg":         "#0a0a0f",
    "bg2":        "#12121a",
    "surface":    "rgba(20, 20, 35, 0.7)",
    "surface2":   "rgba(30, 30, 50, 0.5)",
    "border":     "rgba(99, 102, 241, 0.15)",
    "border_h":   "rgba(99, 102, 241, 0.4)",
    "text":       "#f1f5f9",
    "text2":      "#94a3b8",
    "text3":      "#64748b",
    "indigo":     "#6366f1",
    "violet":     "#8b5cf6",
    "purple":     "#a855f7",
    "cyan":       "#06b6d4",
    "emerald":    "#10b981",
    "amber":      "#f59e0b",
    "rose":       "#f43f5e",
    "blue":       "#3b82f6",
    "glow_i":     "rgba(99, 102, 241, 0.25)",
    "glow_v":     "rgba(139, 92, 246, 0.25)",
    "glow_c":     "rgba(6, 182, 212, 0.25)",
    "glow_e":     "rgba(16, 185, 129, 0.25)",
}

CHART_COLORS = ["#6366f1", "#8b5cf6", "#a855f7", "#06b6d4", "#10b981",
                "#f59e0b", "#3b82f6", "#f43f5e"]

CHART_LAYOUT = dict(
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color=P["text2"], size=12),
)
GRID_COLOR = "rgba(99,102,241,0.08)"


def _load_model_data(model_dir: pathlib.Path) -> tuple[dict, dict]:
    """Load model metadata and feature info."""
    current = model_dir / "current"
    if not current.exists():
        versions = sorted([d.name for d in model_dir.iterdir()
                          if d.is_dir() and d.name.startswith("v")])
        if not versions:
            raise FileNotFoundError("No trained model found")
        version_dir = model_dir / versions[-1]
    else:
        version_dir = current

    with open(version_dir / "metadata.json") as f:
        metadata = json.load(f)
    with open(version_dir / "features.json") as f:
        features = json.load(f)
    return metadata, features


# ══════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ══════════════════════════════════════════════════════════════════════════

def _chart_feature_importance(features: dict) -> str:
    items = list(features["feature_importance"].items())[:15]
    items.reverse()
    names = [n.replace("_", " ").title() for n, _ in items]
    values = [v for _, v in items]
    mx = max(values) if values else 1

    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker=dict(
            color=[f"rgba(99,102,241,{0.35 + 0.65*v/mx})" for v in values],
            line=dict(color="rgba(139,92,246,0.6)", width=1),
        ),
        hovertemplate="<b>%{y}</b><br>Gain: %{x:,.0f}<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, height=440,
                      title=None, showlegend=False,
                      xaxis_title="Information Gain",
                      margin=dict(l=200, r=30, t=20, b=40))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _chart_risk_by_week(predictions: pd.DataFrame) -> str:
    fig = go.Figure()
    weeks = sorted(predictions["week_start"].unique())
    for i, w in enumerate(weeks):
        wd = predictions[predictions["week_start"] == w].sort_values("rank")
        label = pd.Timestamp(w).strftime("%b %d")
        fig.add_trace(go.Scatter(
            x=wd["rank"], y=wd["score"], name=label, mode="lines+markers",
            line=dict(color=CHART_COLORS[i % len(CHART_COLORS)], width=2.5),
            marker=dict(size=6, line=dict(width=1, color="#0a0a0f")),
            hovertemplate=f"<b>{label}</b><br>Rank: %{{x}}<br>Score: %{{y:.4f}}<extra></extra>",
        ))
    fig.update_layout(
        **CHART_LAYOUT, height=360, showlegend=True,
        legend=dict(orientation="h", y=-0.18, x=0.5, xanchor="center",
                    font=dict(size=11, color=P["text2"])),
        xaxis_title="Rank (1 = highest risk)",
        yaxis_title="Risk Score",
        margin=dict(l=60, r=30, t=20, b=70),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _chart_heatmap(predictions: pd.DataFrame) -> str:
    weeks = sorted(predictions["week_start"].unique())
    wl = [pd.Timestamp(w).strftime("%b %d") for w in weeks]
    gateways = sorted(predictions["gateway_id"].unique())

    # Count appearances
    app_count = predictions.groupby("gateway_id").size().to_dict()
    gateways_sorted = sorted(gateways, key=lambda g: (-app_count.get(g, 0),))

    matrix = []
    for gw in gateways_sorted:
        row = []
        for w in weeks:
            m = (predictions["gateway_id"] == gw) & (predictions["week_start"] == w)
            row.append(float(predictions.loc[m, "score"].iloc[0]) if m.any() else 0)
        matrix.append(row)

    fig = go.Figure(go.Heatmap(
        z=matrix, x=wl, y=[g[:8] + "…" for g in gateways_sorted],
        colorscale=[[0, "#0a0a0f"], [0.01, "#1e1b4b"], [0.2, "#3730a3"],
                    [0.5, "#6366f1"], [0.8, "#a78bfa"], [1, "#e0e7ff"]],
        hovertemplate="<b>%{y}</b><br>Week: %{x}<br>Score: %{z:.3f}<extra></extra>",
        colorbar=dict(title="Risk", titlefont=dict(color=P["text2"]),
                      tickfont=dict(color=P["text3"]), thickness=12),
    ))
    fig.update_layout(**CHART_LAYOUT, height=max(420, len(gateways) * 14 + 80),
                      xaxis=dict(side="top", gridcolor="rgba(0,0,0,0)"),
                      yaxis=dict(autorange="reversed", gridcolor="rgba(0,0,0,0)"),
                      margin=dict(l=100, r=60, t=40, b=20))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _chart_cost_comparison(model_eval: dict, baseline_eval: dict) -> str:
    fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                        subplot_titles=["Total Cost (€)", "Gateway Outcomes"],
                        horizontal_spacing=0.12)

    # Cost bars
    fig.add_trace(go.Bar(
        x=["3σ Baseline", "Our Model"],
        y=[baseline_eval["total_cost"], model_eval["total_cost"]],
        marker=dict(color=[P["text3"], P["indigo"]],
                    line=dict(width=0)),
        text=[f"€{baseline_eval['total_cost']:,.0f}", f"€{model_eval['total_cost']:,.0f}"],
        textposition="outside", textfont=dict(color=P["text"], size=14, family="Inter"),
        showlegend=False,
    ), row=1, col=1)

    cats = ["Caught ✓", "Missed ✗", "False Alarms"]
    bvals = [baseline_eval["total_caught"], baseline_eval["total_missed"], baseline_eval["total_false_alarms"]]
    mvals = [model_eval["total_caught"], model_eval["total_missed"], model_eval["total_false_alarms"]]

    fig.add_trace(go.Bar(x=cats, y=bvals, name="Baseline",
                         marker=dict(color=P["text3"], opacity=0.5),
                         text=bvals, textposition="outside",
                         textfont=dict(color=P["text3"])), row=1, col=2)
    fig.add_trace(go.Bar(x=cats, y=mvals, name="Our Model",
                         marker=dict(color=P["indigo"]),
                         text=mvals, textposition="outside",
                         textfont=dict(color=P["text"])), row=1, col=2)

    fig.update_layout(
        **{k: v for k, v in CHART_LAYOUT.items() if k not in ("xaxis", "yaxis")},
        height=360, barmode="group",
        legend=dict(orientation="h", y=-0.15, x=0.75, xanchor="center",
                    font=dict(color=P["text2"])),
        margin=dict(l=50, r=30, t=50, b=60),
    )
    fig.update_xaxes(gridcolor="rgba(0,0,0,0)")
    fig.update_yaxes(gridcolor="rgba(99,102,241,0.08)")
    for ann in fig.layout.annotations:
        ann.font = dict(size=14, color=P["text"], family="Inter")
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _chart_cv(metadata: dict) -> str:
    cv = metadata["cv_metrics"]
    names = ["AUC-ROC", "AUC-PR", "Recall@15"]
    means = [float(cv[f"cv_{k}_mean"]) for k in ["auc_roc", "auc_pr", "recall_at_15"]]
    stds = [float(cv[f"cv_{k}_std"]) for k in ["auc_roc", "auc_pr", "recall_at_15"]]
    colors = [P["indigo"], P["violet"], P["cyan"]]

    fig = go.Figure(go.Bar(
        x=names, y=means,
        error_y=dict(type="data", array=stds, visible=True, color=P["violet"], thickness=2, width=6),
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{m:.3f}" for m in means],
        textposition="outside", textfont=dict(color=P["text"], size=14),
        hovertemplate="<b>%{x}</b><br>%{y:.4f} ± %{error_y.array:.4f}<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, height=360, showlegend=False,
                      yaxis=dict(range=[0, 1.15], gridcolor="rgba(99,102,241,0.08)"),
                      margin=dict(l=50, r=30, t=20, b=40))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _chart_score_histogram(predictions: pd.DataFrame) -> str:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=predictions["score"], nbinsx=30,
        marker=dict(color=P["indigo"], line=dict(color=P["violet"], width=1)),
        opacity=0.85,
        hovertemplate="Score: %{x:.3f}<br>Count: %{y}<extra></extra>",
    ))
    fig.update_layout(**CHART_LAYOUT, height=300, showlegend=False,
                      xaxis_title="Risk Score", yaxis_title="Count",
                      margin=dict(l=50, r=30, t=20, b=50))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


def _chart_weekly_stats(predictions: pd.DataFrame) -> str:
    weeks = sorted(predictions["week_start"].unique())
    labels = [pd.Timestamp(w).strftime("%b %d") for w in weeks]
    means = [predictions[predictions["week_start"] == w]["score"].mean() for w in weeks]
    maxes = [predictions[predictions["week_start"] == w]["score"].max() for w in weeks]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=labels, y=maxes, name="Max Risk", mode="lines+markers",
        line=dict(color=P["rose"], width=2),
        marker=dict(size=8, symbol="diamond"),
        fill="tozeroy", fillcolor="rgba(244,63,94,0.08)",
    ))
    fig.add_trace(go.Scatter(
        x=labels, y=means, name="Avg Risk", mode="lines+markers",
        line=dict(color=P["cyan"], width=2),
        marker=dict(size=8),
        fill="tozeroy", fillcolor="rgba(6,182,212,0.08)",
    ))
    fig.update_layout(**CHART_LAYOUT, height=300, showlegend=True,
                      legend=dict(orientation="h", y=-0.2, x=0.5, xanchor="center"),
                      yaxis_title="Risk Score",
                      margin=dict(l=50, r=30, t=20, b=60))
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════
#  HTML BUILDER
# ══════════════════════════════════════════════════════════════════════════

def _predictions_rows(predictions: pd.DataFrame) -> str:
    rows = ""
    for _, r in predictions.iterrows():
        s = r["score"]
        cls = "high" if s > 0.3 else "med" if s > 0.15 else "low"
        wk = pd.Timestamp(r["week_start"]).strftime("%b %d")
        rows += f"""<tr class="fade-row" data-week="{r['week_start']}">
            <td class="wk">{wk}</td>
            <td class="rk">#{int(r['rank'])}</td>
            <td class="gw"><span class="gw-badge">{r['gateway_id']}</span></td>
            <td class="sc {cls}">{s:.4f}</td>
            <td class="rs">{r['reason']}</td>
        </tr>"""
    return rows


def generate_dashboard(
    predictions_path: pathlib.Path,
    model_dir: pathlib.Path,
    data_dir: pathlib.Path,
    output_path: pathlib.Path,
    config: dict | None = None,
) -> pathlib.Path:
    """Generate a premium interactive HTML dashboard."""
    if config is None:
        config = load_config()

    logger.info("Generating dashboard...")

    predictions = pd.read_csv(predictions_path)
    metadata, features = _load_model_data(model_dir)
    cv = metadata["cv_metrics"]

    truth = load_engineer_truth(data_dir)

    baseline_path = pathlib.Path("predictions_baseline.csv")
    if not baseline_path.exists():
        subprocess.run([sys.executable, "baseline_3sigma.py",
                        "--data", str(data_dir), "--out", str(baseline_path)],
                       capture_output=True)

    model_eval = baseline_eval = None
    cost_delta = cost_pct = caught_delta = 0
    if truth and baseline_path.exists():
        baseline_pred = pd.read_csv(baseline_path)
        model_eval = evaluate_predictions(predictions, truth)
        baseline_eval = evaluate_predictions(baseline_pred, truth)
        cost_delta = baseline_eval["total_cost"] - model_eval["total_cost"]
        cost_pct = cost_delta / max(baseline_eval["total_cost"], 1) * 100
        caught_delta = model_eval["total_caught"] - baseline_eval["total_caught"]

    # Build all charts
    c_feat = _chart_feature_importance(features)
    c_risk = _chart_risk_by_week(predictions)
    c_heat = _chart_heatmap(predictions)
    c_cv = _chart_cv(metadata)
    c_hist = _chart_score_histogram(predictions)
    c_weekly = _chart_weekly_stats(predictions)
    c_cost = _chart_cost_comparison(model_eval, baseline_eval) if model_eval else ""

    pred_rows = _predictions_rows(predictions)
    weeks_json = json.dumps(sorted(predictions["week_start"].unique().tolist()))
    week_labels_json = json.dumps([pd.Timestamp(w).strftime("%b %d, %Y")
                                   for w in sorted(predictions["week_start"].unique())])

    auc = float(cv["cv_auc_roc_mean"])
    auc_std = float(cv["cv_auc_roc_std"])
    n_feat = metadata["n_features"]
    n_trees = cv.get("final_num_trees", "—")
    n_pred = len(predictions)
    n_weeks = predictions["week_start"].nunique()
    n_unique_gw = predictions["gateway_id"].nunique()
    gen_at = dt.datetime.now().strftime("%B %d, %Y at %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>LPDG Dashboard — Gateway Health Prediction</title>
<script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<style>
/* ── Reset & Base ─────────────────────────────────────── */
*{{margin:0;padding:0;box-sizing:border-box}}
html{{scroll-behavior:smooth}}
body{{
  font-family:'Inter',system-ui,sans-serif;
  background:{P["bg"]};
  color:{P["text"]};
  line-height:1.6;
  overflow-x:hidden;
}}

/* ── Animated background ──────────────────────────────── */
body::before{{
  content:'';position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;
  background:
    radial-gradient(ellipse 600px 600px at 20% 10%, rgba(99,102,241,0.07) 0%, transparent 70%),
    radial-gradient(ellipse 500px 500px at 80% 20%, rgba(139,92,246,0.05) 0%, transparent 70%),
    radial-gradient(ellipse 400px 400px at 50% 80%, rgba(6,182,212,0.04) 0%, transparent 70%);
  animation: bgPulse 8s ease-in-out infinite alternate;
}}
@keyframes bgPulse{{
  0%{{opacity:0.7;transform:scale(1)}}
  100%{{opacity:1;transform:scale(1.05)}}
}}

/* ── Nav ──────────────────────────────────────────────── */
.nav{{
  position:sticky;top:0;z-index:100;
  background:rgba(10,10,15,0.85);
  backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);
  border-bottom:1px solid {P["border"]};
  padding:0.8rem 2rem;
  display:flex;align-items:center;justify-content:space-between;
}}
.nav-brand{{
  font-size:1rem;font-weight:700;
  background:linear-gradient(135deg,{P["indigo"]},{P["violet"]},{P["cyan"]});
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
}}
.nav-links{{display:flex;gap:0.3rem}}
.nav-links a{{
  color:{P["text3"]};font-size:0.78rem;font-weight:500;
  padding:0.4rem 0.8rem;border-radius:8px;text-decoration:none;
  transition:all 0.25s ease;
}}
.nav-links a:hover,.nav-links a.active{{
  color:{P["text"]};background:rgba(99,102,241,0.12);
}}

/* ── Hero ─────────────────────────────────────────────── */
.hero{{
  position:relative;z-index:1;
  text-align:center;padding:3.5rem 2rem 2.5rem;
}}
.hero h1{{
  font-size:2.4rem;font-weight:800;
  background:linear-gradient(135deg,#e0e7ff,{P["indigo"]},{P["violet"]});
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;
  margin-bottom:0.5rem;letter-spacing:-0.02em;
}}
.hero p{{color:{P["text3"]};font-size:1rem;font-weight:300;margin-bottom:1rem}}
.hero-badges{{display:flex;gap:0.5rem;justify-content:center;flex-wrap:wrap}}
.badge{{
  display:inline-flex;align-items:center;gap:0.3rem;
  background:{P["surface"]};
  border:1px solid {P["border"]};
  padding:0.3rem 0.9rem;border-radius:99px;
  font-size:0.72rem;font-weight:500;color:{P["text2"]};
  backdrop-filter:blur(10px);
}}
.badge .dot{{width:6px;height:6px;border-radius:50%;display:inline-block}}

/* ── Container & Sections ─────────────────────────────── */
.container{{max-width:1320px;margin:0 auto;padding:0 1.5rem;position:relative;z-index:1}}
section{{margin-bottom:2.5rem;opacity:0;transform:translateY(20px);transition:all 0.6s ease}}
section.visible{{opacity:1;transform:translateY(0)}}
.section-header{{
  display:flex;align-items:center;gap:0.6rem;
  margin-bottom:1.2rem;padding-bottom:0.6rem;
  border-bottom:1px solid {P["border"]};
}}
.section-header h2{{font-size:1.05rem;font-weight:600;color:{P["text"]}}}
.section-header .icon{{font-size:1.2rem}}
.section-desc{{font-size:0.8rem;color:{P["text3"]};margin:-0.6rem 0 1rem;line-height:1.5;max-width:800px}}

/* ── Glass Card ───────────────────────────────────────── */
.glass{{
  background:{P["surface"]};
  border:1px solid {P["border"]};
  border-radius:16px;
  backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);
  padding:1.4rem;
  transition:all 0.35s ease;
}}
.glass:hover{{
  border-color:{P["border_h"]};
  box-shadow:0 8px 32px rgba(99,102,241,0.08);
}}

/* ── KPI Grid ─────────────────────────────────────────── */
.kpi-grid{{
  display:grid;
  grid-template-columns:repeat(auto-fit,minmax(200px,1fr));
  gap:1rem;margin-bottom:2.5rem;
}}
.kpi{{
  background:{P["surface"]};
  border:1px solid {P["border"]};
  border-radius:16px;padding:1.3rem;
  backdrop-filter:blur(12px);
  position:relative;overflow:hidden;
  transition:all 0.35s ease;
}}
.kpi:hover{{
  transform:translateY(-4px);
  border-color:{P["border_h"]};
  box-shadow:0 12px 40px var(--glow,{P["glow_i"]});
}}
.kpi::before{{
  content:'';position:absolute;top:0;right:0;
  width:60px;height:60px;border-radius:0 16px 0 60px;
  background:var(--accent,{P["indigo"]});opacity:0.08;
  transition:opacity 0.3s;
}}
.kpi:hover::before{{opacity:0.15}}
.kpi-label{{font-size:0.72rem;font-weight:500;color:{P["text3"]};text-transform:uppercase;letter-spacing:0.06em}}
.kpi-val{{
  font-size:1.9rem;font-weight:800;
  font-family:'JetBrains Mono',monospace;
  color:var(--accent,{P["indigo"]});
  margin:0.2rem 0;
}}
.kpi-sub{{font-size:0.73rem;color:{P["text3"]}}}

/* ── Charts grid ──────────────────────────────────────── */
.grid-2{{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}}
.grid-3{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.2rem}}
@media(max-width:1024px){{.grid-2,.grid-3{{grid-template-columns:1fr}}}}

/* ── Table ────────────────────────────────────────────── */
.tbl-wrap{{overflow-x:auto;border-radius:12px}}
.tbl{{width:100%;border-collapse:collapse;font-size:0.82rem}}
.tbl thead th{{
  background:rgba(99,102,241,0.08);color:{P["indigo"]};
  font-weight:600;font-size:0.7rem;text-transform:uppercase;letter-spacing:0.05em;
  padding:0.75rem 0.8rem;text-align:left;position:sticky;top:0;z-index:2;
}}
.tbl tbody tr{{transition:all 0.2s;border-bottom:1px solid rgba(99,102,241,0.06)}}
.tbl tbody tr:hover{{background:rgba(99,102,241,0.05)}}
.tbl tbody tr.hidden{{display:none}}
.tbl td{{padding:0.55rem 0.8rem;color:{P["text2"]}}}
.tbl .wk{{font-weight:500;color:{P["text"]}}}
.tbl .rk{{font-family:'JetBrains Mono',monospace;font-weight:600;color:{P["text"]}}}
.tbl .gw-badge{{
  font-family:'JetBrains Mono',monospace;font-size:0.76rem;
  background:rgba(99,102,241,0.1);padding:0.15rem 0.5rem;
  border-radius:6px;color:{P["violet"]};
}}
.tbl .sc{{font-family:'JetBrains Mono',monospace;font-weight:600}}
.tbl .sc.high{{color:{P["rose"]}}}
.tbl .sc.med{{color:{P["amber"]}}}
.tbl .sc.low{{color:{P["emerald"]}}}
.tbl .rs{{font-size:0.74rem;max-width:320px;color:{P["text3"]}}}

/* ── Week filter ──────────────────────────────────────── */
.week-filter{{display:flex;gap:0.4rem;margin-bottom:1rem;flex-wrap:wrap}}
.week-btn{{
  background:{P["surface2"]};border:1px solid {P["border"]};
  color:{P["text3"]};font-size:0.75rem;font-weight:500;font-family:'Inter',sans-serif;
  padding:0.4rem 0.9rem;border-radius:8px;cursor:pointer;
  transition:all 0.25s ease;
}}
.week-btn:hover,.week-btn.active{{
  background:rgba(99,102,241,0.15);color:{P["text"]};border-color:{P["indigo"]};
}}

/* ── Footer ───────────────────────────────────────────── */
.footer{{
  text-align:center;padding:2rem;color:{P["text3"]};font-size:0.72rem;
  border-top:1px solid {P["border"]};margin-top:3rem;
}}

/* ── Scrollbar ────────────────────────────────────────── */
::-webkit-scrollbar{{width:5px;height:5px}}
::-webkit-scrollbar-track{{background:transparent}}
::-webkit-scrollbar-thumb{{background:{P["border_h"]};border-radius:3px}}

/* ── Animations ───────────────────────────────────────── */
@keyframes countUp{{from{{opacity:0;transform:translateY(10px)}}to{{opacity:1;transform:translateY(0)}}}}
.kpi{{animation:countUp 0.5s ease forwards;animation-delay:var(--delay,0s)}}
</style>
</head>
<body>

<!-- Nav -->
<nav class="nav">
  <div class="nav-brand">⚡ LPDG Dashboard</div>
  <div class="nav-links">
    <a href="#overview" class="active">Overview</a>
    <a href="#analysis">Analysis</a>
    <a href="#heatmap">Heatmap</a>
    <a href="#predictions">Predictions</a>
  </div>
</nav>

<!-- Hero -->
<div class="hero">
  <h1>Gateway Health Prediction</h1>
  <p>AI-powered predictive maintenance for 320 LoRaWAN gateways</p>
  <div class="hero-badges">
    <span class="badge"><span class="dot" style="background:{P["emerald"]}"></span> Model {metadata["version"]}</span>
    <span class="badge"><span class="dot" style="background:{P["indigo"]}"></span> {n_feat} Features</span>
    <span class="badge"><span class="dot" style="background:{P["violet"]}"></span> {n_trees} Trees</span>
    <span class="badge"><span class="dot" style="background:{P["cyan"]}"></span> LightGBM</span>
  </div>
</div>

<div class="container">

<!-- KPIs -->
<section id="overview" class="visible">
  <div class="kpi-grid">
    <div class="kpi" style="--accent:{P["indigo"]};--glow:{P["glow_i"]};--delay:0s">
      <div class="kpi-label">AUC-ROC</div>
      <div class="kpi-val">{auc:.3f}</div>
      <div class="kpi-sub">± {auc_std:.3f} (3-fold CV)</div>
    </div>
    <div class="kpi" style="--accent:{P["emerald"]};--glow:{P["glow_e"]};--delay:0.08s">
      <div class="kpi-label">Cost Savings</div>
      <div class="kpi-val">€{cost_delta:,.0f}</div>
      <div class="kpi-sub">{cost_pct:.1f}% vs 3σ baseline</div>
    </div>
    <div class="kpi" style="--accent:{P["violet"]};--glow:{P["glow_v"]};--delay:0.16s">
      <div class="kpi-label">Extra Catches</div>
      <div class="kpi-val">+{caught_delta}</div>
      <div class="kpi-sub">{model_eval["total_caught"] if model_eval else 0} caught vs {baseline_eval["total_caught"] if baseline_eval else 0} baseline</div>
    </div>
    <div class="kpi" style="--accent:{P["cyan"]};--glow:{P["glow_c"]};--delay:0.24s">
      <div class="kpi-label">Predictions</div>
      <div class="kpi-val">{n_pred}</div>
      <div class="kpi-sub">{n_weeks} weeks × 15 gateways</div>
    </div>
    <div class="kpi" style="--accent:{P["amber"]};--glow:rgba(245,158,11,0.2);--delay:0.32s">
      <div class="kpi-label">Unique Gateways</div>
      <div class="kpi-val">{n_unique_gw}</div>
      <div class="kpi-sub">of 320 flagged at least once</div>
    </div>
    <div class="kpi" style="--accent:{P["rose"]};--glow:rgba(244,63,94,0.2);--delay:0.4s">
      <div class="kpi-label">Positive Rate</div>
      <div class="kpi-val">{metadata["positive_rate"]:.1%}</div>
      <div class="kpi-sub">{metadata["training_samples"]:,} training samples</div>
    </div>
  </div>
</section>

<!-- Cost Comparison -->
{"" if not c_cost else f'''
<section id="cost">
  <div class="section-header"><span class="icon">💰</span><h2>Cost Comparison — Model vs 3σ Baseline</h2></div>
  <p class="section-desc">Every visit costs €380. Every missed broken gateway costs €600/week in lost meter data. Our model selects the top 15 gateways per week that maximize catches while minimizing wasted visits. The waterfall shows how our model achieves net savings over the statistical baseline.</p>
  <div class="glass">{c_cost}</div>
</section>
'''}

<!-- Analysis -->
<section id="analysis">
  <div class="section-header"><span class="icon">🔬</span><h2>Model Analysis</h2></div>
  <p class="section-desc"><b>Feature Importance</b> shows which signals the model relies on most — <em>meters_at_risk</em> and <em>read_rate_trend</em> dominate, confirming that declining data quality is the best predictor of gateway failure. <b>Cross-Validation</b> uses 3-fold gateway-level splits to prevent data leakage — no gateway appears in both train and validation.</p>
  <div class="grid-2">
    <div class="glass">
      <div style="font-size:0.8rem;font-weight:600;color:{P["text"]};margin-bottom:0.5rem">Feature Importance (Top 15)</div>
      {c_feat}
    </div>
    <div class="glass">
      <div style="font-size:0.8rem;font-weight:600;color:{P["text"]};margin-bottom:0.5rem">Cross-Validation Performance</div>
      {c_cv}
    </div>
  </div>
</section>

<!-- Risk Trends -->
<section>
  <div class="section-header"><span class="icon">📈</span><h2>Risk Score Trends</h2></div>
  <p class="section-desc">The rank curve shows risk scores drop steeply after rank 5, confirming the model is confident about the worst gateways. Score distributions per week reveal whether certain weeks have more severe outliers — useful for staffing decisions.</p>
  <div class="grid-2">
    <div class="glass">
      <div style="font-size:0.8rem;font-weight:600;color:{P["text"]};margin-bottom:0.5rem">Risk Score by Rank (per Week)</div>
      {c_risk}
    </div>
    <div>
      <div class="glass" style="margin-bottom:1.2rem">
        <div style="font-size:0.8rem;font-weight:600;color:{P["text"]};margin-bottom:0.5rem">Weekly Risk Trend</div>
        {c_weekly}
      </div>
      <div class="glass">
        <div style="font-size:0.8rem;font-weight:600;color:{P["text"]};margin-bottom:0.5rem">Score Distribution</div>
        {c_hist}
      </div>
    </div>
  </div>
</section>

<!-- Heatmap -->
<section id="heatmap">
  <div class="section-header"><span class="icon">🗺️</span><h2>Gateway Selection Heatmap</h2></div>
  <p class="section-desc">Gateways appearing across multiple weeks (top rows) are chronic problem devices — they may need replacement rather than repair. Gateways appearing in only 1–2 weeks may have transient issues. Bright cells indicate higher risk scores.</p>
  <div class="glass">{c_heat}</div>
</section>

<!-- Predictions Table -->
<section id="predictions">
  <div class="section-header"><span class="icon">📋</span><h2>All {n_pred} Predictions</h2></div>
  <p class="section-desc">Each row represents a recommended visit: 15 gateways per week, ranked by risk score. The <em>Reason</em> column provides a human-readable explanation generated from the top contributing features for that gateway. Filter by week to see individual schedules.</p>
  <div class="week-filter">
    <button class="week-btn active" onclick="filterWeek('all')">All Weeks</button>
  </div>
  <div class="glass">
    <div class="tbl-wrap" style="max-height:600px;overflow-y:auto">
      <table class="tbl">
        <thead><tr>
          <th>Week</th><th>Rank</th><th>Gateway ID</th><th>Risk Score</th><th>Reason</th>
        </tr></thead>
        <tbody id="pred-body">{pred_rows}</tbody>
      </table>
    </div>
  </div>
</section>

</div>

<div class="footer">
  Generated on {gen_at} &nbsp;·&nbsp; LPDG Gateway Health Prediction System &nbsp;·&nbsp; Model {metadata["version"]}
</div>

<script>
// ── Scroll reveal ─────────────────────────────────────────
const observer = new IntersectionObserver((entries) => {{
  entries.forEach(e => {{ if(e.isIntersecting) e.target.classList.add('visible') }})
}}, {{ threshold: 0.1 }});
document.querySelectorAll('section').forEach(s => observer.observe(s));

// ── Nav active link ───────────────────────────────────────
document.querySelectorAll('.nav-links a').forEach(link => {{
  link.addEventListener('click', () => {{
    document.querySelectorAll('.nav-links a').forEach(l => l.classList.remove('active'));
    link.classList.add('active');
  }});
}});

// ── Week filter buttons ───────────────────────────────────
const weeks = {weeks_json};
const weekLabels = {week_labels_json};
const filterContainer = document.querySelector('.week-filter');
weeks.forEach((w, i) => {{
  const btn = document.createElement('button');
  btn.className = 'week-btn';
  btn.textContent = weekLabels[i];
  btn.onclick = () => filterWeek(w);
  filterContainer.appendChild(btn);
}});

function filterWeek(week) {{
  document.querySelectorAll('.week-btn').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  document.querySelectorAll('#pred-body tr').forEach(row => {{
    if (week === 'all' || row.dataset.week === week)
      row.classList.remove('hidden');
    else
      row.classList.add('hidden');
  }});
}}

// ── Smooth Plotly resize ──────────────────────────────────
window.addEventListener('resize', () => {{
  document.querySelectorAll('.js-plotly-plot').forEach(p => Plotly.Plots.resize(p));
}});
</script>
</body>
</html>"""

    output_path.write_text(html, encoding="utf-8")
    logger.info("Dashboard saved to %s", output_path)
    return output_path


def main(argv: list[str] | None = None) -> int:
    """Generate dashboard and open in browser."""
    setup_logging()

    parser = argparse.ArgumentParser(description="Generate gateway health dashboard")
    parser.add_argument("--predictions", type=pathlib.Path,
                        default=pathlib.Path("predictions.csv"))
    parser.add_argument("--model-dir", type=pathlib.Path, default=None)
    parser.add_argument("--data", type=pathlib.Path, default=None)
    parser.add_argument("--out", type=pathlib.Path,
                        default=pathlib.Path("dashboard.html"))
    parser.add_argument("--no-open", action="store_true",
                        help="Don't open in browser after generating")
    args = parser.parse_args(argv)

    config = load_config()
    model_dir = args.model_dir or get_model_dir(config)
    data_dir = args.data or get_data_dir(config)

    output = generate_dashboard(
        predictions_path=args.predictions,
        model_dir=model_dir,
        data_dir=data_dir,
        output_path=args.out,
        config=config,
    )

    if not args.no_open:
        webbrowser.open(f"file://{output.resolve()}")
        logger.info("Dashboard opened in browser")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
