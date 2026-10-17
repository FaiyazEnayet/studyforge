"""Build docs/dashboard.html: a self-contained interactive Plotly dashboard.

Uses the real studyforge engine (via studyforge.dashboard). Regenerate with:

    python3 tools/make_dashboard.py

No install needed: the script adds src/ to sys.path. The output HTML
embeds the Plotly.js bundle, so the file works standalone on GitHub Pages.
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import plotly.graph_objects as go  # noqa: E402
from plotly.offline import get_plotlyjs  # noqa: E402

from studyforge import state as st  # noqa: E402
from studyforge import dashboard as dash  # noqa: E402

OUT = os.path.join(ROOT, "docs", "dashboard.html")


def fig_overview(data: dict) -> go.Figure:
    ex = data["exam_summary"]
    names = list(ex)
    fig = go.Figure(
        go.Bar(
            x=names,
            y=[ex[n]["recall_at_exam"] * 100 for n in names],
            text=[f"{ex[n]['recall_at_exam']:.0%}" for n in names],
            textposition="outside",
            marker_color="#2a9d8f",
        )
    )
    fig.update_layout(
        title="Predicted recall at exam time after the plan",
        yaxis_title="recall (%)",
        yaxis_range=[0, 105],
        template="plotly_white",
        height=360,
        margin=dict(t=60, b=120),
    )
    return fig


def fig_daily_load(data: dict) -> go.Figure:
    horizon = data["horizon"]
    cap = data["config"]["daily_budget_minutes"]
    rest = set(data["config"]["rest_days"])
    days = list(range(horizon))
    used = [data["day_minutes"].get(d, 0) for d in days]
    colors = ["#e76f51" if d in rest else "#264653" for d in days]
    fig = go.Figure()
    fig.add_bar(x=days, y=used, marker_color=colors, name="scheduled")
    fig.add_hline(y=cap, line_dash="dash", line_color="#e9c46a",
                  annotation_text="daily cap")
    fig.update_layout(
        title="Minutes per day (red = rest day)",
        xaxis_title="day", yaxis_title="minutes",
        template="plotly_white", height=360,
    )
    return fig


def fig_schedule_table(data: dict) -> go.Figure:
    rows = []
    for d in range(data["horizon"]):
        sessions = data["schedule"].get(d, [])
        if d in set(data["config"]["rest_days"]):
            rows.append([f"Day {d}", "rest", "", ""])
        elif not sessions:
            rows.append([f"Day {d}", "no sessions", "", ""])
        for name, minutes in sessions:
            rows.append([f"Day {d}", data["exam_of"][name], name,
                         f"{minutes} min"])
    fig = go.Figure(go.Table(
        header=dict(values=["Day", "Exam", "Topic", "Minutes"],
                    fill_color="#264653", font_color="white"),
        cells=dict(values=list(zip(*rows)) if rows else [[], [], [], []],
                   fill_color="#f4f1de", align="left"),
    ))
    fig.update_layout(title="Full schedule", template="plotly_white",
                      height=600)
    return fig


def fig_forgetting_curves(data: dict) -> go.Figure:
    horizon = data["horizon"]
    fig = go.Figure()
    buttons = []
    for i, (name, info) in enumerate(sorted(data["topic_info"].items())):
        traj = dash.recall_trajectory(
            name, info["days"], info["S0"], info["exam_day"], horizon)
        fig.add_trace(go.Scatter(
            x=traj["xs"], y=[r * 100 for r in traj["ys"]],
            mode="lines", name=name, visible=(i == 0),
            line=dict(color="#264653"),
        ))
        rx, ry = zip(*traj["review_marks"]) if traj["review_marks"] else ([], [])
        fig.add_trace(go.Scatter(
            x=list(rx), y=[r * 100 for r in ry],
            mode="markers", name=f"{name} reviews", visible=(i == 0),
            marker=dict(color="#e76f51", size=9, symbol="diamond"),
            showlegend=False,
        ))
        vis = [False] * (2 * len(data["topic_info"]))
        vis[2 * i] = vis[2 * i + 1] = True
        buttons.append(dict(
            label=name, method="update",
            args=[{"visible": vis},
                  {"title": f"Recall over time: {name} "
                            f"(stability {info['S0']:.2f} -> "
                            f"{traj['S_final']:.2f} days, "
                            f"recall at exam {traj['recall_at_exam']:.0%})"}],
        ))
    fig.update_layout(
        title="Recall over time (pick a topic)",
        xaxis_title="day", yaxis_title="recall (%)",
        yaxis_range=[0, 105],
        updatemenus=[dict(active=0, buttons=buttons, x=1.02, y=1.0)],
        template="plotly_white", height=460,
    )
    return fig


def fig_whatif_sweep(config: dict, days: list[int]) -> go.Figure:
    gains, labels = [], []
    for d in days:
        w = dash.what_if_data(config, 0, d, 60)
        gains.append(w["delta"])
        labels.append(f"day {d}")
    fig = go.Figure(go.Bar(
        x=labels, y=gains,
        text=[f"{g:+.3f}" for g in gains], textposition="outside",
        marker_color="#2a9d8f",
    ))
    fig.update_layout(
        title="Objective gain from +60 minutes on each day",
        yaxis_title="gain in objective",
        template="plotly_white", height=360,
        margin=dict(b=110),
    )
    return fig


def fig_whatif_detail(config: dict) -> go.Figure:
    w = dash.what_if_data(config, 0, 3, 60)
    rows = []
    for d, exam, name, minutes in w["added"]:
        rows.append([f"Day {d}", "added", exam, name, f"{minutes} min"])
    for d, exam, name, minutes in w["removed"]:
        rows.append([f"Day {d}", "displaced", exam, name, f"{minutes} min"])
    cells = list(zip(*rows)) if rows else [["-"] * 5]
    fig = go.Figure(go.Table(
        header=dict(values=["Day", "Change", "Exam", "Topic", "Minutes"],
                    fill_color="#264653", font_color="white"),
        cells=dict(values=cells, fill_color="#f4f1de", align="left"),
    ))
    fig.update_layout(
        title=f"What +60 min on day 3 changes "
              f"(objective {w['baseline_obj']:.3f} -> {w['new_obj']:.3f}, "
              f"{w['delta']:+.3f})",
        template="plotly_white", height=420,
    )
    return fig


def fig_fit() -> go.Figure:
    fd = dash.fit_demo_data()
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fd["xs"], y=[r * 100 for r in fd["curve"]],
        mode="lines", name=f"fitted curve (S = {fd['S_fit']:.2f} days)",
        line=dict(color="#264653"),
    ))
    px, pr = zip(*fd["pairs"])
    fig.add_trace(go.Scatter(
        x=list(px), y=[r * 100 for r in pr],
        mode="markers", name="your logged reviews",
        marker=dict(color="#e76f51", size=11),
    ))
    fig.update_layout(
        title="Personal stability fit: your recall vs the fitted curve",
        xaxis_title="days since previous review",
        yaxis_title="recall (%)",
        yaxis_range=[0, 105],
        template="plotly_white", height=400,
    )
    return fig


def stat_cards(data: dict) -> str:
    ex = data["exam_summary"]
    cards = [
        ("Sessions", str(data["n_sessions"])),
        ("Planned objective", f"{data['objective']:.3f}"),
        ("Study minutes", f"{data['total_minutes']:,}"),
        ("Topics", str(len(data["topic_info"]))),
        ("Exams", str(len(ex))),
    ]
    inner = "".join(
        f"<div class='card'><div class='num'>{v}</div>"
        f"<div class='lbl'>{k}</div></div>" for k, v in cards)
    return f"<div class='cards'>{inner}</div>"


def main() -> None:
    cfg = st.load_config(os.path.join(ROOT, "exams.yaml"))
    data = dash.plan_data(cfg)
    rest = set(cfg["rest_days"])
    sweep_days = [d for d in range(data["horizon"]) if d not in rest]

    figs = {
        "tab-overview": [stat_cards(data),
                         fig_overview(data), fig_daily_load(data)],
        "tab-schedule": [fig_schedule_table(data)],
        "tab-curves": [fig_forgetting_curves(data)],
        "tab-whatif": [fig_whatif_sweep(cfg, sweep_days),
                       fig_whatif_detail(cfg)],
        "tab-fit": [fig_fit()],
    }

    plotlyjs = get_plotlyjs()
    body = []
    for tab_id, items in figs.items():
        divs = []
        for it in items:
            if isinstance(it, str):
                divs.append(it)
            else:
                divs.append(it.to_html(include_plotlyjs=False,
                                      full_html=False))
        body.append(f"<section id='{tab_id}' class='tab'>"
                    + "".join(divs) + "</section>")

    tabs = [
        ("tab-overview", "Overview"),
        ("tab-schedule", "Schedule"),
        ("tab-curves", "Forgetting curves"),
        ("tab-whatif", "What-if"),
        ("tab-fit", "Personal fit"),
    ]
    nav = "".join(
        f"<button class='tabbtn{' active' if i == 0 else ''}' "
        f"onclick=\"showTab('{tid}', this)\">{label}</button>"
        for i, (tid, label) in enumerate(tabs))

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>studyforge dashboard</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif;
        max-width: 1100px; margin: 0 auto; padding: 24px; color: #222; }}
h1 {{ margin-bottom: 4px; }}
.sub {{ color: #666; margin-bottom: 18px; }}
.tabbar {{ margin: 18px 0; }}
.tabbtn {{ padding: 10px 16px; margin-right: 6px; border: 1px solid #ccc;
           background: #f7f7f7; cursor: pointer; border-radius: 6px;
           font-size: 14px; }}
.tabbtn.active {{ background: #264653; color: white; border-color: #264653; }}
.tab {{ display: none; }}
.tab.active {{ display: block; }}
.cards {{ display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 18px; }}
.card {{ background: #f4f1de; border-radius: 8px; padding: 14px 20px;
         min-width: 130px; }}
.num {{ font-size: 26px; font-weight: 700; color: #264653; }}
.lbl {{ font-size: 12px; color: #666; }}
.note {{ color: #666; font-size: 13px; margin-top: 24px; }}
</style>
<script>{plotlyjs}</script>
<script>
function showTab(id, btn) {{
  document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelectorAll('.tabbtn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  window.dispatchEvent(new Event('resize'));
}}
document.addEventListener('DOMContentLoaded', () => {{
  document.querySelector('.tab').classList.add('active');
}});
</script>
</head>
<body>
<h1>studyforge dashboard</h1>
<p class="sub">Interactive tour of the planner: every number below comes
from the real engine on the included <code>exams.yaml</code>
(3 exams, 15 topics, 21 days, 120 minutes per day).</p>
<div class="tabbar">{nav}</div>
{"".join(body)}
<p class="note">Regenerate any time with
<code>python3 tools/make_dashboard.py</code>. Source:
<a href="https://github.com/FaiyazEnayet/studyforge">FaiyazEnayet/studyforge</a>.</p>
</body>
</html>"""

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(_portfolio_html(html))
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e6:.1f} MB)")



def _portfolio_html(raw):
    """Presentation-only transform of this run’s freshly calculated figures."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "portfolio_ui"))
    from renderer import render_html
    return render_html(root, 'studyforge', raw)[0]

if __name__ == "__main__":
    main()
