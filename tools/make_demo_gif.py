"""Build docs/demo.gif: an animated forgetting-curve demo.

Picks the hardest topic in exams.yaml (Rotation, difficulty 4), runs the
real planner, then animates day by day as the planned reviews land: the
recall curve reshapes and stability grows after every review. Saved with
the matplotlib pillow writer. No install needed:

    python3 tools/make_demo_gif.py
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.animation import FuncAnimation  # noqa: E402

from studyforge import state as st  # noqa: E402
from studyforge import dashboard as dash  # noqa: E402

OUT = os.path.join(ROOT, "docs", "demo.gif")

NAME = "Rotation"


def main() -> None:
    cfg = st.load_config(os.path.join(ROOT, "exams.yaml"))
    data = dash.plan_data(cfg)
    # topic_info is keyed by normalized topic ID, not display name.
    info = data["topic_info"][st.topic_id(NAME)]
    exam_day = dash.topic_exam_day(info)
    horizon = exam_day
    planned = info["days"]

    plt.rcParams.update({'figure.facecolor':'#eeeade','axes.facecolor':'#f7f3e7','text.color':'#344c3c','axes.labelcolor':'#344c3c','xtick.color':'#344c3c','ytick.color':'#344c3c','axes.edgecolor':'#a7ad98','font.family':'DejaVu Serif'})
    fig, ax = plt.subplots(figsize=(9, 5.4))
    fig.suptitle('StudyForge · planned reviews and estimated recall', fontsize=17, y=.98)

    def draw(frame: int):
        ax.clear()
        done = [d for d in planned if d <= frame]
        traj = dash.recall_trajectory(NAME, done, info["S0"],
                                      exam_day, horizon)
        ax.plot(traj["xs"], [r * 100 for r in traj["ys"]],
                color="#264653", lw=2.5)
        if traj["review_marks"]:
            rx, ry = zip(*traj["review_marks"])
            ax.scatter(rx, [r * 100 for r in ry], color="#e76f51",
                       s=70, marker="D", zorder=5, label="reviews so far")
        ax.axvline(exam_day, color="#999", ls="--", lw=1)
        ax.text(exam_day - 0.2, 92, "exam", color="#68735b", fontsize=9, ha='right')
        ax.set_xlim(0, horizon)
        ax.set_ylim(0, 105)
        ax.set_xlabel("day")
        ax.set_ylabel("recall (%)")
        ax.set_title(f"{NAME}: day {frame} of {horizon}  |  "
                     f"stability {traj['S_final']:.2f} days  |  "
                     f"{len(done)} reviews done",
                     fontsize=11)
        if frame == 0:
            ax.legend(loc="upper right", fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='#d9ddcc',linewidth=.7)
        fig.tight_layout(rect=[0,0,1,.92])

    anim = FuncAnimation(fig, draw, frames=list(range(horizon + 1)),
                         interval=600, repeat=False)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    anim.save(OUT, writer="pillow", fps=2)
    print(f"wrote {OUT} ({os.path.getsize(OUT) / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
