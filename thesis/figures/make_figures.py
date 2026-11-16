"""Make the data figures of the thesis:  python3 thesis/figures/make_figures.py
Reads the simulation results from neurodrone/results/series/final_20261115.csv;
VLM numbers are the full-set results of Report_for_Thesis.md, sections 10.1, 10.2 and 10.4."""
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Circle, Rectangle, Wedge, FancyArrowPatch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "savefig.dpi": 200,
})


def vlm_size():
    # (label, RefDrone acc %, median latency s, file GB, group)
    rows = [
        ("2B q4", 43.9, 0.38, 1.9, "q4"), ("4B q4", 69.3, 0.57, 3.3, "q4"), ("8B q4", 63.2, 0.84, 6.2, "q4"),
        ("2B q8", 50.9, 0.40, 2.7, "q8"), ("4B q8", 69.1, 0.60, 5.1, "q8"), ("8B q8", 63.9, 0.90, 9.9, "q8"),
        ("2B bf16", 51.5, 0.55, 4.3, "bf16"), ("4B bf16", 69.4, 1.01, 8.9, "bf16"), ("8B bf16", 63.9, 1.65, 18, "bf16"),
        ("30B-A3B q4", 69.4, 0.83, 19.6, "large"), ("32B q4", 68.5, 2.5, 20.9, "large"),
    ]
    colors = {"q4": BLUE, "q8": AQUA, "bf16": ORANGE, "large": GREY}
    names = {"q4": "4 bit (q4_K_M)", "q8": "8 bit (q8_0)", "bf16": "16 bit (bf16)", "large": "large, q4 (do not fit 8 GB)"}
    fig, ax = plt.subplots(figsize=(6.3, 3.6))
    for g in ["q4", "q8", "bf16", "large"]:
        sel = [r for r in rows if r[4] == g]
        ax.scatter([r[2] for r in sel], [r[1] for r in sel], s=[25 + 9 * r[3] for r in sel],
                   color=colors[g], alpha=0.85, edgecolor="white", linewidth=1.5, label=names[g], zorder=3)
    offs = {"2B q4": (6, -10), "4B q4": (-8, 9), "8B q4": (6, -12), "2B q8": (6, 4), "4B q8": (6, -11),
            "8B q8": (6, 3), "2B bf16": (6, -2), "4B bf16": (6, -12), "8B bf16": (6, -12), "30B-A3B q4": (-8, 9),
            "32B q4": (-10, 8)}
    for lab, acc, lat, _, _g in rows:
        dx, dy = offs[lab]
        ax.annotate(lab, (lat, acc), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color=INK2,
                    ha="right" if dx < 0 else "left")
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1, 2, 3]); ax.set_xticklabels(["0.3", "0.5", "1", "2", "3"])
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.set_xlabel("Median latency per image on the workstation GPU (s, log scale)")
    ax.set_ylabel("RefDrone accuracy, IoU ≥ 0.5 (%)")
    ax.set_ylim(38, 76); ax.set_xlim(0.3, 3.4)
    leg = ax.legend(frameon=False, fontsize=7.5, loc="lower right")
    for h in leg.legend_handles:
        h.set_sizes([40])
    ax.set_title("Qwen3-VL Instruct: size and quantisation", fontsize=10, color=INK, loc="left")
    fig.tight_layout(); fig.savefig(HERE / "fig_vlm_size.png"); plt.close(fig)


POLES = [(6.0, 0.2), (4.29, 2.92), (5.06, -3.13)]  # ENU x, y; radius 0.15 m


def worlds():
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    ax.add_patch(Wedge((0, 0), 15.5, -34.5, 34.5, color=BLUE, alpha=0.07, zorder=1))
    ax.text(15.0, -9.8, "camera view at the start (69°)", color=BLUE, fontsize=7.5, ha="right")
    objs = [("red box", 12, 0, "#e34948", "box", 1.0), ("blue barrel", 8, 5, BLUE, "barrel", 0.4),
            ("green box", 9, -6, "#008300", "box", 1.2), ("white box (distractor)", 14, 6, "#bdbcb6", "box", 1.0)]
    for name, x, y, c, kind, size in objs:
        if kind == "box":
            ax.add_patch(Rectangle((x - size / 2, y - size / 2), size, size, color=c, zorder=3))
        else:
            ax.add_patch(Circle((x, y), size, color=c, zorder=3))
        ax.text(x, y - size / 2 - 0.9, name, ha="center", fontsize=8, color=INK)
        ax.plot([0, x], [0, y], ls=":", color=GREY, lw=0.8, zorder=2)
    for (x, y) in POLES:
        ax.add_patch(Circle((x, y), 0.15, color="#eda100", ec=INK2, lw=0.6, zorder=4))
        ax.add_patch(Circle((x, y), 0.15 + 0.42, fill=False, ls="--", ec="#eda100", lw=0.7, zorder=4))
    ax.text(6.2, 1.0, "poles (nd_obstacles only)", fontsize=7.5, color="#a87300")
    ax.add_patch(FancyArrowPatch((0, 0), (1.6, 0), arrowstyle="-|>", mutation_scale=12, color=INK, zorder=5))
    ax.add_patch(Circle((0, 0), 0.42, fill=False, ec=INK, lw=0.8, zorder=5))
    ax.text(0, -1.1, "start", ha="center", fontsize=8)
    ax.set_aspect("equal"); ax.set_xlim(-1.5, 16); ax.set_ylim(-10.5, 9)
    ax.set_xlabel("x (m), forward at the start"); ax.set_ylabel("y (m), left")
    fig.tight_layout(); fig.savefig(HERE / "fig_worlds.png"); plt.close(fig)


def sim_results():
    rows = list(csv.DictReader(open(REPO / "neurodrone/results/series/final_20261115.csv")))
    targets = ["the red box", "the green box", "the blue barrel"]
    labels = ["red box", "green box", "blue barrel"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 2.9))
    w = 0.36
    for k, (search, col) in enumerate([("sweep", BLUE), ("active", ORANGE)]):
        succ, mean_n, all_n = [], [], []
        for t in targets:
            rr = [r for r in rows if r["instruction"] == t and r["search"] == search]
            ok = [r for r in rr if r["success"] == "True"]
            succ.append(len(ok))
            ns = [int(r["vlm_n"]) for r in ok]
            all_n.append(ns); mean_n.append(sum(ns) / len(ns))
        xs = [i + (k - 0.5) * w for i in range(3)]
        a1.bar(xs, succ, w - 0.04, color=col, label=search, zorder=3)
        for x, s in zip(xs, succ):
            a1.text(x, s + 0.15, f"{s}/10", ha="center", fontsize=7.5, color=INK)
        a2.bar(xs, mean_n, w - 0.04, color=col, alpha=0.35, zorder=2)
        for x, ns in zip(xs, all_n):
            jit = [((j % 5) - 2) * 0.03 for j in range(len(ns))]
            a2.scatter([x + d for d in jit], ns, s=12, color=col, zorder=3, edgecolor="white", linewidth=0.5)
    for a in (a1, a2):
        a.set_xticks(range(3)); a.set_xticklabels(labels); a.grid(axis="x", visible=False)
    a1.set_ylim(0, 11.5); a1.set_ylabel("successful runs (of 10)"); a1.set_title("(a) success", fontsize=9, loc="left")
    a2.set_ylabel("VLM answers per successful run"); a2.set_ylim(0, 7)
    a2.set_title("(b) VLM answers (bar = mean)", fontsize=9, loc="left")
    fig.legend(*a1.get_legend_handles_labels(), frameon=False, fontsize=8, loc="upper center", ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.92)); fig.savefig(HERE / "fig_sim_results.png"); plt.close(fig)


def failures():
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.6, 3.0))
    # (a) occluder inside the box
    a1.add_patch(Wedge((0, 0), 11, -12, 12, color=BLUE, alpha=0.10))
    a1.add_patch(Rectangle((9.5, -0.6), 1.2, 1.2, color="#008300"))
    a1.add_patch(Circle((5, 0.15), 0.15, color="#eda100", ec=INK2, lw=0.6))
    a1.plot([0, 10.1], [0, 0], ls=":", color=GREY)
    a1.add_patch(Circle((3.5, 0.1), 0.12, color=ORANGE))
    a1.annotate("goal from the pole's depth", (3.5, 0.1), xytext=(0.5, 2.2), fontsize=7.5, color=INK,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    a1.annotate("target", (10.1, 0.6), xytext=(8.3, 2.2), fontsize=7.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    a1.annotate("pole inside the box", (5, 0.3), xytext=(4.2, -2.6), fontsize=7.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    a1.add_patch(Circle((0, 0), 0.42, fill=False, ec=INK))
    a1.set_title("(a) occlusion inside the box", fontsize=9, loc="left")
    # (b) corner cut after go-around
    a2.add_patch(Circle((6, 0), 0.15, color="#eda100", ec=INK2, lw=0.6))
    path_x = [0, 3.5, 4.6, 5.6, 6.3, 7.4, 9.5]
    path_y = [0, 0, -0.5, -0.95, -0.62, -0.2, 0]
    a2.plot(path_x, path_y, color=BLUE, lw=2, label="flown path")
    a2.plot([0, 9.5], [0, 0], ls=":", color=GREY, lw=0.8)
    a2.add_patch(Wedge((6.3, -0.62), 3.5, 5 - 36.5, 5 + 36.5, color=BLUE, alpha=0.10))
    a2.text(8.2, -2.4, "camera view", color=BLUE, fontsize=7.5)
    a2.add_patch(Circle((6.3, -0.62), 0.42, fill=False, ec=ORANGE, lw=1))
    a2.annotate("side-step", (4.6, -0.5), xytext=(2.0, -2.2), fontsize=7.5, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    a2.annotate("pole out of view;\npropellers pass close", (6.3, -0.2), xytext=(5.8, 1.4), fontsize=7.5,
                arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    a2.set_title("(b) no memory after the side-step", fontsize=9, loc="left")
    for a in (a1, a2):
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([]); a.grid(False)
        for s in a.spines.values():
            s.set_visible(False)
    a1.set_xlim(-0.8, 11.5); a1.set_ylim(-3.2, 3.2)
    a2.set_xlim(-0.8, 10); a2.set_ylim(-3.2, 3.2)
    fig.tight_layout(); fig.savefig(HERE / "fig_failures.png"); plt.close(fig)


if __name__ == "__main__":
    vlm_size(); worlds(); sim_results(); failures()
    print("figures written to", HERE)
