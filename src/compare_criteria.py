"""
compare_criteria.py
================================================================
Run the MILP under five different fairness criteria on the SAME data
and constraints, then tabulate and plot the results.

This reproduces the spirit of Chiarandini et al. (2019) Table 2:
isolate the effect of the *objective* by holding everything else fixed.

    1. minimax_weighted   baseline (lexicographic: worst rank, then total)
    2. minimax            worst rank only
    3. weighted           pure utilitarian (Σ rank)
    4. exponential        first-choice favouring weights
    5. stability          baseline objective + hard stability constraints

Outputs:
    output/criteria_comparison.png   side-by-side rank distributions
    output/criteria_summary.png      summary table as an image
    output/criteria_results.pkl      raw results for the report
================================================================
"""

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spa_milp import load_data, solve

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"

RUNS = [
    ("Minimax+W",   dict(criterion="minimax_weighted")),
    ("Minimax",     dict(criterion="minimax")),
    ("Weighted",    dict(criterion="weighted")),
    ("Exponential", dict(criterion="exponential")),
    ("Stable",      dict(criterion="minimax_weighted", enforce_stability=True)),
]

COLORS = {1: "#27AE60", 2: "#2980B9", 3: "#C0392B"}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    results = []
    for label, kwargs in RUNS:
        print(f"Running: {label} ...")
        res = solve(data, **kwargs)
        if not res["feasible"]:
            print(f"  {label}: INFEASIBLE")
            continue
        results.append((label, res))
        rc = res["rank_counts"]
        n_unstable = len(res["unstable"])
        stable_txt = "yes" if n_unstable == 0 else f"{n_unstable} unstable"
        print(f"  Rank1={rc.get(1,0)} Rank2={rc.get(2,0)} Rank3={rc.get(3,0)} "
              f"Sat={res['satisfaction_pct']:.1f}% Stable={stable_txt}")

    # ── Console table ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  {'Metric':<18}" + "".join(f"{lbl:>12}" for lbl, _ in results))
    print("=" * 70)
    for rank in (1, 2, 3):
        print(f"  {'Rank ' + str(rank):<18}" +
              "".join(f"{res['rank_counts'].get(rank,0):>12}" for _, res in results))
    print(f"  {'Satisfaction %':<18}" +
          "".join(f"{res['satisfaction_pct']:>11.1f}" for _, res in results))
    print(f"  {'Worst rank':<18}" +
          "".join(f"{res['worst_rank']:>12}" for _, res in results))
    print(f"  {'Unstable':<18}" +
          "".join(f"{len(res['unstable']):>12}" for _, res in results))
    print("=" * 70)

    # ── Chart 1: side-by-side rank distributions ───────────────────────────────
    fig, axes = plt.subplots(1, len(results), figsize=(3.2 * len(results), 4), sharey=True)
    for ax, (label, res) in zip(axes, results):
        ranks = [1, 2, 3]
        counts = [res["rank_counts"].get(r, 0) for r in ranks]
        bars = ax.bar(ranks, counts, color=[COLORS[r] for r in ranks],
                      width=0.65, edgecolor="white", linewidth=1.2)
        for b, c in zip(bars, counts):
            if c:
                ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.4,
                        str(c), ha="center", va="bottom", fontsize=11, fontweight="bold")
        ax.set_title(f"{label}\n{res['satisfaction_pct']:.1f}%", fontsize=11, fontweight="bold")
        ax.set_xticks(ranks)
        ax.set_xlabel("Rank")
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(0, 50)
    axes[0].set_ylabel("Students")
    fig.suptitle("Rank Distribution by Fairness Criterion", fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "criteria_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUTPUT_DIR / 'criteria_comparison.png'}")

    # ── Chart 2: summary table image ────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 3))
    ax2.axis("off")
    headers = ["Metric"] + [lbl for lbl, _ in results]
    rows = []
    for rank in (1, 2, 3):
        rows.append([f"Rank {rank}"] + [str(res["rank_counts"].get(rank, 0)) for _, res in results])
    rows.append(["Satisfaction %"] + [f"{res['satisfaction_pct']:.1f}" for _, res in results])
    rows.append(["Worst rank"] + [str(res["worst_rank"]) for _, res in results])
    rows.append(["Unstable students"] + [str(len(res["unstable"])) for _, res in results])

    table = ax2.table(cellText=rows, colLabels=headers, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.7)
    for j in range(len(headers)):
        table[0, j].set_facecolor("#193764")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(len(rows)):
        for j in range(len(headers)):
            table[i + 1, j].set_facecolor("#F5F5F5" if i % 2 == 0 else "white")
    ax2.set_title("Five Fairness Criteria — Comparison", fontsize=14, fontweight="bold", pad=16)
    fig2.savefig(OUTPUT_DIR / "criteria_summary.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUTPUT_DIR / 'criteria_summary.png'}")

    # ── Persist ─────────────────────────────────────────────────────────────────
    with open(OUTPUT_DIR / "criteria_results.pkl", "wb") as f:
        pickle.dump({lbl: res for lbl, res in results}, f)
    print(f"Saved → {OUTPUT_DIR / 'criteria_results.pkl'}")


if __name__ == "__main__":
    main()
