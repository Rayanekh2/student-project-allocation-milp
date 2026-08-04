"""
compare_penalty.py
================================================================
Demonstrate the track-mismatch penalty by sweeping lambda.

Because the real student track data has not yet been provided by the
administration, this script can run in two modes:

  * If data/student_tracks.csv has real tracks filled in, it uses them.
  * Otherwise it injects a TEST labelling (first 10 students = "Data
    Science") purely to demonstrate the mechanism, then restores the
    file to empty afterwards.

For each lambda it reports the rank distribution, satisfaction, and the
number of track mismatches in the final allocation, then plots them.

Outputs:
    output/penalty_comparison.png
    output/penalty_summary.png
================================================================
"""

import csv
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spa_milp import load_data, load_student_tracks, solve, TRACKS_FILE

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"
LAMBDAS = [0, 50, 500]
C_RANK = {1: "#27AE60", 2: "#2980B9", 3: "#C0392B"}


def read_rows():
    with open(TRACKS_FILE, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_rows(rows):
    with open(TRACKS_FILE, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["student_name", "email", "track"])
        w.writeheader(); w.writerows(rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    rows = read_rows()
    already_filled = any(r.get("track", "").strip() for r in rows)
    test_mode = not already_filled

    if test_mode:
        print("No real track data found — injecting TEST labels (10 × Data Science).")
        backup = [dict(r) for r in rows]
        for r in rows[:10]:
            r["track"] = "Data Science"
        write_rows(rows)

    tracks = load_student_tracks()
    results = []
    for lam in LAMBDAS:
        res = solve(data, criterion="minimax_weighted",
                    track_lambda=lam, student_tracks=tracks)
        results.append((lam, res))
        ts = res["track_stats"]
        print(f"λ={lam:>3}: R1={res['rank_counts'].get(1,0)} "
              f"R2={res['rank_counts'].get(2,0)} R3={res['rank_counts'].get(3,0)} "
              f"Sat={res['satisfaction_pct']:.1f}%  mismatch={ts['mismatch'] if ts else '-'}")

    if test_mode:
        write_rows(backup)
        print("Restored student_tracks.csv to empty.")

    # ── Chart: side-by-side rank distributions ─────────────────────────────────
    fig, axes = plt.subplots(1, len(results), figsize=(3.4*len(results), 4), sharey=True)
    for ax, (lam, res) in zip(axes, results):
        ranks = [1, 2, 3]
        counts = [res["rank_counts"].get(r, 0) for r in ranks]
        bars = ax.bar(ranks, counts, color=[C_RANK[r] for r in ranks], width=0.6,
                      edgecolor="white", linewidth=1.2)
        for b, c in zip(bars, counts):
            if c:
                ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.4, str(c),
                        ha="center", va="bottom", fontsize=11, fontweight="bold")
        ts = res["track_stats"]
        mm = ts["mismatch"] if ts else 0
        ax.set_title(f"λ = {lam}\n{res['satisfaction_pct']:.1f}% · {mm} mismatch",
                     fontsize=11, fontweight="bold")
        ax.set_xticks(ranks); ax.set_xlabel("Rank")
        ax.spines[["top", "right"]].set_visible(False); ax.set_ylim(0, 50)
    axes[0].set_ylabel("Students")
    fig.suptitle("Track Penalty Sweep (test labelling)", fontsize=14, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "penalty_comparison.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUTPUT_DIR / 'penalty_comparison.png'}")

    # ── Summary table image ─────────────────────────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(7, 2.6)); ax2.axis("off")
    headers = ["Metric"] + [f"λ={lam}" for lam, _ in results]
    table_rows = [
        ["Rank 1"] + [str(r["rank_counts"].get(1, 0)) for _, r in results],
        ["Rank 2"] + [str(r["rank_counts"].get(2, 0)) for _, r in results],
        ["Rank 3"] + [str(r["rank_counts"].get(3, 0)) for _, r in results],
        ["Satisfaction %"] + [f"{r['satisfaction_pct']:.1f}" for _, r in results],
        ["Track mismatches"] + [str(r["track_stats"]["mismatch"]) if r["track_stats"] else "-" for _, r in results],
    ]
    t = ax2.table(cellText=table_rows, colLabels=headers, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.7)
    for j in range(len(headers)):
        t[0, j].set_facecolor("#193764"); t[0, j].set_text_props(color="white", fontweight="bold")
    ax2.set_title("Track Penalty — Effect of λ", fontsize=13, fontweight="bold", pad=14)
    fig2.savefig(OUTPUT_DIR / "penalty_summary.png", dpi=150, bbox_inches="tight", facecolor="white")
    print(f"Saved → {OUTPUT_DIR / 'penalty_summary.png'}")


if __name__ == "__main__":
    main()
