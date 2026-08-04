"""
whatif.py
================================================================
Robustness and what-if analysis (project requirement §3.3).

Re-solves the allocation under small administrative changes and reports
the impact on fairness and satisfaction:

  Scenario A — Quota change : add one seat to the most oversubscribed
                              project and measure the gain.
  Scenario B — Fixed assignment : force a chosen student onto a chosen
                              project (administration override) and
                              measure the ripple effect.
  Scenario C — Preference change : a student drops their 1st choice and
                              re-ranks; measure the personal/global cost.

For each scenario we report Δrank-distribution and Δsatisfaction versus
the baseline, and a one-line interpretation.

Outputs:
    output/whatif_report.txt
    output/whatif_quota_sweep.png   satisfaction vs added seats
================================================================
"""

import copy
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from spa_milp import load_data, solve

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "output"


def dist(res):
    rc = res["rank_counts"]
    return f"{rc.get(1,0)}/{rc.get(2,0)}/{rc.get(3,0)}"


def most_oversubscribed(data):
    demand = defaultdict(int)
    for s in data["students"]:
        for p in s["choices"]:
            demand[p] += 1
    best, bd, bcap = None, -1, None
    for p, prj in data["projects"].items():
        _, maxq = prj["quota"]
        d = demand.get(p, 0)
        if d - maxq > bd:
            bd, best, bcap = d - maxq, p, maxq
    return best, demand[best], bcap


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_data()

    base = solve(data, criterion="minimax_weighted")
    base_sat = base["satisfaction_pct"]
    lines = []
    lines.append("WHAT-IF / SENSITIVITY ANALYSIS")
    lines.append("=" * 60)
    lines.append(f"Baseline: ranks {dist(base)}  satisfaction {base_sat:.1f}%\n")

    # ── Scenario A: add seats to the most oversubscribed project ───────────────
    target, dem, cap = most_oversubscribed(data)
    lines.append("SCENARIO A — Quota change")
    lines.append("-" * 60)
    lines.append(f"Most oversubscribed: \u201c{data['projects'][target]['name'][:45]}\u201d "
                 f"({dem} applicants, {cap} seats).")

    sweep_seats, sweep_sat = [0], [base_sat]
    for extra in (1, 2, 3):
        d2 = copy.deepcopy(data)
        mn, mx = d2["projects"][target]["quota"]
        d2["projects"][target]["quota"] = (mn, mx + extra)
        r = solve(d2, criterion="minimax_weighted")
        sweep_seats.append(extra)
        sweep_sat.append(r["satisfaction_pct"])
        lines.append(f"  +{extra} seat(s): ranks {dist(r)}  "
                     f"satisfaction {r['satisfaction_pct']:.1f}%  "
                     f"(Δ {r['satisfaction_pct']-base_sat:+.1f})")
    lines.append("  Interpretation: extra seats on the top bottleneck convert "
                 "rank-2/3 students to better ranks until demand is met.\n")

    # ── Scenario B: fixed assignment (admin override) ──────────────────────────
    lines.append("SCENARIO B — Fixed assignment (admin override)")
    lines.append("-" * 60)
    # pick a student who did NOT get their first choice and force their 2nd
    victim = next((a for a in base["assignments"] if a["rank"] and a["rank"] > 1), None)
    if victim:
        s = victim["student"]
        forced_p = s["choices"][0]  # force their actual 1st choice
        d2 = copy.deepcopy(data)
        # implement override by making this their only choice
        for st in d2["students"]:
            if st["id"] == s["id"]:
                st["choices"] = [forced_p]
                st["ranks"] = {forced_p: 1}
        r = solve(d2, criterion="minimax_weighted")
        lines.append(f"Force {s['name']} onto their 1st choice "
                     f"\u201c{data['projects'][forced_p]['name'][:40]}\u201d.")
        lines.append(f"  Result: ranks {dist(r)}  satisfaction {r['satisfaction_pct']:.1f}%  "
                     f"(Δ {r['satisfaction_pct']-base_sat:+.1f})")
        lines.append("  Interpretation: the override helps one student but can "
                     "displace others competing for the same project.\n")

    # ── Scenario C: preference change ──────────────────────────────────────────
    lines.append("SCENARIO C — Preference change")
    lines.append("-" * 60)
    # a student who got rank 1 drops it; measure their new outcome
    happy = next((a for a in base["assignments"] if a["rank"] == 1
                  and len(a["student"]["choices"]) >= 2), None)
    if happy:
        s = happy["student"]
        d2 = copy.deepcopy(data)
        for st in d2["students"]:
            if st["id"] == s["id"]:
                new_choices = st["choices"][1:]            # drop first choice
                st["choices"] = new_choices
                st["ranks"] = {p: i + 1 for i, p in enumerate(new_choices)}
        r = solve(d2, criterion="minimax_weighted")
        new_a = next((a for a in r["assignments"] if a["student"]["id"] == s["id"]), None)
        lines.append(f"{s['name']} drops their 1st choice.")
        lines.append(f"  Their new assignment rank: {new_a['rank'] if new_a else '-'}")
        lines.append(f"  Global: ranks {dist(r)}  satisfaction {r['satisfaction_pct']:.1f}%  "
                     f"(Δ {r['satisfaction_pct']-base_sat:+.1f})")
        lines.append("  Interpretation: dropping a popular first choice usually "
                     "costs the individual a rank but frees a seat for others.\n")

    # ── write report ────────────────────────────────────────────────────────────
    report = OUTPUT_DIR / "whatif_report.txt"
    with open(report, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))
    print(f"\nSaved → {report}")

    # ── quota sweep chart ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(sweep_seats, sweep_sat, "o-", color="#193764", linewidth=2, markersize=8)
    for x, y in zip(sweep_seats, sweep_sat):
        ax.text(x, y + 0.05, f"{y:.1f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    ax.set_xlabel("Extra seats added to top bottleneck project")
    ax.set_ylabel("Satisfaction %")
    ax.set_title("What-If: Satisfaction vs Added Capacity",
                 fontsize=13, fontweight="bold", pad=16)
    ax.set_xticks(sweep_seats)
    ax.margins(y=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "whatif_quota_sweep.png", dpi=150, facecolor="white")
    print(f"Saved → {OUTPUT_DIR / 'whatif_quota_sweep.png'}")


if __name__ == "__main__":
    main()
