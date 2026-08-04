"""
spa_milp.py
================================================================
Core MILP engine for Student–Project Allocation (Spring 2026).

This single module is the *engine* used by every experiment.  It exposes
one main function, `solve()`, parameterised by:

    criterion           which objective to optimise
    track_lambda        weight of the track-mismatch soft penalty (0 = off)
    enforce_stability   add Chiarandini's one-sided stability constraints
    enforce_partners    keep partner groups together
    enforce_lower_bounds use the y_j open/close variable

Decision variables
-------------------
    x[s,p] ∈ {0,1}   student s assigned to project p
    y[p]   ∈ {0,1}   project p is "open"
    z      ∈ ℤ≥1     the worst (largest) rank assigned to any student

Constraints (always on unless noted)
------------------------------------
    C1  each student → exactly one project
    C2  Σ x ≤ max_q · y[p]              (upper capacity, gated by y)
    C3  Σ x ≥ min_q · y[p]              (lower capacity, gated by y)   [lower bounds]
    C4  z ≥ Σ rank · x   per student    (z tracks worst rank)
    C5  x[s1,p] = x[s2,p]               (partners share a project)     [partners]
    C6  stability: a student at rank r implies every better-ranked
        project of theirs is full                                      [stability]

Objectives (`criterion`)
------------------------
    "minimax_weighted"   min 1000·z + Σ rank·x          (baseline, lexicographic)
    "minimax"            min z                          (worst rank only)
    "weighted"           min Σ rank·x                   (pure utilitarian)
    "exponential"        min Σ 2^(4-rank)·x             (favour first choice)

Track penalty (soft, added to whichever objective is chosen)
    + track_lambda · Σ mismatch(s,p) · x[s,p]
    where mismatch = 1 when the student's track ∉ the project's target tracks.
================================================================
"""

import csv
import pickle
from collections import defaultdict
from pathlib import Path

import pulp
from pulp import LpBinary, LpInteger, LpMinimize, LpProblem, LpVariable, lpSum

DATA_DIR    = Path(__file__).resolve().parent.parent / "data"
PKL_FILE    = DATA_DIR / "parsed_data.pkl"
TRACKS_FILE = DATA_DIR / "student_tracks.csv"


# ── Loaders ───────────────────────────────────────────────────────────────────
def load_data(path: Path = PKL_FILE) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


def load_student_tracks(path: Path = TRACKS_FILE) -> dict | None:
    """
    Return {email_lower: track}.  None if file absent, {} if present-but-empty.
    Email is the key because names in the raw data are inconsistent.
    """
    if not path.exists():
        return None
    tracks = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            email = row.get("email", "").strip().lower()
            track = row.get("track", "").strip()
            if email and track:
                tracks[email] = track
    return tracks


# ── Objective term builders ──────────────────────────────────────────────────
def _rank_weight(rank: int, scheme: str) -> float:
    """
    Weight applied to an assignment at a given rank, per objective scheme.
    Both schemes are *costs* in a minimisation problem, so a better (lower)
    rank must carry a lower weight.
    """
    if scheme == "weighted":
        return rank                      # 1, 2, 3   (linear cost)
    if scheme == "exponential":
        return 2 ** rank                 # rank1→2, rank2→4, rank3→8
                                         # worse ranks penalised exponentially,
                                         # so the solver fights hardest to keep
                                         # students at rank 1
    raise ValueError(scheme)


# ── Main solve ────────────────────────────────────────────────────────────────
def solve(data: dict,
          criterion: str = "minimax_weighted",
          track_lambda: float = 0.0,
          student_tracks: dict | None = None,
          enforce_stability: bool = False,
          enforce_partners: bool = True,
          enforce_lower_bounds: bool = True,
          verbose: bool = False):
    """Build and solve one MILP.  Returns a results dict (see bottom)."""
    projects       = data["projects"]
    students       = data["students"]
    partner_groups = data["partner_groups"]

    prob = LpProblem("SPA_Spring2026", LpMinimize)

    # demand per project (for pre-solve closure)
    demand = defaultdict(int)
    for s in students:
        for p in s["choices"]:
            demand[p] += 1

    # ── Variables ────────────────────────────────────────────────────────────
    x = {(s["id"], p): LpVariable(f"x_{s['id']}_{p}", cat=LpBinary)
         for s in students for p in s["choices"]}

    active_projects = {p for s in students for p in s["choices"]}
    y = {p: LpVariable(f"y_{p}", cat=LpBinary) for p in active_projects}

    z = LpVariable("z", lowBound=1, cat=LpInteger)

    # ── Pre-solve: force-close projects that cannot reach their minimum ────────
    forced_closed = set()
    if enforce_lower_bounds:
        for p in active_projects:
            min_q, _ = projects[p]["quota"]
            if min_q > 0 and demand[p] < min_q:
                prob += y[p] == 0
                forced_closed.add(p)

    # ── C1: one project per student ────────────────────────────────────────────
    for s in students:
        prob += lpSum(x[(s["id"], p)] for p in s["choices"]) == 1

    # ── C2 / C3: capacity, gated by y ──────────────────────────────────────────
    for p in active_projects:
        min_q, max_q = projects[p]["quota"]
        applicants = [s for s in students if p in s["choices"]]
        total = lpSum(x[(s["id"], p)] for s in applicants)
        prob += total <= max_q * y[p]
        if enforce_lower_bounds and min_q > 0:
            prob += total >= min_q * y[p]

    # ── C4: worst-rank tracking (needed by minimax objectives) ─────────────────
    for s in students:
        prob += z >= lpSum(s["ranks"][p] * x[(s["id"], p)] for p in s["choices"])

    # ── C5: partner groups share a project ─────────────────────────────────────
    #
    # A group is only constrained if its members share at least one project in
    # common.  If they have NO common choice (e.g. they listed entirely
    # different projects), keeping them together is impossible — forcing it
    # would make the whole model infeasible.  In that case we skip the group
    # and let each member be placed independently (reported afterwards as a
    # "split" group).
    if enforce_partners:
        for group in partner_groups:
            members = [s for s in students if s["id"] in group]
            if len(members) < 2:
                continue
            common = set(members[0]["choices"])
            for m in members[1:]:
                common &= set(m["choices"])
            if not common:
                continue                      # cannot keep them together — skip
            # outside the common set, none of them may be placed
            for m in members:
                for p in m["choices"]:
                    if p not in common:
                        prob += x[(m["id"], p)] == 0
            anchor = members[0]
            for m in members[1:]:
                for p in common:
                    prob += x[(anchor["id"], p)] == x[(m["id"], p)]

    # ── C6: one-sided stability (Chiarandini §5.6) ─────────────────────────────
    #
    # A student s placed on project p is "unstable" if some project p' that s
    # ranks strictly higher still has free capacity.  We forbid this with a
    # linear constraint linking s's assignment to the fill level of every
    # better-ranked project.
    #
    # For every (s, p') with rank(s,p') = r':  if p' is not full, then s must be
    # assigned to a project ranked r' or better.  We encode "p' has free space"
    # with a binary full[p'] and tie it to occupancy.
    #
    # IMPORTANT — partnered students are exempted from stability.  This mirrors a
    # classical result (Roth, on couples in the medical match): coupling
    # constraints can make a stable matching impossible, because a paired student
    # may be "blocked" by a better project that has room but that their partner
    # did not rank.  Enforcing both at once is infeasible on our data, so we keep
    # partner togetherness (a hard institutional requirement) and relax stability
    # for the partnered students only.  Non-partnered students remain stable.
    if enforce_stability:
        partnered_ids = {sid for g in partner_groups if len(g) > 1 for sid in g}
        occupancy = {p: lpSum(x[(s["id"], p)] for s in students if p in s["choices"])
                     for p in active_projects}
        full = {p: LpVariable(f"full_{p}", cat=LpBinary) for p in active_projects}
        for p in active_projects:
            _, max_q = projects[p]["quota"]
            prob += occupancy[p] >= max_q * full[p]
        for s in students:
            if s["id"] in partnered_ids:
                continue                       # partnered students exempt (see note)
            for p_better in s["choices"]:
                r_better = s["ranks"][p_better]
                better_or_equal = [q for q in s["choices"] if s["ranks"][q] <= r_better]
                prob += (lpSum(x[(s["id"], q)] for q in better_or_equal)
                         >= 1 - full[p_better])

    # ── Objective ──────────────────────────────────────────────────────────────
    total_rank = lpSum(s["ranks"][p] * x[(s["id"], p)]
                       for s in students for p in s["choices"])

    if criterion == "minimax_weighted":
        objective = 1000 * z + total_rank
    elif criterion == "minimax":
        objective = z
    elif criterion == "weighted":
        objective = lpSum(_rank_weight(s["ranks"][p], "weighted") * x[(s["id"], p)]
                          for s in students for p in s["choices"])
    elif criterion == "exponential":
        objective = lpSum(_rank_weight(s["ranks"][p], "exponential") * x[(s["id"], p)]
                          for s in students for p in s["choices"])
    else:
        raise ValueError(f"Unknown criterion: {criterion}")

    # track-mismatch soft penalty (added to any criterion)
    n_mismatch_pairs = 0
    if student_tracks and track_lambda > 0:
        mismatch_pairs = []
        for s in students:
            track = student_tracks.get(s["email"].lower())
            if not track:
                continue
            for p in s["choices"]:
                if track not in projects[p]["tracks"]:
                    mismatch_pairs.append((s["id"], p))
        n_mismatch_pairs = len(mismatch_pairs)
        objective = objective + track_lambda * lpSum(x[(sid, p)] for sid, p in mismatch_pairs)

    # Tiny tie-breaker (weight 1e-4): among equally-optimal allocations prefer
    # opening FEWER projects.  Administratively cleaner (fewer supervisors for
    # the same satisfaction) and makes the result deterministic.  The weight is
    # far smaller than any rank cost so it never overrides the chosen criterion.
    objective = objective + 1e-4 * lpSum(y[p] for p in active_projects)

    prob += objective

    # ── Solve ──────────────────────────────────────────────────────────────────
    prob.solve(pulp.PULP_CBC_CMD(msg=verbose))

    if prob.status != 1:
        return {"status": pulp.LpStatus[prob.status], "feasible": False}

    return _extract(data, x, y, z, total_rank, criterion, track_lambda,
                    student_tracks, forced_closed, n_mismatch_pairs)


# ── Result extraction ─────────────────────────────────────────────────────────
def _extract(data, x, y, z, total_rank, criterion, track_lambda,
             student_tracks, forced_closed, n_mismatch_pairs) -> dict:
    projects, students, partner_groups = (
        data["projects"], data["students"], data["partner_groups"])

    assignments, rank_counts = [], defaultdict(int)
    for s in students:
        chosen, rank = None, None
        for p in s["choices"]:
            v = x[(s["id"], p)].value()
            if v is not None and v > 0.5:
                chosen, rank = p, s["ranks"][p]
                break
        assignments.append({"student": s, "project": chosen, "rank": rank})
        if rank:
            rank_counts[rank] += 1

    n = len(students)
    total = total_rank.value()
    best, worst = n * 1, n * 3
    satisfaction = (1 - (total - best) / (worst - best)) * 100 if worst > best else 100.0

    # opened projects + occupancy
    opened = {}
    for p, var in y.items():
        if var.value() and var.value() > 0.5:
            opened[p] = sum(1 for a in assignments if a["project"] == p)

    # partner satisfaction
    partner_status = []
    for g in partner_groups:
        projs = {a["project"] for a in assignments if a["student"]["id"] in g}
        partner_status.append({
            "members": [s["name"] for s in students if s["id"] in g],
            "satisfied": len(projs) == 1 and None not in projs,
            "projects": projs,
        })

    # track stats
    track_stats = None
    if student_tracks:
        match = mismatch = unknown = 0
        for a in assignments:
            if a["project"] is None:
                continue
            t = student_tracks.get(a["student"]["email"].lower(), "")
            if not t:
                unknown += 1
            elif t in projects[a["project"]]["tracks"]:
                match += 1
            else:
                mismatch += 1
        track_stats = {"match": match, "mismatch": mismatch, "unknown": unknown}

    # stability audit (independent of whether it was enforced)
    unstable = audit_stability(assignments, projects, students)

    return {
        "status": "Optimal", "feasible": True,
        "criterion": criterion, "track_lambda": track_lambda,
        "assignments": assignments,
        "rank_counts": dict(rank_counts),
        "satisfaction_pct": satisfaction,
        "total_rank": total, "worst_rank": int(z.value()),
        "opened": opened, "n_opened": len(opened),
        "forced_closed": forced_closed,
        "partner_status": partner_status,
        "n_partners_ok": sum(1 for p in partner_status if p["satisfied"]),
        "track_stats": track_stats,
        "n_mismatch_pairs": n_mismatch_pairs,
        "unstable": unstable,
    }


def audit_stability(assignments, projects, students) -> list:
    """
    Post-hoc check: list students who could see a strictly-preferred project
    that still has free capacity (a 'blocking' situation).
    """
    occupancy = defaultdict(int)
    for a in assignments:
        if a["project"]:
            occupancy[a["project"]] += 1

    sid_to_student = {s["id"]: s for s in students}
    unstable = []
    for a in assignments:
        s, p, r = a["student"], a["project"], a["rank"]
        if p is None:
            continue
        for q in s["choices"]:
            if s["ranks"][q] < r:                       # q is strictly preferred
                _, max_q = projects[q]["quota"]
                if occupancy[q] < max_q:                # and q has room
                    unstable.append({
                        "student": s["name"],
                        "got": projects[p]["name"], "got_rank": r,
                        "prefers": projects[q]["name"], "prefers_rank": s["ranks"][q],
                    })
                    break
    return unstable
