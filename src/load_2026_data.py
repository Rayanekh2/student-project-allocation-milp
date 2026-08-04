"""
load_2026_data.py
================================================================
Parse the Spring 2026 project/student raw files into a clean pickle
that all downstream MILP scripts consume.

Input :  data/Spring 2026 Project Proposals-79_records-20260313_1124.ods
         data/Spring 2026 Student Project Selections.xlsx
Output:  data/parsed_data.pkl

Pickle schema
-------------
projects        dict[proj_id, {name, supervisor, tracks, num_groups,
                               max_per_group, quota=(min,max)}]
students        list[ {id, name, email, choices=[proj_id,...],
                       ranks={proj_id: rank}, partner_group} ]
partner_groups  list[ [student_id, ...] ]
stats           {demand, oversubscribed, zero_demand}

Why a pickle?  It separates *parsing* (slow, messy, done once) from
*optimisation* (fast, run many times with different objectives).  Every
experiment loads the identical, validated dataset — so any difference in
results is attributable to the model, never to the parsing.
================================================================
"""

import re
import pickle
import unicodedata
from collections import defaultdict
from pathlib import Path

import pandas as pd

# ── Paths ───────────────────────────────────────────────────────────────────
DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
PROJECTS_FILE = DATA_DIR / "Spring 2026 Project Proposals-79_records-20260313_1124.ods"
STUDENTS_FILE = DATA_DIR / "Spring 2026 Student Project Selections.xlsx"
OUTPUT_FILE   = DATA_DIR / "parsed_data.pkl"

# ── Column names in the raw files ────────────────────────────────────────────
COL_TRACKS  = "Tracks: Select the programs the project is aimed at"
COL_GROUPS  = "Number of groups possible for this project (MAX 3)"
COL_MAXPG   = "Maximum number of students PER GROUP"
COL_SUPER   = "Name of Main Supervisor"
COL_ASSIGN  = "Name of student/s already assigned to this project (optional)"

# The partner column header is long; we match it by prefix to stay robust.
PARTNER_PREFIX = "Are you working with a classmate"

# Known typo in the source data.
TRACK_TYPO_FIX = {"Date Science": "Data Science"}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _norm_title(s: str) -> str:
    """Normalise a title for matching: unicode-NFKC, lowercase, single spaces."""
    s = unicodedata.normalize("NFKC", s)
    return re.sub(r"\s+", " ", s.strip().lower())


def parse_max_per_group(val) -> int:
    """Parse 2, 2.0, '2', or '1 or 2' -> the maximum integer."""
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return int(val)
    nums = re.findall(r"\d+", str(val))
    if nums:
        return int(nums[-1])          # "1 or 2" -> 2
    raise ValueError(f"Cannot parse max-per-group value: {val!r}")


def parse_choice_title(raw: str) -> str | None:
    """
    Student choices look like '42. Supervisor Name - Project Title'.
    Strip the leading number and supervisor, return the title (or None).
    Tolerates a tab or double-space before the ' - '.
    """
    s = re.sub(r"^\d+\.\s*", "", raw.strip())     # remove "42. "
    m = re.search(r"\s+-\s+", s)                  # find " - "
    if m:
        return s[m.end():].strip()
    return None


def _name_tokens(name: str) -> set[str]:
    """Lowercase alphabetic tokens of a name, for fuzzy partner matching."""
    return set(re.sub(r"[^a-z\s]", "", name.lower()).split())


def find_student_by_name(raw: str, students: list) -> dict | None:
    """Match a free-text partner name to a student via token overlap."""
    query = _name_tokens(raw)
    if not query:
        return None
    best, best_score = None, 0
    for s in students:
        score = len(query & _name_tokens(s["name"]))
        if score > best_score:
            best_score, best = score, s
    return best if best_score >= 1 else None


def parse_partner_raw(raw: str) -> list[str]:
    """Extract partner name(s) from the messy free-text partner field."""
    if not raw or raw.strip().upper() in ("NO", "NAN", "N/A", "NONE", ""):
        return []
    s = re.sub(r"^(?:yes|no)\s*[,\-:]\s*", "", raw.strip(), flags=re.IGNORECASE).strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [p for p in parts if p.upper() not in ("YES", "NO")]


# ── Load projects ─────────────────────────────────────────────────────────────
def load_projects(path: Path) -> tuple[dict, dict]:
    df = pd.read_excel(path, engine="odf")
    projects, title_to_id = {}, {}

    for i, row in df.iterrows():
        proj_id = f"2026SPRING-{i + 1:02d}"
        title   = str(row["Title"]).strip()

        # Tracks (fix the known typo, split on ##)
        raw_tracks = str(row[COL_TRACKS]) if pd.notna(row[COL_TRACKS]) else ""
        tracks = []
        for t in raw_tracks.split("##"):
            t = TRACK_TYPO_FIX.get(t.strip(), t.strip())
            if t:
                tracks.append(t)

        # Quota.  min is 1 once a project is opened; max = groups * per-group.
        num_groups    = int(row[COL_GROUPS]) if pd.notna(row[COL_GROUPS]) else 1
        max_per_group = parse_max_per_group(row[COL_MAXPG]) if pd.notna(row[COL_MAXPG]) else 1
        quota         = (1, num_groups * max_per_group)

        supervisor = str(row[COL_SUPER]).strip() if pd.notna(row[COL_SUPER]) else ""

        projects[proj_id] = {
            "name":          title,
            "supervisor":    supervisor,
            "tracks":        tracks,
            "num_groups":    num_groups,
            "max_per_group": max_per_group,
            "quota":         quota,
        }
        title_to_id[_norm_title(title)] = proj_id

    return projects, title_to_id


# ── Load students ─────────────────────────────────────────────────────────────
def load_students(path: Path, title_to_id: dict) -> tuple[list, list]:
    df = pd.read_excel(path)

    # Locate the partner column by prefix (its full header is very long).
    partner_col = next((c for c in df.columns if c.startswith(PARTNER_PREFIX)), df.columns[-1])

    students, unmatched = [], []
    for i, row in df.iterrows():
        name  = str(row["User full name"]).strip()
        email = str(row["Email address"]).strip() if pd.notna(row["Email address"]) else ""

        choices = []
        for col in ("Choice 1", "Choice 2", "Choice 3"):
            raw = row.get(col)
            if pd.isna(raw):
                continue
            title = parse_choice_title(str(raw))
            if title is None:
                unmatched.append((name, col, str(raw)))
                continue
            pid = title_to_id.get(_norm_title(title))
            if pid is None:
                unmatched.append((name, col, title))
            elif pid not in choices:           # dedupe (one student repeated a choice)
                choices.append(pid)

        raw_partner = row.get(partner_col)
        partner_raw = None if pd.isna(raw_partner) else str(raw_partner).strip()

        students.append({
            "id":          i + 1,
            "name":        name,
            "email":       email,
            "choices":     choices,
            "ranks":       {pid: r + 1 for r, pid in enumerate(choices)},
            "partner_raw": partner_raw,
        })

    return students, unmatched


# ── Partner groups (connected components of the mutual-mention graph) ──────────
def build_partner_groups(students: list) -> tuple[list, list]:
    adjacency = defaultdict(set)
    warnings  = []

    for s in students:
        for part in parse_partner_raw(s["partner_raw"]):
            partner = find_student_by_name(part, students)
            if partner is None:
                warnings.append(f"No match for partner '{part}' (listed by {s['name']})")
            elif partner["id"] != s["id"]:
                adjacency[s["id"]].add(partner["id"])
                adjacency[partner["id"]].add(s["id"])

    visited, groups = set(), []
    for sid in adjacency:
        if sid in visited:
            continue
        comp, queue = [], [sid]
        while queue:
            cur = queue.pop()
            if cur in visited:
                continue
            visited.add(cur)
            comp.append(cur)
            queue.extend(adjacency[cur] - visited)
        groups.append(sorted(comp))

    return groups, warnings


# ── Demand statistics ─────────────────────────────────────────────────────────
def compute_stats(projects: dict, students: list) -> dict:
    demand = defaultdict(int)
    for s in students:
        for pid in s["choices"]:
            demand[pid] += 1

    oversubscribed, zero_demand = [], []
    for pid, proj in projects.items():
        _, max_q = proj["quota"]
        d = demand.get(pid, 0)
        if d == 0:
            zero_demand.append(pid)
        elif d > max_q:
            oversubscribed.append((pid, d, max_q))

    return {"demand": dict(demand),
            "oversubscribed": oversubscribed,
            "zero_demand": zero_demand}


# ── Summary printer ───────────────────────────────────────────────────────────
def print_summary(projects, students, groups, warnings, stats, unmatched) -> None:
    id_to_name = {s["id"]: s["name"] for s in students}
    SEP = "─" * 64
    print(SEP)
    print("  SPRING 2026 — DATA LOAD SUMMARY")
    print(SEP)
    print(f"  Projects loaded : {len(projects)}")
    print(f"  Students loaded : {len(students)}")
    in_groups = sum(len(g) for g in groups)
    print(f"  Partner groups  : {len(groups)}  ({in_groups} students)")
    print()

    all_tracks = sorted({t for p in projects.values() for t in p["tracks"]})
    print(f"  Tracks ({len(all_tracks)}):")
    for t in all_tracks:
        n = sum(1 for p in projects.values() if t in p["tracks"])
        print(f"    • {t}  ({n} projects)")
    print()

    print(f"  Oversubscribed projects : {len(stats['oversubscribed'])}")
    print(f"  Zero-demand projects    : {len(stats['zero_demand'])}")
    print()

    if groups:
        print("  Partner groups:")
        for g in groups:
            print(f"    [{len(g)}]  " + ", ".join(id_to_name[s] for s in g))
        print()

    if warnings:
        print(f"  [WARN] partner match failures ({len(warnings)}):")
        for w in warnings:
            print("    •", w)
        print()

    if unmatched:
        print(f"  [WARN] unmatched choices ({len(unmatched)}):")
        for name, col, val in unmatched:
            print(f"    • {name} / {col}: {str(val)[:60]}")
        print()

    print(SEP)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print(f"Loading projects  ←  {PROJECTS_FILE.name}")
    projects, title_to_id = load_projects(PROJECTS_FILE)

    print(f"Loading students  ←  {STUDENTS_FILE.name}")
    students, unmatched = load_students(STUDENTS_FILE, title_to_id)

    groups, warnings = build_partner_groups(students)
    stats = compute_stats(projects, students)

    print_summary(projects, students, groups, warnings, stats, unmatched)

    # attach group index, drop the raw partner string
    sid_to_group = {sid: gi for gi, g in enumerate(groups) for sid in g}
    for s in students:
        s["partner_group"] = sid_to_group.get(s["id"])
        s.pop("partner_raw", None)

    with open(OUTPUT_FILE, "wb") as f:
        pickle.dump({"projects": projects,
                     "students": students,
                     "partner_groups": groups,
                     "stats": stats}, f)
    print(f"Saved  →  {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
