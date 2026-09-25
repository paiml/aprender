#!/usr/bin/env python3
"""APR-EPIC-001 row A0: re-derive the spec's §1 ground truth from fresh reads, diff it
against the [S] snapshot (triage report 2026-09-25T14:16Z), and pre-compute what later
rows need to know before any write: §2 reuse/absorb liveness and the rule-6 never-park
floor (S-2). Read-only: consumes the fetches in this dir, writes a0-state.json."""
import collections
import datetime as dt
import json
import os
import re
import statistics
import subprocess
import sys

W = os.path.dirname(os.path.abspath(__file__))
J = lambda f: json.load(open(os.path.join(W, f)))
OUT = sys.argv[1]
SNAP = sys.argv[2]  # the [S] snapshot's triage.json
REPO = "/home/noah/src/aprender"

fetched = open(os.path.join(W, "fetched_at")).read().strip()
NOW = dt.datetime.fromisoformat(fetched.replace("Z", "+00:00"))
issues = J("issues.json")
rel = {int(k): v for k, v in J("rel.json").items()}
prs = J("prs.json")
ms = {m["title"]: m for m in J("milestones.json")}
closed = J("closed.json")
merged7d = J("merged7d.json")
tri = {r["number"]: r for r in J("triage.json")["issues"]}
snap = json.load(open(SNAP))
S = {r["number"]: r for r in snap["issues"]}
OPEN = {i["number"]: i for i in issues}
LIVE = set(open(os.path.join(W, "live.txt")).read().split())


def git(*a):
    return subprocess.run(["git", "-C", REPO, *a], capture_output=True, text=True).stdout.strip()


def age_h(ts):
    return (NOW - dt.datetime.fromisoformat(ts.replace("Z", "+00:00"))).total_seconds() / 3600


labels = lambda i: {l["name"] for l in i["labels"]}
EPICS = sorted(n for n, r in tri.items() if r["is_epic"])  # same predicate as [S]: label | sub-issues | EPIC title
BODY_REF = re.compile(r"#(\d{3,5})\b")


def epic_body_ref(i):
    return [int(x) for x in BODY_REF.findall(i.get("body") or "") if int(x) in EPICS and int(x) != i["number"]]


non_epic = [n for n in OPEN if n not in EPICS]
parent_null = [n for n in non_epic if not (rel.get(n) or {}).get("parent")]
orphan_strict = [n for n in parent_null if not epic_body_ref(OPEN[n])]
no_ms = sorted(n for n in OPEN if not OPEN[n]["milestone"])
PRIO = re.compile(r"^P[0-3]$")
p_unknown = sorted(n for n in non_epic if not any(PRIO.match(l) for l in labels(OPEN[n])))
p_unknown_24h = [n for n in p_unknown if age_h(OPEN[n]["createdAt"]) > 24]

status = collections.Counter(r["status"] for r in tri.values())
verdict = collections.Counter(r["verdict"] for r in tri.values())
strength = collections.Counter(
    ("strong" if "strong" in (r.get("reason") or "") else "medium")
    for r in tri.values() if r["verdict"] == "SUPERSEDED/DONE-ON-MAIN")


def train(t):
    m = ms.get(t) or {}
    closes_7d = sum(1 for c in closed if (c.get("milestone") or {}).get("title") == t
                    and age_h(c["closedAt"]) <= 168)
    return {"due_on": (m.get("due_on") or "")[:10] or None, "open": m.get("open_issues"),
            "closed": m.get("closed_issues"), "closes_7d": closes_7d,
            "closes_per_day_7d": round(closes_7d / 7, 1)}


shipped = [t for t, m in ms.items() if m["state"] == "closed" and re.fullmatch(r"0\.(6[7-9])\.[01]", t)]
per_train = {t: ms[t]["closed_issues"] for t in sorted(shipped)}
by_day = collections.Counter(p["mergedAt"][:10] for p in merged7d)

# ── rule 6: the never-park floor ────────────────────────────────────────────
linked = collections.defaultdict(set)   # GitHub-linked (closingIssuesReferences) or named in title/head
mention = collections.defaultdict(set)  # named only in a PR body
for p in prs:
    for c in p.get("closingIssuesReferences") or []:
        linked[c["number"]].add(p["number"])
    for x in re.findall(r"(?<!\d)(\d{3,5})(?!\d)", p["headRefName"]) + re.findall(r"#(\d{3,5})\b", p["title"] or ""):
        linked[int(x)].add(p["number"])
    for x in re.findall(r"#(\d{3,5})\b", p["body"] or ""):
        mention[int(x)].add(p["number"])
TRAIN_ORDER = ["0.70.0", "0.70.1", "0.71.0", "0.72.0"]  # current = 0.70; rule 6 protects P0 in trains <= current+2
live_claim = {}
for n, r in tri.items():
    who = set(re.findall(r"\b((?:aprender|infra|apex|rmedia|ruchy|xpile|manzana)-[0-9a-f]{2})\b", " ".join(r.get("claim") or [])))
    if who & LIVE:
        live_claim[n] = sorted(who & LIVE)
floor = {}
for n in non_epic:
    i = OPEN[n]
    why = []
    if linked.get(n):
        why.append(f"open PR linked {sorted(linked[n])}")
    if n in live_claim:
        why.append(f"live claim {live_claim[n]}")
    ml = (i["milestone"] or {}).get("title")
    if "P0" in labels(i) and ml in TRAIN_ORDER:
        why.append(f"P0 on {ml}")
    if any(a["login"] == "noahgift" for a in i["assignees"]) and age_h(tri[n]["last_activity_utc"]) <= 24:
        why.append("assignee noahgift active <24h")
    if why:
        floor[n] = why
floor_broad = set(floor) | {n for n in non_epic if mention.get(n)}

# ── §2 epic set: reuse issues and absorbed epics, live ──────────────────────
E2 = {"E0": (3269, [4071, 4072, 4073, 3847, 4069, 3972, 4047, 4070, 4075, 4330, 4074, 4077, 4078, 4079, 3715, 3856, 3559, 3560, 4319, 4379, 4420, 3610, 3611, 3624, 4100]),
      "E1": (3998, [4033, 4288, 4232, 3081, 3058]), "E2": (3598, [4249]), "E3": (3994, [3062, 3597]),
      "E4": (4381, [4354, 4252]), "E5": (4000, []), "E6": (3999, [2728, 2880, 3144]), "E7": (4001, [3421, 3423, 3428]),
      "E8": (4002, []), "E9": (4164, [3146, 3370]), "S1": (3863, [2876, 2877, 2878, 2879, 2503]),
      "S2": (2870, [2556, 4080, 4081, 4082, 4083, 4084, 4122, 4139, 4166, 4168, 4197, 4198, 4199, 4200, 4201, 4202, 4238, 4239, 4240, 4241, 4244, 4347, 4351]),
      "S3": (3997, [4057, 2814, 2582, 2373])}
epic_set = {}
for eid, (reuse, absorbs) in E2.items():
    def st(n):
        if n in OPEN:
            r = rel.get(n) or {}
            return {"n": n, "state": "open", "is_epic": n in EPICS, "milestone": (OPEN[n]["milestone"] or {}).get("title"),
                    "children_open": (r.get("subIssuesSummary") or {}).get("total", 0) - (r.get("subIssuesSummary") or {}).get("completed", 0)}
        return {"n": n, "state": "not-open"}
    epic_set[eid] = {"reuse": st(reuse), "absorbs": [st(a) for a in absorbs]}
not_open_refs = sorted({x["n"] for e in epic_set.values() for x in [e["reuse"], *e["absorbs"]] if x["state"] != "open"})
# every open epic the §2 set does not name (would be orphaned by A4)
named = {x["n"] for e in epic_set.values() for x in [e["reuse"], *e["absorbs"]]}
unnamed_epics = [{"n": n, "title": OPEN[n]["title"][:90], "milestone": (OPEN[n]["milestone"] or {}).get("title")}
                 for n in EPICS if n not in named]


def fact(name, cmd, value, s_value, mark="M"):
    d = None
    if isinstance(value, (int, float)) and isinstance(s_value, (int, float)):
        d = value - s_value
    return {"fact": name, "cmd": cmd, "value": value, "snapshot": s_value, "delta": d, "mark": mark}


s_status = collections.Counter(r["status"] for r in S.values())
s_orphan = sum(1 for r in S.values() if not r.get("is_epic") and not r.get("epic_parent") and not r.get("epic_link"))
facts = [
    fact("open_issues", "gh issue list -R paiml/aprender --state open --limit 2000 --json number | jq length", len(OPEN), 824),
    fact("open_epics", "label epic | subIssuesSummary.total>0 | title ^EPIC (analyze.py, same as [S])", len(EPICS), 41),
    fact("no_epic_link_strict", "GraphQL parent==null and no #<open epic> body ref, epics excluded", len(orphan_strict), 509),
    fact("parent_null", "GraphQL parent==null, epics excluded (rule 1: a body ref does not count)", len(parent_null), None),
    fact("no_milestone", "gh issue list --search no:milestone", no_ms, [4337, 4350, 4377, 4422]),
    fact("status", "analyze.py (same generator as [S])", dict(status), {"in-PR": 67, "branch-only": 223, "claimed": 33, "unclaimed": 253, "stale": 248}),
    fact("done_on_main_candidates", "git log origin/main --grep '#N' (analyze.py)", dict(strength), {"strong": 6, "medium": 35}),
    fact("close_candidates", "idle>7d and no claim (analyze.py)", verdict["CLOSE-CANDIDATE"], 25),
    fact("p_unknown_over_24h", "open non-epic issues with no P0-P3 label, created >24h ago", len(p_unknown_24h), None),
    fact("train_0.70.0", "gh api repos/paiml/aprender/milestones + closed.json", train("0.70.0"), {"due_on": "2026-09-26", "open": 81, "closes_per_day_7d": 4.9}),
    fact("train_0.71.0", "milestone API", train("0.71.0"), {"due_on": "2026-09-29"}),
    fact("train_0.72.0", "milestone API", train("0.72.0"), {"due_on": "2026-10-02"}),
    fact("trains_0.73-0.76_due", "milestone API", {t: (ms.get(t) or {}).get("due_on") for t in ("0.73.0", "0.74.0", "0.75.0", "0.76.0")}, "none"),
    fact("throughput_per_train",
         "SPEC CMD IS VACUOUS: `gh pr list --state merged --search milestone:X` returns 11/0/0/1 for 0.69.0-0.69.3 because PRs carry no milestone. Substitutes: closed issues per shipped train (milestone API) and merged PRs/day (gh pr list --search merged:>=2026-09-18)",
         {"closed_issues_per_shipped_train": per_train,
          "median_closed_issues_per_train": statistics.median(per_train.values()) if per_train else None,
          "merged_prs_7d": len(merged7d), "merged_prs_per_day_7d": round(len(merged7d) / 7, 1),
          "merged_prs_by_day": dict(sorted(by_day.items()))}, 29, mark="M; [S] was [U]"),
]
state = {
    "spec": "APR-EPIC-001", "row": "A0", "ticket": "aprender#4432", "session": "aprender-d1",
    "fetched_at_utc": fetched,
    "tree": {"origin_main": git("rev-parse", "origin/main")},
    "snapshot": {"source": "~/Desktop/aprender-triage-report.json", "fetched_at_utc": snap.get("fetched_at")},
    "facts": facts,
    "churn_since_snapshot": {
        "opened": sorted(set(OPEN) - set(S)), "closed_or_gone": sorted(set(S) - set(OPEN)),
        "milestone_changed": sorted(n for n in set(OPEN) & set(S) if (OPEN[n]["milestone"] or {}).get("title") != S[n]["milestone"]),
    },
    "never_park_floor": {
        "rule": "rule 6: open PR linked (closingIssuesReferences, or #N in PR title / number in head branch) | claim by a session live in ListAgents | P0 on 0.70.0-0.72.0 | assignee noahgift active <24h",
        "count": len(floor), "count_if_body_mentions_count": len(floor_broad),
        "s2_check": {"floor_plus_13_epics": len(floor) + 13, "cap": 100, "stop": len(floor) + 13 > 100,
                     "broad_floor_plus_13": len(floor_broad) + 13, "broad_stop": len(floor_broad) + 13 > 100},
        "by_reason": dict(collections.Counter(w.split(" [")[0].split(" on ")[0] for v in floor.values() for w in v)),
        "issues": {str(n): v for n, v in sorted(floor.items())},
    },
    "epic_set_live": epic_set,
    "epic_set_refs_not_open": not_open_refs,
    "open_epics_not_named_in_spec": unnamed_epics,
    "live_sessions_listagents": sorted(LIVE),
    "open_prs": len(prs),
}
json.dump(state, open(OUT, "w"), indent=1, default=str)
print(json.dumps({k: state["never_park_floor"][k] for k in ("count", "count_if_body_mentions_count", "s2_check", "by_reason")}))
print("refs not open:", not_open_refs, "| unnamed open epics:", len(unnamed_epics))
for f in facts:
    print(f["fact"], "=", json.dumps(f["value"])[:160], "| S:", json.dumps(f["snapshot"])[:80])
