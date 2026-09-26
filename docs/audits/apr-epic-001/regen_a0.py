#!/usr/bin/env python3
"""APR-EPIC-001 row A0 (#4432): re-derive the spec §1 ground truth at HEAD and
diff it against the 2026-09-25 triage snapshot. Read-only: every GitHub call
is a GET or a GraphQL query; nothing is written to GitHub.

Usage: regen_a0.py SNAPSHOT_JSON INBOX_MD OUT_JSON
Status rules (first match wins), from the snapshot's own method section:
  in-PR       an open PR's title/body/branch names #N
  branch-only a remote branch names N, and no open PR does
  claimed     an inbox CLAIM line or an epic owner names #N
  stale       no activity > 7 d (milestone-move bookkeeping comments ignored)
  unclaimed   none of the above
Rule 16: epics are excluded from the unclaimed and no-epic-link sets.
"""
import json, re, subprocess, sys, datetime as dt, collections

REPO = "paiml/aprender"
OWNER, NAME = REPO.split("/")
SNAP, INBOX, OUT = sys.argv[1:4]
NOW = dt.datetime.now(dt.timezone.utc)
BOOKKEEPING = re.compile(r"milestone|moved (to|from) 0\.\d+|triage", re.I)
CLOSING = re.compile(r"\b(close[sd]?|fix(e[sd])?|resolve[sd]?)\s+#(\d+)", re.I)


def sh(*a):
    return subprocess.run(a, check=True, capture_output=True, text=True).stdout


def gql_issues():
    q = """query($o:String!,$n:String!,$c:String){repository(owner:$o,name:$n){
      issues(states:OPEN,first:100,after:$c){pageInfo{hasNextPage endCursor}
        nodes{number title body createdAt updatedAt lastEditedAt
          milestone{title dueOn} labels(first:30){nodes{name}}
          parent{number} subIssuesSummary{total completed}
          comments(last:10){nodes{createdAt body}}}}}}"""
    out, cur = [], None
    while True:
        args = ["gh", "api", "graphql", "-f", f"query={q}", "-f", f"o={OWNER}", "-f", f"n={NAME}"]
        if cur:
            args += ["-f", f"c={cur}"]
        page = json.loads(sh(*args))["data"]["repository"]["issues"]
        out += page["nodes"]
        if not page["pageInfo"]["hasNextPage"]:
            return out
        cur = page["pageInfo"]["endCursor"]


def ts(s):
    return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))


def refs(text, universe):
    return {int(m) for m in re.findall(r"#(\d{3,5})\b", text or "") if int(m) in universe}


issues = gql_issues()
nums = {i["number"] for i in issues}
labels = {i["number"]: [l["name"] for l in i["labels"]["nodes"]] for i in issues}
is_epic = {i["number"]: ("epic" in labels[i["number"]]
                         or i["subIssuesSummary"]["total"] > 0
                         or re.search(r"\bepic\b", i["title"], re.I) is not None)
           for i in issues}
epics = {n for n, e in is_epic.items() if e}

prs = json.loads(sh("gh", "pr", "list", "-R", REPO, "--state", "open", "--limit", "500",
                    "--json", "number,title,body,headRefName"))
in_pr = collections.defaultdict(list)
for p in prs:
    hit = refs(p["title"] + "\n" + (p["body"] or ""), nums)
    hit |= {int(m) for m in re.findall(r"(?<!\d)(\d{4})(?!\d)", p["headRefName"]) if int(m) in nums}
    for n in hit:
        in_pr[n].append(p["number"])

branches = [l.split("refs/heads/", 1)[1] for l in sh("git", "ls-remote", "--heads", "origin").splitlines()]
on_branch = collections.defaultdict(list)
for b in branches:
    for m in re.findall(r"(?<!\d)(\d{4})(?!\d)", b):
        if int(m) in nums:
            on_branch[int(m)].append(b)

claimed = collections.defaultdict(list)
for line in open(INBOX, encoding="utf-8", errors="replace"):
    if "CLAIM" in line:
        for n in refs(line, nums):
            claimed[n].append(line.split("|")[1].strip() if "|" in line else "inbox")
for i in issues:
    if i["number"] in epics:
        m = re.search(r"owner:\s*([\w-]+)", i["body"] or "", re.I)
        if m:
            claimed[i["number"]].append("owner:" + m.group(1))

mainlog = sh("git", "log", "origin/main", "--since=2026-06-01", "--format=%h%x09%s%n%b%x1e")
strong, medium = collections.defaultdict(set), collections.defaultdict(set)
for rec in mainlog.split("\x1e"):
    if not rec.strip():
        continue
    sha = rec.strip().split("\t", 1)[0]
    for m in CLOSING.finditer(rec):
        if int(m.group(3)) in nums:
            strong[int(m.group(3))].add(sha)
    subj = rec.strip().split("\n", 1)[0]
    if re.search(r"\b(fix|feat)\b", subj, re.I):
        for n in refs(subj, nums):
            medium[n].add(sha)

rows = {}
for i in issues:
    n = i["number"]
    acts = [ts(i["createdAt"])] + ([ts(i["lastEditedAt"])] if i["lastEditedAt"] else [])
    acts += [ts(c["createdAt"]) for c in i["comments"]["nodes"] if not BOOKKEEPING.search(c["body"] or "")]
    idle = (NOW - max(acts)).days
    if in_pr[n]:
        st = "in-PR"
    elif on_branch[n]:
        st = "branch-only"
    elif claimed[n]:
        st = "claimed"
    elif idle > 7:
        st = "stale"
    else:
        st = "unclaimed"
    linked = i["parent"] is not None or bool(refs(i["body"], epics - {n}))
    rows[n] = {"number": n, "title": i["title"], "milestone": (i["milestone"] or {}).get("title"),
               "is_epic": n in epics, "epic_linked": linked, "status": st, "idle_days": idle,
               "prs": sorted(in_pr[n]), "branches": sorted(on_branch[n])[:5], "claims": claimed[n][:3],
               "main_strong": sorted(strong[n]), "main_medium": sorted(medium[n] - strong[n])}

work = [r for r in rows.values() if not r["is_epic"]]            # rule 16
ms = json.loads(sh("gh", "api", f"repos/{REPO}/milestones?state=open&per_page=100"))
ms_rows = sorted(({"title": m["title"], "due": m["due_on"], "open": m["open_issues"],
                   "closed": m["closed_issues"]} for m in ms), key=lambda m: m["title"])

state = {
    "row": "APR-EPIC-001 A0", "ticket": "GH-4432",
    "derived_at_utc": NOW.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "head": sh("git", "rev-parse", "origin/main").strip(),
    "generator": "docs/audits/apr-epic-001/regen_a0.py",
    "rules": {"status_first_match": ["in-PR", "branch-only", "claimed", "stale", "unclaimed"],
              "rule16": "epics excluded from unclaimed and no-epic-link sets",
              "is_epic": "label epic OR has GitHub sub-issues OR title matches \\bepic\\b"},
    "caveats": [
        "claims read only from the inbox files present at derive time (rotated lines are gone); roadmap rows not read",
        "idle ignores comments matching milestone/triage bookkeeping; only the last 10 comments are read",
        "done_on_main scans origin/main since 2026-06-01; every candidate must be verified before any close",
    ],
    "counts": {
        "open_issues": len(rows),
        "open_epics": len(epics),
        "open_epics_label_only": sum("epic" in labels[n] for n in nums),
        "no_epic_link_non_epic": sum(not r["epic_linked"] for r in work),
        "no_milestone": sorted(r["number"] for r in rows.values() if not r["milestone"]),
        "status_non_epic": dict(collections.Counter(r["status"] for r in work)),
        "done_on_main_strong": sorted(r["number"] for r in work if r["main_strong"]),
        "done_on_main_medium": sorted(r["number"] for r in work if r["main_medium"]),
        "close_candidates_idle_gt7_no_claim": sorted(r["number"] for r in work if r["status"] == "stale"),
    },
    "milestones": ms_rows,
    "issues": [rows[n] for n in sorted(rows)],
}

snap = json.load(open(SNAP))
s_issues = {i["number"]: i for i in snap["issues"]}
now_set, snap_set = set(rows), set(s_issues)
status_moves = collections.Counter(
    f'{s_issues[n]["status"]}->{rows[n]["status"]}' for n in now_set & snap_set
    if s_issues[n]["status"] != rows[n]["status"])
state["diff_vs_snapshot"] = {
    "snapshot_file": "~/Desktop/aprender-triage-report.json (aprender-d1)",
    "snapshot_fetched_at_utc": snap["fetched_at_utc"],
    "spec_cites": "2026-09-25T14:16Z — the file says fetched_at_utc above; spec timestamp differs",
    "open_issues": {"snapshot": snap["open_count"], "now": len(rows)},
    "open_epics": {"snapshot": sum(i["is_epic"] for i in snap["issues"]), "now": len(epics)},
    "closed_since": sorted(snap_set - now_set),
    "opened_since": sorted(now_set - snap_set),
    "milestone_changed": sorted(n for n in now_set & snap_set
                                if (s_issues[n]["milestone"] or None) != rows[n]["milestone"]),
    "epic_flag_changed": sorted(n for n in now_set & snap_set if s_issues[n]["is_epic"] != rows[n]["is_epic"]),
    "status_snapshot": dict(collections.Counter(i["status"] for i in snap["issues"] if not i["is_epic"])),
    "status_now": state["counts"]["status_non_epic"],
    "status_moves": dict(status_moves.most_common()),
}
json.dump(state, open(OUT, "w"), indent=1, sort_keys=False)
print(json.dumps({k: v for k, v in state["diff_vs_snapshot"].items()
                  if k not in ("closed_since", "opened_since", "milestone_changed")}, indent=1))
print("counts", json.dumps({k: (v if not isinstance(v, list) or len(v) < 8 else len(v))
                            for k, v in state["counts"].items()}))
