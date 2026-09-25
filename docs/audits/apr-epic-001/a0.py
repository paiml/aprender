#!/usr/bin/env python3
"""APR-EPIC-001 A0 (#4432): re-derive live issue state, read-only. Writes a0-state.json beside this file:
the spec §1 facts measured now, and a diff against the 2026-09-25T14:16Z snapshot values the spec records.
No GitHub writes: only `gh api graphql` queries."""
import json, os, subprocess, time

R = ("paiml", "aprender")
SNAPSHOT = {"at": "2026-09-25T14:16Z", "open_issues": 824, "open_epics": 41, "no_epic_link": 509, "no_milestone": 4,
            "milestone_open": {"0.70.0": 81}}
Q = """query($o:String!,$n:String!,$c:String){repository(owner:$o,name:$n){issues(states:OPEN,first:100,after:$c){
pageInfo{hasNextPage endCursor} nodes{number title milestone{title} labels(first:30){nodes{name}}
parent{number} subIssuesSummary{total}}}}}"""


def gql(q, **v):
    args = ["gh", "api", "graphql", "-f", f"query={q}"] + [a for k, x in v.items() if x is not None for a in ("-F", f"{k}={x}")]
    return json.loads(subprocess.run(args, capture_output=True, text=True, check=True).stdout)["data"]


issues, cur = [], None
while True:
    page = gql(Q, o=R[0], n=R[1], c=cur)["repository"]["issues"]
    issues += page["nodes"]
    if not page["pageInfo"]["hasNextPage"]:
        break
    cur = page["pageInfo"]["endCursor"]
if not issues:
    raise SystemExit("VACUOUS: 0 open issues read")  # an empty read is a failed read, not a clean repo


def is_epic(i):
    return any(l["name"] == "epic" for l in i["labels"]["nodes"]) or i["title"].startswith("EPIC ")


epics = [i for i in issues if is_epic(i)]
unlinked = [i["number"] for i in issues if not is_epic(i) and i["parent"] is None]
nomile = sorted(i["number"] for i in issues if i["milestone"] is None)
by_ms = {}
for i in issues:
    k = (i["milestone"] or {}).get("title", "(none)")
    by_ms[k] = by_ms.get(k, 0) + 1
live = {"open_issues": len(issues), "open_epics": len(epics), "no_epic_link": len(unlinked),
        "no_milestone": len(nomile), "milestone_open": dict(sorted(by_ms.items()))}
diff = {k: {"snapshot": SNAPSHOT[k], "live": live[k], "delta": live[k] - SNAPSHOT[k]}
        for k in ("open_issues", "open_epics", "no_epic_link", "no_milestone")}
diff["milestone_open.0.70.0"] = {"snapshot": 81, "live": by_ms.get("0.70.0", 0), "delta": by_ms.get("0.70.0", 0) - 81}
out = {"row": "A0", "issue": 4432, "measured_at": time.strftime("%Y-%m-%dT%H:%MZ", time.gmtime()),
       "source": "gh api graphql repository.issues(states:OPEN), paginated", "epic_rule": "label `epic` or title `EPIC `",
       "live": live, "snapshot": SNAPSHOT, "diff": diff,
       "epics": sorted(({"number": e["number"], "title": e["title"], "children": e["subIssuesSummary"]["total"]}
                        for e in epics), key=lambda e: e["number"]),
       "no_milestone_numbers": nomile, "no_epic_link_numbers": sorted(unlinked),
       "not_rederived": {"status (in-PR/branch-only/claimed/unclaimed/stale)": "the snapshot generator is not in the tree",
                         "done-on-main candidates": "needs a git log scan per issue; out of scope for a read-only receipt"}}
p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "a0-state.json")
json.dump(out, open(p, "w"), indent=1)
print(json.dumps(diff))
