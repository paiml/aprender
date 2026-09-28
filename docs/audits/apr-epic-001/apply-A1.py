#!/usr/bin/env python3
"""Apply plan-A1.json. Idempotent (skips issues already at the planned title+body), throttled
to 1 write / 2 s (≤30/min), and reads every edit back. Refuses to run without --go."""
import json, subprocess, sys, time, hashlib
if "--go" not in sys.argv: sys.exit("dry: pass --go after the cop's OK")
P = json.load(open("plan-A1.json")); R = []
def view(n):
    return json.loads(subprocess.check_output(["gh","issue","view",str(n),"-R","paiml/aprender","--json","title,body,labels"]))
for p in P["plan"]:
    body = open(p["body_file"]).read(); cur = view(p["issue"])
    need_lab = [l for l in p["add_labels"] if l not in {x["name"] for x in cur["labels"]}]
    if cur["title"] == p["title_after"] and cur["body"].strip() == body.strip() and not need_lab:
        R.append({"issue": p["issue"], "action": "skip-already-applied"}); continue
    cmd = ["gh","issue","edit",str(p["issue"]),"-R","paiml/aprender","--title",p["title_after"],"--body-file",p["body_file"]]
    for l in need_lab: cmd += ["--add-label", l]
    rc = subprocess.run(cmd, capture_output=True).returncode; time.sleep(2)
    after = view(p["issue"])
    ok = rc == 0 and after["title"] == p["title_after"] and hashlib.sha256(after["body"].encode()).hexdigest() in (p["body_sha256"], hashlib.sha256(after["body"].strip().encode()).hexdigest()) and "epic" in {x["name"] for x in after["labels"]}
    ok = ok and after["body"].strip() == body.strip()
    R.append({"issue": p["issue"], "action": "edited", "rc": rc, "read_back_ok": ok})
json.dump(R, open("applied-A1.json","w"), indent=1)
bad = [r for r in R if r.get("read_back_ok") is False]
print(f"applied={sum(r['action']=='edited' for r in R)} skipped={sum(r['action']!='edited' for r in R)} bad={len(bad)}")
sys.exit(1 if bad else 0)
