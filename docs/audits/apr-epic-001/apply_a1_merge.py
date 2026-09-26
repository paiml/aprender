#!/usr/bin/env python3
"""A1 apply, header-merge form. The planned bodies were written before the v1.5 link/park/collapse runs appended
## Parked / ## Collapsed rows to these epics, so a wholesale body replace would erase them (FLOW-014 RED).
New body = planned A1 header + '## Prior body' + the LIVE body. Idempotent on the A1 marker. Reads every edit back."""
import json, os, subprocess, sys, time
D = os.path.dirname(os.path.abspath(__file__)); R = "paiml/aprender"; MARK = "<!-- APR-EPIC-001 A1 -->"
PRIOR = "<!-- APR-EPIC-001 prior body"
if "--go" not in sys.argv: sys.exit("refusing: pass --go")
def view(n): return json.loads(subprocess.run(["gh","issue","view",str(n),"-R",R,"--json","title,body,labels"],capture_output=True,text=True,check=True).stdout)
out = []
for p in json.load(open(f"{D}/plan-A1.json"))["plan"]:
    n = p["issue"]; cur = view(n); live = cur["body"] or ""
    header = open(f"{D}/{p['body_file']}").read().split(PRIOR)[0].rstrip()
    if "\n## Parked" in live: header = header.replace("\n## Parked\n", "\n")  # live body already carries its own
    body = live if MARK in live else f"{header}\n\n{PRIOR}, kept verbatim so existing links and history survive -->\n## Prior body\n\n{live}"
    labs = {l["name"] for l in cur["labels"]}
    if cur["title"] == p["title_after"] and MARK in live and "epic" in labs:
        out.append({"issue": n, "result": "skip-already-applied"}); continue
    cmd = ["gh","issue","edit",str(n),"-R",R,"--title",p["title_after"],"--body-file","-"] + ([] if "epic" in labs else ["--add-label","epic"])
    rc = subprocess.run(cmd, input=body, capture_output=True, text=True).returncode; time.sleep(3.1)
    a = view(n)
    ok = rc == 0 and a["title"] == p["title_after"] and a["body"].strip() == body.strip() and live.strip() in a["body"] and "epic" in {l["name"] for l in a["labels"]}
    out.append({"issue": n, "result": "applied" if ok else "readback-mismatch", "rc": rc})
    print(n, out[-1]["result"], flush=True)
json.dump(out, open(f"{D}/applied-A1.json","w"), indent=1)
bad = [o for o in out if o["result"] == "readback-mismatch"]; print(f"DONE applied={sum(o['result']=='applied' for o in out)} skipped={sum(o['result'].startswith('skip') for o in out)} bad={len(bad)}"); sys.exit(1 if bad else 0)
