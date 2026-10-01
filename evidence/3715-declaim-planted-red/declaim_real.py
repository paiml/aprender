import sys, copy, yaml
sys.path.insert(0, "scripts/lib"); import model_ladder_cells as J
L = yaml.safe_load(open("contracts/model-capability-ladder-v1.yaml"))["ladder"]
def run(L, tag):
    lines = []
    dec, drc = J.declaimed(L.get("cells") or {}, {h["id"] for h in L["hosts"]}, lines.append)
    rc = drc or J.declaim_still_claimed(L, dec, lines.append)
    print(f"{tag:34} rc={rc} declaimed={sorted(dec)}"); [print("   ", l) for l in lines]
    return rc
a = run(L, "real contract")
P = copy.deepcopy(L)   # planted: a qwen3moe rung with no hosts: key claims it on every host, gx10 included
P["rungs"].append({"id": "qwen3moe-30b-a3b-q4km", "family": "qwen3moe", "arch": "qwen3moe", "required": True,
                   "gguf": "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf", "backends": ["cpu", "cuda"]})
b = run(P, "PLANTED qwen3moe rung (all hosts)")
Q = copy.deepcopy(P); Q["rungs"][-1]["hosts"] = ["lambda"]
c = run(Q, "planted rung narrowed to lambda")
R = copy.deepcopy(L); R["cells"]["long_rungs_for"]["representatives"]["qwen3moe"] = "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"
R["cells"]["declaimed"].append(dict(R["cells"]["declaimed"][-1], host="lambda"))
d = run(R, "PLANTED rep, de-claimed on all hosts")
sys.exit(0 if (a, b, c, d) == (0, 1, 0, 1) else 1)
