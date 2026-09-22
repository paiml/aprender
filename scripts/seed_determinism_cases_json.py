"""Build evidence/response-contract/cases.json from det.sh output directories."""
import json, sys, os, glob
out = {"measured_by": "aprender-fd", "contract": "apr-response-contract-v1", "cases": []}
for host, backend, d in [tuple(a.split("=", 2)) for a in sys.argv[1:-1]]:
    ver = open(os.path.join(d, "version.txt")).read().strip()
    for sha in sorted(glob.glob(os.path.join(d, "*.sha256sum"))):
        model = os.path.basename(sha)[: -len(".sha256sum")]
        want = open(sha).read().strip()
        for verb in ("run", "serve"):
            for mode in ("greedy", "sampled"):
                def load(i):
                    try: return json.load(open(f"{d}/{model}.{verb}.{mode}.{i}.json"))
                    except Exception: return None
                a, b = load(1), load(2)
                if a is None or b is None: continue
                text = (lambda x: x.get("text") if verb == "run" else ((x.get("choices") or [{}])[0].get("message") or {}).get("content"))
                greedy = (lambda: load(1) if mode == "greedy" else json.load(open(f"{d}/{model}.{verb}.greedy.1.json")))()
                out["cases"].append({
                    "host": host, "backend": backend, "apr": ver, "model": model, "verb": verb,
                    "mode": mode,
                    "model_digest": a.get("model_digest") or "",
                    "sha256sum": want,
                    "digest_matches": (a.get("model_digest") or "") == want,
                    "identical": text(a) is not None and text(a) == text(b),
                    "sampling_engaged": mode == "greedy" or text(a) != text(greedy),
                })
json.dump(out, open(sys.argv[-1], "w"), indent=1)
json.dump(out, sys.stdout, indent=1)
