import sys, os, subprocess, collections, json, hashlib
import numpy as np, importlib.metadata as md
from gguf import GGUFReader
from gguf.quants import dequantize
BIN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "iqdec.bin")
TYPES = {"IQ4_NL", "IQ4_XS", "IQ3_S"}
QS_BYTE = {"IQ4_NL": 2, "IQ4_XS": 8, "IQ3_S": 2}   # first quant byte of block 0 (after d / scales)
CENSUS = {"Qwen2.5-0.5B-Instruct-IQ3_M.gguf": {"IQ4_NL": 96, "IQ3_S": 21},
          "Qwen2.5-0.5B-Instruct-IQ4_XS.gguf": {"IQ4_NL": 120, "IQ4_XS": 24},
          "Qwen3.5-4B-UD-Q4_K_XL.gguf": {"IQ4_XS": 10}}
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dec.f32")

def ours(path, t, flip=None):
    cmd = [BIN, path, str(t.data_offset), str(t.n_bytes), t.tensor_type.name, out]
    if flip is not None: cmd.append(str(flip))
    subprocess.run(cmd, check=True)
    return np.fromfile(out, dtype="<f4")

def cmp(a, ref):
    if a.size != ref.size: return dict(size_mismatch=[int(a.size), int(ref.size)])
    d = np.abs(a - ref)
    rel = d / np.maximum(np.abs(ref), 1e-30)
    return dict(n=int(ref.size), bit_exact=int((a.view("<u4") == ref.view("<u4")).sum()),
                max_abs=float(d.max()), max_rel=float(rel[ref != 0].max() if (ref != 0).any() else 0.0),
                mismatch_gt_1e6rel=int((d > 1e-6 * np.maximum(np.abs(ref), 1.0)).sum()),
                nonfinite_ours=int((~np.isfinite(a)).sum()))

print("gguf-py", md.version("gguf"), "numpy", np.__version__)
report = {}
for fname, want in CENSUS.items():
    path = os.path.expanduser("~/models/" + fname)
    r = GGUFReader(path)
    per = collections.defaultdict(lambda: dict(tensors=0, elems=0, bit_exact=0, mismatch=0, max_abs=0.0, max_rel=0.0, nonfinite=0, size_mismatch=0))
    control_done = set()
    for t in r.tensors:
        ty = t.tensor_type.name
        if ty not in TYPES: continue
        ref = dequantize(t.data, t.tensor_type).astype("<f4").ravel()
        # positive control FIRST, on the first tensor of each type: a one-bit flip must turn it RED
        if ty not in control_done:
            c = cmp(ours(path, t, flip=QS_BYTE[ty]), ref)
            red = c.get("mismatch_gt_1e6rel", 1) > 0
            print(f"  CONTROL {fname} {ty} {t.name}: flip bit0 of byte {QS_BYTE[ty]} -> mismatches={c.get('mismatch_gt_1e6rel')} max_abs={c.get('max_abs'):.6g} => {'RED (control works)' if red else 'GREEN — CONTROL FAILED, comparison is vacuous'}")
            if not red: sys.exit(2)
            control_done.add(ty)
        c = cmp(ours(path, t), ref)
        p = per[ty]; p["tensors"] += 1
        if "size_mismatch" in c: p["size_mismatch"] += 1; print("  SIZE MISMATCH", t.name, c); continue
        p["elems"] += c["n"]; p["bit_exact"] += c["bit_exact"]; p["mismatch"] += c["mismatch_gt_1e6rel"]
        p["max_abs"] = max(p["max_abs"], c["max_abs"]); p["max_rel"] = max(p["max_rel"], c["max_rel"]); p["nonfinite"] += c["nonfinite_ours"]
        if c["mismatch_gt_1e6rel"]: print("  MISMATCH", fname, ty, t.name, c)
    for ty, p in per.items():
        ok_count = p["tensors"] == want[ty]
        verdict = "GREEN" if ok_count and p["mismatch"] == 0 and p["size_mismatch"] == 0 and p["nonfinite"] == 0 else "RED"
        print(f"{verdict} ({fname}, {ty}) tensors={p['tensors']}/{want[ty]} elems={p['elems']} bit_exact={p['bit_exact']} mismatches={p['mismatch']} max_abs={p['max_abs']:.3g} max_rel={p['max_rel']:.3g} nonfinite={p['nonfinite']}")
        report[f"{fname}|{ty}"] = dict(p, census=want[ty], verdict=verdict)
    missing = set(want) - set(per)
    if missing: print("RED — types in census never compared:", missing)
json.dump(report, open("compare_report.json", "w"), indent=1)
