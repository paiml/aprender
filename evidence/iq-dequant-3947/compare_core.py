# Oracle for #3947's port: aprender-core GgufReader::get_tensor_f32 vs gguf-py, every IQ tensor.
import sys, os, subprocess, shutil, collections, json
import numpy as np, importlib.metadata as md
from gguf import GGUFReader
from gguf.quants import dequantize
HERE = os.path.dirname(os.path.abspath(__file__)); BIN = os.path.join(HERE, "coredec.bin")
TYPES = {"IQ4_NL", "IQ4_XS", "IQ3_S"}; QS_BYTE = {"IQ4_NL": 2, "IQ4_XS": 8, "IQ3_S": 2}
CENSUS = {"Qwen2.5-0.5B-Instruct-IQ3_M.gguf": {"IQ4_NL": 96, "IQ3_S": 21},
          "Qwen2.5-0.5B-Instruct-IQ4_XS.gguf": {"IQ4_NL": 120, "IQ4_XS": 24},
          "Qwen3.5-4B-UD-Q4_K_XL.gguf": {"IQ4_XS": 10}}
OUT = os.path.join(HERE, "coreout")

def run(path, tensors):
    shutil.rmtree(OUT, ignore_errors=True); os.makedirs(OUT)
    p = subprocess.run([BIN, path, OUT] + [t.name for t in tensors], capture_output=True, text=True, check=True)
    return [l.split("\t") for l in p.stdout.strip().split("\n")]

def compare(path, tensors, reader_tensors_ref):
    rows = run(path, tensors); res = []
    for (i, name, shape), t, ref in zip(rows, tensors, reader_tensors_ref):
        if shape.startswith("ERR"): res.append(dict(name=name, err=shape)); continue
        a = np.fromfile(f"{OUT}/{i}", dtype="<f4") if False else np.fromfile(f"{OUT}/{i}.f32", dtype="<f4")
        core_shape = json.loads(shape)
        d = np.abs(a - ref) if a.size == ref.size else None
        res.append(dict(name=name, ty=t.tensor_type.name, n=int(ref.size), size_ok=a.size == ref.size,
                        core_shape=core_shape, ggufpy_ne=[int(x) for x in t.shape],
                        shape_ok=sorted(core_shape) == sorted(int(x) for x in t.shape),
                        bit_exact=int((a.view("<u4") == ref.view("<u4")).sum()) if d is not None else 0,
                        mismatch=int((d > 1e-6 * np.maximum(np.abs(ref), 1.0)).sum()) if d is not None else -1,
                        max_abs=float(d.max()) if d is not None else float("nan")))
    return res

print("oracle gguf-py", md.version("gguf"), "numpy", np.__version__, "| subject coredec.bin (aprender-core GgufReader::get_tensor_f32)")
ok_all = True; report = {}
for fname, want in CENSUS.items():
    path = os.path.expanduser("~/models/" + fname); r = GGUFReader(path)
    iq = [t for t in r.tensors if t.tensor_type.name in TYPES]
    refs = [dequantize(t.data, t.tensor_type).astype("<f4").ravel() for t in iq]
    # POSITIVE CONTROL (0.5B files only): flip bit 0 of the first quant byte of the first tensor of
    # each type in a COPY of the file; exactly those tensors must mismatch, by exactly 1 element.
    if os.path.getsize(path) < 1 << 30:
        firsts = {}
        for t in iq: firsts.setdefault(t.tensor_type.name, t)
        mpath = os.path.join(HERE, "mutant.gguf"); shutil.copyfile(path, mpath)
        with open(mpath, "r+b") as f:
            for ty, t in firsts.items():
                off = int(t.data_offset) + QS_BYTE[ty]; f.seek(off); b = f.read(1); f.seek(off); f.write(bytes([b[0] ^ 1]))
        ctl = compare(mpath, list(firsts.values()), [refs[iq.index(t)] for t in firsts.values()])
        os.remove(mpath)
        for c in ctl:
            red = c.get("mismatch") == 1
            print(f"  CONTROL {fname} {c.get('ty')} {c['name']}: one flipped bit -> mismatches={c.get('mismatch')} max_abs={c.get('max_abs')} => {'RED (control works)' if red else 'CONTROL FAILED'}")
            if not red: sys.exit(2)
    res = compare(path, iq, refs)
    per = collections.defaultdict(lambda: dict(tensors=0, elems=0, bit_exact=0, mismatch=0, max_abs=0.0, errors=0, shape_bad=0))
    for c in res:
        ty = c.get("ty") or next(t.tensor_type.name for t in iq if t.name == c["name"]); p = per[ty]; p["tensors"] += 1
        if "err" in c or not c["size_ok"]: p["errors"] += 1; print("  ERROR", c); continue
        p["elems"] += c["n"]; p["bit_exact"] += c["bit_exact"]; p["mismatch"] += c["mismatch"]; p["max_abs"] = max(p["max_abs"], c["max_abs"])
        if not c["shape_ok"]: p["shape_bad"] += 1
        if c["mismatch"]: print("  MISMATCH", c)
    ex = res[0]; print(f"  shape convention: core {ex['core_shape']} vs gguf-py ne {ex['ggufpy_ne']} ({ex['name']})")
    for ty, p in per.items():
        green = p["tensors"] == want[ty] and p["mismatch"] == 0 and p["errors"] == 0 and p["shape_bad"] == 0 and p["bit_exact"] == p["elems"]
        ok_all &= green
        print(f"{'GREEN' if green else 'RED'} ({fname}, {ty}, aprender-core) tensors={p['tensors']}/{want[ty]} elems={p['elems']} bit_exact={p['bit_exact']} mismatches={p['mismatch']} max_abs={p['max_abs']:.3g} errors={p['errors']} shape_disagree={p['shape_bad']}")
        report[f"{fname}|{ty}"] = dict(p, census=want[ty], verdict="GREEN" if green else "RED")
    ok_all &= set(per) == set(want)
json.dump(report, open(os.path.join(HERE, "compare_core_report.json"), "w"), indent=1)
shutil.rmtree(OUT, ignore_errors=True)
sys.exit(0 if ok_all else 1)
