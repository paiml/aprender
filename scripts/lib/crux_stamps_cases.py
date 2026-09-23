"""crux_stamps_cases.py -- hermetic case table + mutants for crux_stamps.py (#4051).

    python3 scripts/lib/crux_stamps_cases.py            # every row must hold
    python3 scripts/lib/crux_stamps_cases.py --mutants  # every mutant must break its NAMED row
"""

import importlib.util
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

CELLS = {
    "c/gpu.sh": {"cell": "c/gpu.sh", "t_req": 10.0, "t_acquired": 14.5, "lock": "gpu"},
    "c/cpu.sh": {"cell": "c/cpu.sh", "t_req": 10.0, "t_acquired": 10.2, "lock": "none"},
}
LINES = [
    {"line": "d/apr-p1", "cell": "c/gpu.sh", "t_start": 15.0, "t_end": 20.0, "n0": 0, "n1": 0},
    {"line": "d/hf-p1.driver", "cell": "c/gpu.sh", "t_start": 20.0, "t_end": 30.0, "n0": 1, "n1": 2},
    {"line": "d/apr-sweep", "cell": "c/gpu.sh", "t_start": 30.0, "t_end": 50.0, "n0": 2, "n1": 2},
    {"line": "d/apr-p10x", "cell": "c/cpu.sh", "t_start": 11.0, "t_end": 12.0, "n0": 5, "n1": 5},
    {"line": "d/llama-p3", "cell": "c/nolock.sh", "t_start": 60.0, "t_end": 61.0, "n0": 9, "n1": 9},
    {"line": "d/ollama-p1.driver", "cell": "c/gpu.sh", "t_start": 70.0, "t_end": 71.0, "n0": 6, "n1": 7},  # wrote a refusal
]


def rows():
    return [
        {"kind": "tok", "engine": "llama.cpp"},                                             # 0: not a gen row
        {"kind": "gen", "engine": "hf", "prompt_id": "p1", "stdout": "/elsewhere/hf.json"},  # 1: appended by the driver line
        {"kind": "gen", "engine": "apr", "prompt_id": "p1", "stdout": "d/apr-p1.out"},       # 2: cell_result, by its prefix
        {"kind": "gen", "engine": "apr", "prompt_id": "p2", "verb": "serve run", "stdout": "d/apr/p2-nonstream.json"},  # 3: serve sweep
        {"kind": "gen", "engine": "apr", "prompt_id": "p10", "stdout": "d/apr-p10.out"},     # 4: `d/apr-p1` is NOT its prefix
        {"kind": "gen", "engine": "apr", "prompt_id": "p10x", "stdout": "d/apr-p10x.out"},   # 5: the cpu lane (no lock)
        {"kind": "gen", "engine": "ollama", "prompt_id": "p1", "stdout": None, "refused": "lock not had"},  # 6: a refusal, even one a line wrote
        {"kind": "gen", "engine": "llama.cpp", "prompt_id": "p9", "stdout": "z/unknown.out"},  # 7: nothing accounts for it
        {"kind": "gen", "engine": "llama.cpp", "prompt_id": "p3", "stdout": "d/llama-p3.out"},  # 8: its cell wrote no record
    ]


def run(mod):
    rs = rows()
    mod.attach(rs, [dict(x) for x in LINES], {k: dict(v) for k, v in CELLS.items()})
    t = [r.get("timing") for r in rs]
    res = {}
    res["driver-row-by-range"] = (t[1] is not None and t[1]["t_start"] == 20.0 and t[1]["t_end"] == 30.0, t[1])
    res["result-row-by-prefix"] = (t[2] is not None and t[2]["t_start"] == 15.0 and t[2]["t_end"] == 20.0, t[2])
    res["serve-row-by-sweep-dir"] = (t[3] is not None and t[3]["t_start"] == 30.0 and t[3]["t_end"] == 50.0, t[3])
    res["prefix-needs-a-dot"] = (t[4] is None, t[4])
    res["gpu-wait-measured"] = (t[2] is not None and t[2]["lock"] == "gpu" and t[2]["lock_wait_s"] == 4.5, t[2])
    res["cpu-lane-lock-none-zero"] = (t[5] is not None and t[5]["lock"] == "none" and t[5]["lock_wait_s"] == 0.0, t[5])
    res["refused-row-unstamped"] = (t[6] is None, t[6])
    res["unattributable-row-unstamped"] = (t[7] is None, t[7])
    res["no-cell-record-invalid-wait"] = (t[8] is not None and t[8]["lock"] is None and t[8]["lock_wait_s"] is None, t[8])
    res["non-gen-untouched"] = (t[0] is None, t[0])
    return res


MUTANTS = [
    ("range-ignored", "    if len(by_range) == 1:\n        return by_range[0]\n", "", "driver-row-by-range"),
    ("prefix-ignored", 'hit = out.startswith(p + ".") or', 'hit = False or', "result-row-by-prefix"),
    ("sweep-ignored", ' or (p.endswith("-sweep") and out.startswith(p[:-len("-sweep")] + "/"))', "", "serve-row-by-sweep-dir"),
    ("dot-dropped", 'hit = out.startswith(p + ".") or', 'hit = out.startswith(p) or', "prefix-needs-a-dot"),
    ("wait-zeroed", 'wait = round(c["t_acquired"] - c["t_req"], 3)', "wait = 0.0", "gpu-wait-measured"),
    ("none-waits", 'if lock == "none":\n        wait = 0.0', 'if lock == "none":\n        wait = round(c["t_acquired"] - c["t_req"], 3)', "cpu-lane-lock-none-zero"),
    ("refused-stamped", 'if r.get("kind") != "gen" or r.get("refused") or r.get("timing") is not None:',
     'if r.get("kind") != "gen" or r.get("timing") is not None:', "refused-row-unstamped"),
    ("missing-cell-ok", "        wait = None   # the cell's lock record", "        lock, wait = \"gpu\", 0.0   # the cell's lock record", "no-cell-record-invalid-wait"),
]


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    src = os.path.join(HERE, "crux_stamps.py")
    res = run(load(src, "crux_stamps"))
    bad = [r for r, (ok, _) in res.items() if not ok]
    for r, (ok, got) in res.items():
        print("%s %-30s %s" % ("PASS" if ok else "FAIL", r, "" if ok else got))
    print("rows: %d pass, %d fail" % (len(res) - len(bad), len(bad)))
    if bad:
        return 1
    if "--mutants" not in sys.argv:
        return 0
    text, survived, tmp = open(src).read(), 0, tempfile.mkdtemp(prefix="crux-stamps-mut-")
    for name, old, new, row in MUTANTS:
        if text.count(old) != 1:
            print("MUTANT %-18s ANCHOR %d != 1" % (name, text.count(old)))
            survived += 1
            continue
        p = os.path.join(tmp, name + ".py")
        open(p, "w").write(text.replace(old, new))
        killed = not run(load(p, "m_" + name.replace("-", "_")))[row][0]
        print("MUTANT %-18s %s by %s" % (name, "KILLED" if killed else "SURVIVED", row))
        survived += not killed
    print("mutants: %d/%d killed by their named row" % (len(MUTANTS) - survived, len(MUTANTS)))
    return 1 if survived else 0


if __name__ == "__main__":
    sys.exit(main())
