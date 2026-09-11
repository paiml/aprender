#!/usr/bin/env python3
import sys, os, argparse, re, tempfile, subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from scripts.lib.roadmap_diff import split_entries

def parse_id(eid):
    m = re.match(r"^([A-Za-z][A-Za-z0-9_]*)-([0-9]+)", eid)
    if not m:
        return ("", 0)
    num = m.group(2)
    if os.environ.get("ROADMAP_MERGE_MUTATE") == "1":
        return (m.group(1), num)
    return (m.group(1), int(num))

def _conflict(msg):
    sys.stderr.write(f"conflict: {msg}\n")
    sys.exit(1)

def resolve_header(base_pre, ours_pre, theirs_pre):
    if ours_pre == base_pre: return theirs_pre
    if theirs_pre == base_pre: return ours_pre
    if ours_pre == theirs_pre: return ours_pre
    return _conflict("header")

def _resolve_existing(eid, b, o, t):
    if o is not None and t is not None:
        if o == b: return t
        if t == b: return o
        if o == t: return o
        return _conflict(eid)
    if o is not None and o != b: return _conflict(eid)
    if t is not None and t != b: return _conflict(eid)
    return None

def resolve_entry(eid, b, o, t):
    if b is not None:
        return _resolve_existing(eid, b, o, t)
    if o is not None and t is not None and o != t:
        return _conflict(eid)
    return o or t

def sort_entries(final_entries, base_ids, ours_ids, theirs_ids):
    base_idx = {eid: i for i, eid in enumerate(base_ids)}
    keys = {}
    
    def _add_keys(seq, src_id):
        last = -1
        for i, eid in enumerate(seq):
            if eid in base_idx:
                last = base_idx[eid]
                keys[eid] = (last, 0, "", 0, 0)
            else:
                p, n = parse_id(eid)
                keys[eid] = (last, 1, p, n, src_id, i)

    _add_keys(base_ids, 0)
    _add_keys(ours_ids, 1)
    _add_keys(theirs_ids, 2)
    return sorted(list(final_entries.keys()), key=lambda x: keys[x])

def merge(b_txt, o_txt, t_txt):
    b_pre, b_ent = split_entries(b_txt)
    o_pre, o_ent = split_entries(o_txt)
    t_pre, t_ent = split_entries(t_txt)

    f_pre = resolve_header(b_pre, o_pre, t_pre)
    
    b_dict = {k: v for k, v in b_ent}
    o_dict = {k: v for k, v in o_ent}
    t_dict = {k: v for k, v in t_ent}

    f_ent = {}
    for eid in set(b_dict) | set(o_dict) | set(t_dict):
        val = resolve_entry(eid, b_dict.get(eid), o_dict.get(eid), t_dict.get(eid))
        if val is not None:
            f_ent[eid] = val

    sorted_ids = sort_entries(f_ent, [k for k, _ in b_ent], [k for k, _ in o_ent], [k for k, _ in t_ent])
    return f_pre + "".join(f_ent[k] for k in sorted_ids)

def run_selftest():
    h = "roadmap:\n"
    e1 = "- id: PMAT-1\n  a: 1\n"
    e2 = "- id: PMAT-2\n  a: 2\n"
    e3 = "- id: PMAT-3\n  a: 3\n"
    e10 = "- id: PMAT-10\n  a: 10\n"

    cases = [
        ("both-append", h+e1, h+e1+e2, h+e1+e3, False, False),
        ("edit", h+e1, h+e1.replace("a: 1", "a: 11"), h+e1, False, False),
        ("conflict", h+e1, h+e1.replace("a: 1", "a: 11"), h+e1.replace("a: 1", "a: 12"), True, False),
        ("del-unc", h+e1+e2, h+e1, h+e1+e2, False, False),
        ("del-edit", h+e1+e2, h+e1, h+e1+e2.replace("a: 2", "a: 22"), True, False),
        ("mut", h+e1, h+e1+e10, h+e1+e2, False, True),
    ]

    failed = 0
    for name, b, o, t, exp_err, mut in cases:
        print(f"RUN {name}")
        with tempfile.TemporaryDirectory() as td:
            bp, op, tp, out = [os.path.join(td, f) for f in ("b","o","t","out")]
            for p, content in zip((bp, op, tp), (b, o, t)):
                with open(p, "w") as f: f.write(content)
            
            env = os.environ.copy()
            if mut: env["ROADMAP_MERGE_MUTATE"] = "1"
            
            p = subprocess.run([sys.executable, __file__, bp, op, tp, "--out", out], env=env)
            if (p.returncode != 0) != exp_err:
                failed = 1; print(f"FAIL {name}: p.returncode={p.returncode}")
                continue
            if exp_err: continue
            
            subprocess.run(["git", "init", "-q"], cwd=td)
            chk = subprocess.run(["bash", os.path.abspath("scripts/check_roadmap_sorted.sh"), out], cwd=td, capture_output=True)
            if (chk.returncode != 0) != mut:
                failed = 1; print(f"FAIL {name}: p.returncode={p.returncode}")

    sys.exit(failed)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--out")
    p.add_argument("b", nargs="?")
    p.add_argument("o", nargs="?")
    p.add_argument("t", nargs="?")
    args, _ = p.parse_known_args()

    if args.selftest: return run_selftest()
    if not all([args.b, args.o, args.t]): return 2

    txts = []
    for f in (args.b, args.o, args.t):
        with open(f, "r", encoding="utf-8") as fh: txts.append(fh.read())

    out_text = merge(*txts)
    with open(args.out or args.o, "w", encoding="utf-8") as f: f.write(out_text)
    return 0

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.exit(main())
