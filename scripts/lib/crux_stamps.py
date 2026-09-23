"""crux_stamps.py -- attach #4051 timing stamps to a CRUX manifest's engine rows.

crux_inference_dogfood.sh runs every engine call as one LINE of a CELL script (cell_add), and every
cell under one lock acquisition (run_cell). Each writes a stamp record to `stamps.jsonl`:

  line  {"line": <prefix>, "cell": <cell script>, "t_start", "t_end", "n0", "n1"}
        the engine line's own wall span, and the manifest's line count before and after it
  cell  {"cell": <cell script>, "t_req", "t_acquired", "lock": "gpu"|"none"}
        when the cell asked for the lock, and when its script actually started under it

A manifest `gen` row is attributed to exactly one line, and nothing is guessed:
  - a row the line APPENDED itself (an engine driver writes its own row) sits at a manifest index in
    [n0, n1) of that line;
  - a row built after the cell from the line's files (cell_result, serve `rows`) names them: its
    `stdout` is `<prefix>.<ext>`, or for a serve sweep `<prefix minus -sweep>/...`.
The row then carries timing = the line's t_start/t_end plus its cell's lock ("gpu": lock_wait_s =
t_acquired - t_req; "none": 0). A refused row made no call and gets none. A row that no stamp
accounts for is left WITHOUT timing, and the judge (--require-timing) declines it by name.

    python3 scripts/lib/crux_stamps.py attach --manifest M --stamps S   # rewrites M in place
"""

import argparse
import json
import os
import sys


def load_stamps(path):
    lines, cells = [], {}
    try:
        fh = open(path, encoding="utf-8")
    except OSError:
        return lines, cells
    with fh:
        for ln in fh:
            try:
                x = json.loads(ln)
            except ValueError:
                continue
            if "line" in x:
                lines.append(x)
            elif "cell" in x:
                cells[x["cell"]] = x
    return lines, cells


def _owner(i, row, lines):
    """The one line that produced manifest row i, or None."""
    by_range = [s for s in lines if isinstance(s.get("n0"), int) and isinstance(s.get("n1"), int)
                and s["n0"] <= i < s["n1"]]
    if len(by_range) == 1:
        return by_range[0]
    out = row.get("stdout") or ""
    best = None
    for s in lines:
        p = s.get("line") or ""
        if not p:
            continue
        hit = out.startswith(p + ".") or (p.endswith("-sweep") and out.startswith(p[:-len("-sweep")] + "/"))
        if hit and (best is None or len(p) > len(best["line"])):
            best = s
    return best


def timing_for(line, cells):
    c = cells.get(line.get("cell")) or {}
    lock = c.get("lock")
    t0, t1 = line.get("t_start"), line.get("t_end")
    if lock == "none":
        wait = 0.0
    elif lock == "gpu" and isinstance(c.get("t_acquired"), (int, float)) and isinstance(c.get("t_req"), (int, float)):
        wait = round(c["t_acquired"] - c["t_req"], 3)
    else:
        wait = None   # the cell's lock record is missing: the judge refuses an invalid lock wait
    return {"t_start": t0, "t_end": t1, "t_acquired": c.get("t_acquired"), "lock": lock, "lock_wait_s": wait}


def attach(rows, lines, cells):
    """-> number of rows stamped. `rows` are the manifest's parsed lines, in order (mutated)."""
    n = 0
    for i, r in enumerate(rows):
        if r.get("kind") != "gen" or r.get("refused") or r.get("timing") is not None:
            continue
        s = _owner(i, r, lines)
        if s is None:
            continue
        r["timing"] = timing_for(s, cells)
        n += 1
    return n


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("attach")
    a.add_argument("--manifest", required=True)
    a.add_argument("--stamps", required=True)
    args = ap.parse_args(argv)
    with open(args.manifest, encoding="utf-8") as fh:
        raw = [ln for ln in fh if ln.strip()]
    rows = [json.loads(ln) for ln in raw]
    lines, cells = load_stamps(args.stamps)
    n = attach(rows, lines, cells)
    tmp = args.manifest + ".stamped"
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(r) + "\n" for r in rows))
    os.replace(tmp, args.manifest)
    gens = sum(1 for r in rows if r.get("kind") == "gen" and not r.get("refused"))
    print("crux_stamps: stamped %d of %d engine row(s) that ran" % (n, gens))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
