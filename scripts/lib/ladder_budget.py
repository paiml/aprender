#!/usr/bin/env python3
"""ladder_budget.py — judge ladder_meter.py records against the ladder's declared budgets (#4520 step 4).

A budget is an assertion, not a note: going over it is RED on that cell, so thrashing the host is a
defect the gate reports rather than a side effect nobody measured.

  load_budgets(contract) -> dict      the `ladder.budgets` block, validated; raises ValueError if absent
                                      or malformed (a missing budget must decline, never read as "no limit")
  judge(records, budgets) -> list     one violation dict per (record or cell) over budget

Rules (every number comes from the contract):
  header verb   bytes_read <= header_bytes_read_max                 (#3761: header-only means header bytes)
  every verb    peak_rss_bytes <= peak_rss_max_factor * file_bytes + peak_rss_slack_bytes
  non-serve     wall_s <= wall_s_max                                (serve is bounded by serve_health)
  per cell      sum(bytes_read) <= cell_bytes_read_max_factor * file_bytes   (load once, check many)
A record with no cell (a lock probe) is not judged. A cell record with no file_bytes is itself a
violation: a budget that cannot be computed is not a budget that passed.

CLI:  ladder_budget.py <contract.yaml> <meter.jsonl>   -> prints violations as JSON lines; exit 1 if any,
      0 if none, 2 if the contract declares no valid budgets or the meter file is unreadable.
"""
import json
import sys

KEYS_NUM = ("header_bytes_read_max", "peak_rss_max_factor", "peak_rss_slack_bytes", "peak_rss_kill_factor",
            "wall_s_max", "cell_bytes_read_max_factor")


def load_budgets(contract):
    b = (contract.get("ladder") or {}).get("budgets")
    if not isinstance(b, dict):
        raise ValueError("ladder.budgets is absent")
    for k in KEYS_NUM:
        v = b.get(k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"ladder.budgets.{k} must be a positive number, got {v!r}")
    hv = b.get("header_verbs")
    if not isinstance(hv, list) or not hv or not all(isinstance(x, str) and x for x in hv):
        raise ValueError("ladder.budgets.header_verbs must be a non-empty list of verb names")
    return b


def judge(records, b):
    out, cells = [], {}
    for r in records:
        cell = r.get("cell")
        if not cell:
            continue
        fb, verb = r.get("file_bytes"), r.get("verb")
        if not isinstance(fb, int) or fb <= 0:
            out.append({"cell": cell, "verb": verb, "budget": "file_bytes",
                        "measured": fb, "limit": None, "why": "no file size recorded: the budget cannot be computed"})
            continue
        c = cells.setdefault(cell, {"file_bytes": fb, "bytes_read": 0, "calls": 0})
        c["bytes_read"] += int(r.get("bytes_read") or 0)
        c["calls"] += 1
        if verb in b["header_verbs"] and r.get("bytes_read", 0) > b["header_bytes_read_max"]:
            out.append({"cell": cell, "verb": verb, "budget": "header_bytes_read_max",
                        "measured": r["bytes_read"], "limit": b["header_bytes_read_max"],
                        "why": "a header-only verb read past the header (#3761)"})
        rss_max = int(b["peak_rss_max_factor"] * fb + b["peak_rss_slack_bytes"])
        if r.get("peak_rss_bytes", 0) > rss_max:
            out.append({"cell": cell, "verb": verb, "budget": "peak_rss",
                        "measured": r["peak_rss_bytes"], "limit": rss_max, "why": "peak RSS over budget"})
        if verb != "serve" and r.get("wall_s", 0) > b["wall_s_max"]:
            out.append({"cell": cell, "verb": verb, "budget": "wall_s_max",
                        "measured": r["wall_s"], "limit": b["wall_s_max"], "why": "wall time over budget"})
    for cell, c in sorted(cells.items()):
        lim = int(b["cell_bytes_read_max_factor"] * c["file_bytes"])
        if c["bytes_read"] > lim:
            out.append({"cell": cell, "verb": None, "budget": "cell_bytes_read",
                        "measured": c["bytes_read"], "limit": lim,
                        "why": f"the cell read {c['bytes_read'] / c['file_bytes']:.1f}x its model file off the disk "
                               f"over {c['calls']} calls: load once, check many"})
    return out


def main(argv):
    import yaml
    if len(argv) != 3:
        print("usage: ladder_budget.py <contract.yaml> <meter.jsonl>", file=sys.stderr)
        return 2
    try:
        b = load_budgets(yaml.safe_load(open(argv[1])))
        recs = [json.loads(l) for l in open(argv[2]) if l.strip()]
    except (OSError, ValueError) as e:
        print(f"decline: {e}", file=sys.stderr)
        return 2
    v = judge(recs, b)
    for x in v:
        print(json.dumps(x, sort_keys=True))
    return 1 if v else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
