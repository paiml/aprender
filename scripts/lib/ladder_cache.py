#!/usr/bin/env python3
"""ladder_cache.py — reuse a cell that the SAME binary already proved green (#4520 step 5).

A rerun of one binary must not re-read every held model off the disk: a cell whose key matches a
green row in the host's previous receipt is copied, not re-measured. The key is exact, never fuzzy:

  (host, apr_sha, apr_bin_sha256, contract_sha256, cell id, model sha256, backends)

apr_bin_sha256 is the binary's own bytes (a rebuild at the same HEAD is a different binary until
proved otherwise); contract_sha256 is the ladder contract (a changed judge re-judges everything).
A receipt without any of these fields is a miss, so no receipt written before this cache can seed it.
Only a row that is green with no budget violation is reused. The copy carries "cached_from" naming
the run that measured it, so a cached green is never mistaken for a fresh measurement.

CLI:  ladder_cache.py <prev-receipt> <host> <apr_sha> <apr_bin_sha256> <contract_sha256> <id> <model_sha256> <backends-csv>
      -> prints the reused row (one JSON line) and exits 0 on a hit; exits 1 on a miss (prints why on stderr).
"""
import json
import sys


def lookup(prev, host, apr_sha, bin_sha, contract_sha, rid, model_sha, backends):
    for k, want in (("host", host), ("apr_sha", apr_sha), ("apr_bin_sha256", bin_sha),
                    ("contract_sha256", contract_sha)):
        if not want or want == "unknown" or prev.get(k) != want:
            return None, f"{k} differs ({prev.get(k)!r} vs {want!r})"
    for r in prev.get("rungs") or []:
        if r.get("id") != rid:
            continue
        if r.get("sha256") != model_sha:
            return None, "model sha256 differs"
        if r.get("green") is not True or r.get("budget_violations"):
            return None, "previous row not green"
        if set((r.get("backends") or {}).keys()) != set(b for b in backends.split(",") if b):
            return None, "backend set differs"
        row = dict(r)
        # a row already copied keeps the date of the run that MEASURED it, not of the run that copied it
        row.setdefault("cached_from", {"date": prev.get("date"), "apr_sha": apr_sha, "apr_bin_sha256": bin_sha})
        return row, None
    return None, "cell not in previous receipt"


def main(argv):
    if len(argv) != 9:
        print(__doc__.strip().splitlines()[-2], file=sys.stderr)
        return 2
    try:
        prev = json.load(open(argv[1]))
    except (OSError, ValueError) as e:
        print(f"miss: no readable previous receipt ({e})", file=sys.stderr)
        return 1
    row, why = lookup(prev, *argv[2:9])
    if row is None:
        print(f"miss: {why}", file=sys.stderr)
        return 1
    print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
