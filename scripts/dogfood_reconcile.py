#!/usr/bin/env python3
"""Reconcile the shipped CLI surface against docs/audits/surface_audit.csv.

This is G2.2 of the apr-dogfood protocol and the executable half of
contracts/apr-dogfood-coverage-v1.yaml F-DOGCOV-002.

WHY BOTH DIRECTIONS ARE REPORTED SEPARATELY
-------------------------------------------
A ledger is not a list. The two mismatch directions mean opposite things and
have opposite remedies:

    in the BINARY, not in the LEDGER  -> the ledger went stale; a shipped
                                         feature is unaccounted for, and the
                                         coverage RATIO is inflated because the
                                         denominator is short
    in the LEDGER, not in the BINARY  -> the surface shrank unnoticed; either a
                                         command was deleted without updating
                                         the ledger, or it is hidden behind a
                                         cargo feature

Silently "fixing" the CSV to match the binary erases which of those happened,
which is exactly what turns a ledger back into a list. So this script never
edits the CSV. It reports, and exits non-zero.

SCOPE IS DECLARED, NOT IMPLIED
------------------------------
`--help` can authoritatively enumerate the subcommand TREE and nothing else.
Rows describing flags, HTTP routes, MCP tools, REPL verbs and annotated
positional forms are OUT of scope and are counted in the summary rather than
dropped -- an uncounted exclusion is how a gate quietly stops measuring.

Usage:
    bash scripts/dogfood_surfaces.sh --emit-features > /tmp/runtime.txt
    python3 scripts/dogfood_reconcile.py /tmp/runtime.txt
"""
import csv
import collections
import sys

CSV_PATH = "docs/audits/surface_audit.csv"
HTTP_VERBS = ("GET ", "POST ", "PUT ", "DELETE ", "PATCH ", "HEAD ")

# In-scope ledger rows a DEFAULT cargo build cannot show. Each was proven by
# rebuilding with the feature and watching the subcommand appear -- not inferred
# from reading a #[cfg] attribute, because an attribute says what the source
# intends and a rebuild says what the binary does.
#
#   dev      -> apr mono {publish, shims, audit, archive}
#   hf-hub   -> alimentar {hub push, import hf}   (+ the `apr data x` mirror)
#   doctest  -> alimentar doctest {extract, merge} (+ the `apr data x` mirror)
#   eval     -> trueno-rag eval {7 verbs}          (+ the `apr rag` mirror)
#
# ENUMERATED, never a bare count: a tolerance of "<= 28 mismatches" would absorb
# a real deletion. Anything not on this list turns the gate RED.
FEATURE_GATED = {
    "apr mono archive", "apr mono audit", "apr mono publish", "apr mono shims",
    "alimentar hub push", "alimentar import hf",
    "alimentar doctest extract", "alimentar doctest merge",
    "apr data x hub push", "apr data x import hf",
    "apr data x doctest extract", "apr data x doctest merge",
    "trueno-rag eval compare", "trueno-rag eval gate", "trueno-rag eval generate",
    "trueno-rag eval judge", "trueno-rag eval metrics", "trueno-rag eval retrieve",
    "trueno-rag eval sample",
    "apr rag eval compare", "apr rag eval gate", "apr rag eval generate",
    "apr rag eval judge", "apr rag eval metrics", "apr rag eval retrieve",
    "apr rag eval sample",
}

# The emitter inherits dogfood_surfaces.sh's `grep -vE '^(help)$'` filter, so the
# literal `help` subcommand never appears in the runtime set even though clap
# does advertise it and the ledger records it.
EMITTER_FILTERED = {"apr sim help", "simular help"}

ALLOWED_ABSENT = FEATURE_GATED | EMITTER_FILTERED


def load_runtime(path):
    """Runtime (binary, feature) pairs, plus the emitter's own metadata lines."""
    runtime, unprobed, featureset = set(), [], "<unstated>"
    for line in open(path, encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        head, _, rest = line.partition("\t")
        if head == "UNPROBED":
            unprobed.append(rest)
        elif head == "FEATURESET":
            featureset = rest
        else:
            runtime.add((head, rest))
    return runtime, unprobed, featureset


def out_of_scope_reason(binary, feature):
    """Why `--help` cannot authoritatively enumerate this row, or None.

    Returning a REASON rather than a bool is what keeps the exclusions counted.
    A bare `if in_scope` would collapse six distinct populations into one
    number, and an uncounted exclusion is how a gate quietly stops measuring.
    """
    if feature.startswith(HTTP_VERBS):
        return "http route (live-server only)"
    if feature.startswith("mcp:"):
        return "mcp tool (needs tools/list)"
    if not (feature == binary or feature.startswith(binary + " ")):
        return "non-invocation prose"
    tail = feature[len(binary):]
    if not tail.strip():
        return "bare binary invocation"
    if any(t.startswith("-") for t in tail.split()):
        return "flag variant (not a subcommand)"
    if any(c in feature for c in "(<["):
        return "annotated / positional form"
    return None


def classify(rows):
    """Split ledger rows into the emitter's scope and the counted remainder."""
    in_scope, out_of_scope = set(), collections.Counter()
    for r in rows:
        binary, feature = r["binary"], r["feature"]
        reason = out_of_scope_reason(binary, feature)
        if reason is None:
            in_scope.add((binary, feature))
        else:
            out_of_scope[reason] += 1
    return in_scope, out_of_scope


def reconcile(rows, runtime):
    """Both mismatch directions, with tree-internal nodes excluded from A."""
    in_scope, out_of_scope = classify(rows)
    every_feature = {(r["binary"], r["feature"]) for r in rows}

    direction_a = []
    for binary, feature in sorted(runtime - in_scope):
        # The ledger records LEAF commands. `apr cgp` is a real tree node whose
        # children `apr cgp contract generate` etc. are recorded, so the parent
        # is not a gap -- flagging it would bury the real findings in 52 false
        # ones and get the gate switched off.
        if any(f.startswith(feature + " ") for (b, f) in every_feature if b == binary):
            continue
        direction_a.append(feature)

    direction_b = [f for _, f in sorted(in_scope - runtime) if f not in ALLOWED_ABSENT]
    return direction_a, direction_b, in_scope, out_of_scope


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    runtime, unprobed, featureset = load_runtime(sys.argv[1])
    rows = list(csv.DictReader(open(CSV_PATH, newline="", encoding="utf-8")))
    direction_a, direction_b, in_scope, out_of_scope = reconcile(rows, runtime)

    print(f"ledger rows          : {len(rows)}  ({CSV_PATH})")
    print(f"runtime features     : {len(runtime)}   featureset: {featureset}")
    print(f"in the emitter scope : {len(in_scope)}")
    print(f"outside its scope    : {sum(out_of_scope.values())} (counted, not dropped)")
    for label, n in out_of_scope.most_common():
        print(f"      {label:34} {n}")

    if unprobed:
        print()
        print("UNPROBED -- cargo DECLARES these binaries but did not build them.")
        print("They ship to anyone who enables the feature, and no probe has ever run:")
        for u in unprobed:
            print(f"      {u}")

    print()
    print(f"=== in the BINARY, not in the LEDGER (ledger is stale): {len(direction_a)} ===")
    for f in direction_a:
        print(f"      {f}")
    print(f"=== in the LEDGER, not in the BINARY (surface shrank): {len(direction_b)} ===")
    for f in direction_b:
        print(f"      {f}")

    if direction_a or direction_b:
        print()
        print("RECONCILIATION FAILED. Record WHICH DIRECTION moved before editing")
        print("the CSV; a silent sync destroys the evidence the ledger exists for.")
        return 1
    print()
    print("RECONCILED: both directions empty within the declared scope.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
