#!/usr/bin/env bash
# check_dag_invariants.sh — the PP-066 obligation DAG is data, and its
# invariants are checked by machine (PMAT-987, G-4, #2902; spec §4 C10, §5 G-4).
#
# WHY THIS EXISTS
# ---------------
# The 0.66 report's slack claim was checked by hand and missed the physical
# queue; the review of 2026-09-04 found W-G scheduled before W-B on gx10 and
# W-H expiring the day of the row it should block. A DAG whose invariants are
# prose is re-checked by every reader and by none of them the same way. So
# the rules live in scripts/lib/dag_invariants.py (D1..D6, one RED and one
# GREEN row each below) and this file is the CI wiring plus the case table.
#
#   bash scripts/check_dag_invariants.sh [<dag.yaml>] [--min-slack-days N] [--today YYYY-MM-DD]
#   bash scripts/check_dag_invariants.sh --selftest
#
# Defaults: dag = docs/specifications/pp-066-dag.yaml, min-slack-days = 6
# (spec §2, never a number invented here). A missing DAG file is exit 2 — the
# box cannot answer — never a passing DAG.
#
# NOT `pmat comply check --rule obligation-dag`: pmat 3.37.0 has no such
# rule (measured 2026-09-05). This checker is the gate until a pmat ticket
# lands the rule; the row's DAG entry says so.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LIB="$ROOT/scripts/lib/dag_invariants.py"
DEFAULT_DAG="docs/specifications/pp-066-dag.yaml"

usage() {
    printf 'usage: check_dag_invariants.sh [<dag.yaml>] [--min-slack-days N] [--today YYYY-MM-DD]\n' >&2
    printf '       check_dag_invariants.sh --selftest\n' >&2
    exit 2
}

[ -f "$LIB" ] || { printf 'check_dag_invariants: ENV - %s is missing\n' "$LIB" >&2; exit 2; }
command -v python3 >/dev/null 2>&1 || { printf 'check_dag_invariants: ENV - python3 is missing\n' >&2; exit 2; }

# ---------------------------------------------------------------------------
# --selftest: a 6-row synthetic DAG, mutated one rule at a time, BOTH
# polarities. Every fixture is written under mktemp -d and removed.
# ---------------------------------------------------------------------------
if [ "${1:-}" = "--selftest" ]; then
    TD=$(mktemp -d "${TMPDIR:-/tmp}/dag-selftest.XXXXXX")
    # SEC011: never `rm -rf` an unvalidated variable. Same idiom as
    # check_pr_review_arm4.sh: the victim must be non-empty, not `/`, and
    # carry the fixture prefix, checked again on the line that removes it.
    safe_rm_scratch() {
        local victim=${1:-} must=${2:-}
        [ -n "$victim" ] || return 0
        [ -n "$must" ]   || return 0
        [ "$victim" != "/" ] || return 0
        case "$victim" in
          *"$must"*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
          *) return 0 ;;
        esac
    }
    cleanup() { safe_rm_scratch "$TD" 'dag-selftest.'; }
    trap cleanup EXIT
    good="$TD/good.yaml"
    cat > "$good" <<'EOF'
schema: pp-066-dag/v1
epic: 2873
generated: '2026-09-05'
min_slack_days: 6
host_queues:
  gx10: [I-1, I-15, T-0]
  lambda: [R-4, S-1]
rows:
- id: I-1
  lane: '0.66'
  blockers: []
  owner: serve
  expiry: '2026-09-19'
  status: open
- id: I-15
  lane: '0.66'
  blockers: [I-1]
  owner: perf-gate
  expiry: {anchor: I-1, days: 7}
  status: open
- id: T-0
  lane: '0.66'
  blockers: [I-15]
  owner: perf-gate
  expiry: '2026-10-03'
  status: open
- id: R-4
  lane: '0.66'
  blockers: []
  owner: perf-gate
  expiry: '2026-09-19'
  status: open
- id: S-1
  lane: '0.66'
  blockers: [I-1]
  owner: perf-gate
  expiry: '2026-10-16'
  status: open
- id: W-H
  lane: '0.67'
  blockers: [S-1]
  owner: perf-gate
  expiry: '2026-10-17'
  status: open
EOF
    n=0; red=0
    row() { # row <id> <want rc> <label> <fixture> [<root holding docs/audits/impl-*-receipt.md>]
        local id=$1 want=$2 label=$3 f=$4 root=${5:-$TD/root} rc=0
        n=$((n + 1))
        python3 "$LIB" check "$f" --min-slack-days 6 --today 2026-09-05 --root "$root" >"$TD/out.$n" 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then
            printf 'ok    row %-2s %-6s rc=%s  %s\n' "$n" "$id" "$rc" "$label"
        else
            printf 'FAIL  row %-2s %-6s rc=%s (wanted %s)  %s\n' "$n" "$id" "$rc" "$want" "$label"
            sed 's/^/        /' "$TD/out.$n"
            red=1
        fi
    }
    mut() { # mut <name> <python edit over the doc>  -> writes $TD/<name>.yaml
        python3 - "$good" "$TD/$1.yaml" "$2" <<'PY'
import sys, yaml
src, dst, edit = sys.argv[1], sys.argv[2], sys.argv[3]
doc = yaml.safe_load(open(src))
rows = {r["id"]: r for r in doc["rows"]}
exec(edit)
yaml.safe_dump(doc, open(dst, "w"), sort_keys=False)
PY
    }
    row D0 0 "the good DAG holds every rule" "$good"
    mut d1 'rows["T-0"]["blockers"].append("NOPE")';                       row D1 1 "a blocker that is not a row" "$TD/d1.yaml"
    mut d2 'rows["I-1"]["blockers"].append("T-0")';                        row D2 1 "a cycle I-1 -> T-0 -> I-15 -> I-1" "$TD/d2.yaml"
    mut d3 'rows["W-H"]["expiry"] = "2026-10-16"; rows["W-H"]["lane"] = "0.66"'; row D3 1 "zero slack: W-H expires the day of its blocker S-1" "$TD/d3.yaml"
    mut d3ok 'rows["W-H"]["expiry"] = "2026-10-16"';                       row D3 0 "the same pair is not gated on the 0.67 lane" "$TD/d3ok.yaml"
    mut d4 'doc["host_queues"]["gx10"] = ["I-1", "T-0", "I-15"]';          row D4 1 "gx10 queue inversion: T-0 (10-03) before I-15 (09-26)" "$TD/d4.yaml"
    mut d5 'rows["R-4"]["owner"] = ""';                                    row D5 1 "a row with no owner" "$TD/d5.yaml"
    mut d6a 'rows["R-4"]["expiry"] = {"anchor": "I-1", "days": 7, "date": "2026-09-19"}'; row D6 1 "two expiry forms on one row" "$TD/d6a.yaml"
    mut d6b 'del rows["R-4"]["expiry"]';                                   row D6 1 "a row with no expiry" "$TD/d6b.yaml"
    mut d6c 'rows["I-15"]["expiry"] = {"anchor": "GHOST", "days": 7}';     row D6 1 "an anchor that is not a row" "$TD/d6c.yaml"
    mut rep 'rows["R-4"]["expiry"] = "2026-09-01"';                        row REP 0 "a past-expiry row is REPORTED, not a violation here" "$TD/rep.yaml"
    grep -q "REPORT past-expiry" "$TD/out.$n" || { printf 'FAIL  row %-2s the past-expiry report line is missing\n' "$n"; red=1; }
    # D7 (G-11, PMAT-1062): status is DERIVED from docs/audits/impl-<pmat_id>-receipt.md; a typed status is at most a cache
    mkdir -p "$TD/root/docs/audits"; printf -- '---\nstatus: complete\n---\n' > "$TD/root/docs/audits/impl-PMAT-7-receipt.md"; printf -- '---\nstatus: partial\n---\n' > "$TD/root/docs/audits/impl-PMAT-8-receipt.md"
    mut d7a 'rows["R-4"]["status"] = "complete"';                                       row D7 1 "typed status: complete on a row with no receipt (the registered mutation)" "$TD/d7a.yaml"
    mut d7b 'rows["R-4"]["pmat_id"] = "PMAT-7"; rows["R-4"]["status"] = "complete"';    row D7 0 "typed status: complete agrees with a complete receipt (a cache, tolerated)" "$TD/d7b.yaml"
    mut d7c 'rows["R-4"]["pmat_id"] = "PMAT-8"; rows["R-4"]["status"] = "complete"';    row D7 1 "typed status: complete over a receipt that says partial" "$TD/d7c.yaml"
    mut d7d 'del rows["R-4"]["status"]; rows["R-4"]["pmat_id"] = "PMAT-7"; rows["R-4"]["expiry"] = "2026-09-01"'; row D7 0 "no typed status: a complete receipt makes a past-expiry row NOT expired (derived)" "$TD/d7d.yaml"
    if grep -q 'expired' "$TD/out.$n"; then printf 'FAIL  row %-2s D7     the complete receipt was not read: the row still reports as expired\n' "$n"; fails=1; fi
    printf -- '---\r\nstatus: complete\r\n---\r\n' > "$TD/root/docs/audits/impl-PMAT-9-receipt.md"
    mut d7e 'rows["R-4"]["pmat_id"] = "PMAT-9"; rows["R-4"]["status"] = "complete"';    row D7 1 "a CRLF receipt is no front matter for bash (head -n1 keeps the CR) and none for python too: typed complete is RED" "$TD/d7e.yaml"
    printf '%s/%s rows\n' "$((n - red))" "$n"
    [ "$red" = 0 ] || exit 1
    exit 0
fi

# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------
DAG="$DEFAULT_DAG"; ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --min-slack-days|--today) [ $# -ge 2 ] || usage; ARGS+=("$1" "$2"); shift 2 ;;
        --help|-h) usage ;;
        --*) usage ;;
        *) DAG=$1; shift ;;
    esac
done
[ -f "$ROOT/$DAG" ] || [ -f "$DAG" ] || { printf 'check_dag_invariants: ENV - %s is missing (the box cannot answer; a missing DAG is not a passing one)\n' "$DAG" >&2; exit 2; }
[ -f "$DAG" ] || DAG="$ROOT/$DAG"
rc=0
python3 "$LIB" check "$DAG" "${ARGS[@]+"${ARGS[@]}"}" || rc=$?
if [ "$rc" = 0 ]; then
    printf 'PASS  every DAG invariant holds (D1 edges, D2 acyclic, D3 slack, D4 queues, D5 owner, D6 expiry form, D7 status derived from receipts)\n'
else
    printf 'FAIL  a DAG invariant is violated (rc=%s). Amend the DAG with a dated row under `amendments:` naming who, why, date — never by editing a blocker away.\n' "$rc" >&2
fi
exit "$rc"
