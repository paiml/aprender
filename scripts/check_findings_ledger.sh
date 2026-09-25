#!/usr/bin/env bash
# check_findings_ledger.sh — a defect a session finds goes to the findings
# ledger, not to a new GitHub issue (APR-EPIC-001 v1.3 intake control,
# contracts/findings-ledger-v1.yaml, #4455).
#
# WHY THIS EXISTS
# ---------------
# Sessions opened an issue for every defect they found, so issues were created
# faster than PRs closed them and the backlog only grew. Now only the cop mints
# issues, and only when the target epic has budget. A session appends ONE JSON
# line per finding to docs/findings/<file>.jsonl, and the cop mints from there.
# The ledger is only useful if every row can be triaged without asking its
# author anything, so every row must be complete:
#
#   {"id": "...", "title": "...", "evidence": "...", "repro": "...",
#    "suspected_epic": "...", "severity": "P0|P1|P2|P3",
#    "found_at_sha": "<7-40 hex>"}
#
# A row must have all seven keys as non-empty strings, each given once (no
# extras, so a typo such as `severty` cannot slip through). severity is one of P0..P3,
# found_at_sha is 7-40 lowercase hex, and id matches [A-Za-z0-9._-]+ and is
# unique across every ledger file. A ledger with zero rows is RED (vacuous),
# never a pass.
#
#   bash scripts/check_findings_ledger.sh              # case table, then docs/findings/
#   bash scripts/check_findings_ledger.sh <dir>        # validate <dir>/*.jsonl only
#   bash scripts/check_findings_ledger.sh --self-test  # case table only
#
# Exit: 0 every row valid · 1 a row is invalid, or the ledger is empty · 2 usage
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEDGER_DIR="docs/findings"

# check_dir <dir> -> prints one line per defect, then a verdict; exit 0|1
check_dir() {
    python3 - "$1" <<'PY'
import json, pathlib, re, sys

KEYS = ("id", "title", "evidence", "repro", "suspected_epic", "severity", "found_at_sha")
d = pathlib.Path(sys.argv[1])
bad, seen, rows = [], {}, 0
for f in sorted(d.glob("*.jsonl")):
    for n, raw in enumerate(f.read_text().splitlines(), 1):
        if not raw.strip():
            continue
        rows += 1
        where = f"{f.name}:{n}"
        dup_keys = []
        def pairs(kv):
            seen_k = set()
            for k, _ in kv:
                if k in seen_k:
                    dup_keys.append(k)
                seen_k.add(k)
            return dict(kv)
        try:
            r = json.loads(raw, object_pairs_hook=pairs)
        except json.JSONDecodeError as e:
            bad.append(f"{where}: not JSON ({e.msg})")
            continue
        if not isinstance(r, dict):
            bad.append(f"{where}: not a JSON object")
            continue
        if dup_keys:
            bad.append(f"{where}: key(s) given twice {','.join(sorted(set(dup_keys)))} (json keeps only the last)")
        missing = [k for k in KEYS if not (isinstance(r.get(k), str) and r[k].strip())]
        extra = sorted(set(r) - set(KEYS))
        if missing:
            bad.append(f"{where}: missing or empty {','.join(missing)}")
        if extra:
            bad.append(f"{where}: unknown key(s) {','.join(extra)}")
        if isinstance(r.get("severity"), str) and r["severity"] and r["severity"] not in ("P0", "P1", "P2", "P3"):
            bad.append(f"{where}: severity {r['severity']!r} not in P0..P3")
        sha = r.get("found_at_sha")
        if isinstance(sha, str) and sha and not re.fullmatch(r"[0-9a-f]{7,40}", sha):
            bad.append(f"{where}: found_at_sha {sha!r} is not 7-40 lowercase hex")
        rid = r.get("id")
        if isinstance(rid, str) and rid:
            if not re.fullmatch(r"[A-Za-z0-9._-]+", rid):
                bad.append(f"{where}: id {rid!r} has characters outside [A-Za-z0-9._-]")
            elif rid in seen:
                bad.append(f"{where}: duplicate id {rid!r} (first at {seen[rid]})")
            else:
                seen[rid] = where
for b in bad:
    print(f"FAIL {b}")
if bad:
    print(f"FAIL findings ledger: {len(bad)} defect(s) in {rows} row(s) under {d}")
    sys.exit(1)
if rows == 0:
    print(f"FAIL findings ledger: 0 rows under {d} (vacuous, not a pass)")
    sys.exit(1)
print(f"OK: findings ledger: {rows} row(s) under {d}, all complete")
PY
}

SELF_N=0
SELF_FAILS=0
self_test() {
    local good
    SELF_TMP="$(mktemp -d)"
    trap 'rm -rf "${SELF_TMP:?}"' EXIT
    local tmp="$SELF_TMP"
    good='{"id":"F-1","title":"t","evidence":"e","repro":"r","suspected_epic":"#1","severity":"P1","found_at_sha":"00058476c"}'
    # case NAME WANT MARKER LINE... : each case is its own ledger directory
    case_() {
        local name="$1" want="$2" marker="$3" out rc=0
        shift 3
        mkdir -p "$tmp/$name"
        printf '%s\n' "$@" > "$tmp/$name/a.jsonl"
        out="$(check_dir "$tmp/$name")" || rc=$?
        SELF_N=$((SELF_N + 1))
        if [ "$rc" -ne "$want" ] || ! grep -qF -- "$marker" <<< "$out"; then
            printf 'self-test FAIL %s: want rc=%s marker %q, got rc=%s:\n%s\n' "$name" "$want" "$marker" "$rc" "$out" >&2
            SELF_FAILS=$((SELF_FAILS + 1))
        fi
    }
    case_ good 0 'OK: findings ledger: 1 row' "$good"
    case_ blank-lines-ignored 0 '1 row(s)' '' "$good" ''
    case_ two-rows 0 '2 row(s)' "$good" "${good/F-1/F-2}"
    case_ empty 1 '0 rows' ''
    case_ not-json 1 'not JSON' '{"id":'
    case_ not-object 1 'not a JSON object' '["F-1"]'
    case_ missing-repro 1 'missing or empty repro' "${good/\"repro\":\"r\",/}"
    case_ empty-evidence 1 'missing or empty evidence' "${good/\"evidence\":\"e\"/\"evidence\":\" \"}"
    case_ number-epic 1 'missing or empty suspected_epic' "${good/\"#1\"/1}"
    case_ typo-key 1 'unknown key(s) severty' "${good/\"severity\":\"P1\"/\"severity\":\"P1\",\"severty\":\"P1\"}"
    case_ bad-severity 1 "severity 'high'" "${good/P1/high}"
    case_ bad-sha 1 'not 7-40 lowercase hex' "${good/00058476c/HEAD}"
    case_ short-sha 1 'not 7-40 lowercase hex' "${good/00058476c/000584}"
    case_ bad-id 1 'characters outside' "${good/F-1/F 1}"
    case_ duplicate-key 1 'key(s) given twice severity' "${good/\"severity\":\"P1\"/\"severity\":\"P9\",\"severity\":\"P1\"}"
    case_ duplicate-id 1 "duplicate id 'F-1'" "$good" "$good"
    # a duplicate across two FILES is still a duplicate
    mkdir -p "$tmp/dup-files"
    printf '%s\n' "$good" > "$tmp/dup-files/a.jsonl"
    printf '%s\n' "$good" > "$tmp/dup-files/b.jsonl"
    local out rc=0
    out="$(check_dir "$tmp/dup-files")" || rc=$?
    SELF_N=$((SELF_N + 1))
    if [ "$rc" -ne 1 ] || ! grep -qF "b.jsonl:1: duplicate id 'F-1' (first at a.jsonl:1)" <<< "$out"; then
        printf 'self-test FAIL dup-files: rc=%s\n%s\n' "$rc" "$out" >&2
        SELF_FAILS=$((SELF_FAILS + 1))
    fi
    # a file that is not *.jsonl is not ledger: README.md beside a good row
    mkdir -p "$tmp/readme"
    printf '%s\n' "$good" > "$tmp/readme/a.jsonl"
    printf 'not json\n' > "$tmp/readme/README.md"
    rc=0
    out="$(check_dir "$tmp/readme")" || rc=$?
    SELF_N=$((SELF_N + 1))
    if [ "$rc" -ne 0 ]; then
        printf 'self-test FAIL readme-ignored: rc=%s\n%s\n' "$rc" "$out" >&2
        SELF_FAILS=$((SELF_FAILS + 1))
    fi
    if [ "$SELF_FAILS" -gt 0 ]; then
        printf 'self-test FAILED: %s of %s case(s).\n' "$SELF_FAILS" "$SELF_N" >&2
        return 1
    fi
    printf 'self-test OK: %s case(s).\n' "$SELF_N"
}

main() {
    case "${1:-}" in
        --self-test) self_test ;;
        -h | --help) sed -n '2,29p' "${BASH_SOURCE[0]}" ;;
        -*) printf 'usage: check_findings_ledger.sh [<dir> | --self-test]\n' >&2; exit 2 ;;
        '') self_test; check_dir "$ROOT/$LEDGER_DIR" ;;
        *) [ -d "$1" ] || { printf 'check_findings_ledger: no such directory: %s\n' "$1" >&2; exit 2; }
           check_dir "$1" ;;
    esac
}

main "$@"
