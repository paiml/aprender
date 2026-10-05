#!/usr/bin/env bash
# ci_tools_dag_status_parity_test.sh -- the identical-output case table for
# `aprender-ci-tools dag-status` against scripts/lib/dag_status.py (#4786).
#
# Each case is a receipt tree plus the DAG rows as `[id, row]` JSON pairs, the form
# the callers send. The .py side is a driver over dag_status.py's own functions
# (derived_status per row in order, then d7_typed_status), printing the same compact,
# key-sorted JSON line the binary prints. stdout must be byte-identical and both must
# agree on success vs failure. dag_status.py is the EXTERNAL VALIDATOR, never a build
# step.
#
# Not vacuous (L25): the run fails if fewer cases ran than the table declares, and a
# planted mismatch must be reported as one before any real case counts.
#
# Usage: scripts/tests/ci_tools_dag_status_parity_test.sh   (CI_TOOLS_BIN=<path> to skip the build)
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT" || exit 1
PY="${PYTHON:-python3}"
EXPECTED_CASES=24

. scripts/ci_tools_bin.sh || exit 1
BIN="$CI_TOOLS_BIN"

tmp="$(mktemp -d)"
trap 'rm -rf "${tmp:?}"' EXIT
pass=0
fail=0

cat >"$tmp/driver.py" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[1])
import dag_status as ds
root = sys.argv[2]
pairs = json.load(sys.stdin)
status = [[rid, ds.derived_status(root, r)] for rid, r in pairs]
d7 = ds.d7_typed_status(root, {rid: r for rid, r in pairs})
print(json.dumps({"status": status, "d7": d7}, separators=(",", ":"), sort_keys=True, ensure_ascii=False))
PY

# same ROOTDIR ROWS_JSON -> 0 when both sides agree on status class and stdout bytes
same() {
    local root="$1" rows="$2" py_s=0 rs_s=0
    printf '%s' "$rows" >"$tmp/in.json"
    "$PY" "$tmp/driver.py" "$ROOT/scripts/lib" "$root" <"$tmp/in.json" >"$tmp/py.out" 2>/dev/null || py_s=1
    "$BIN" dag-status --root "$root" <"$tmp/in.json" >"$tmp/rs.out" 2>/dev/null || rs_s=1
    [[ "$py_s" == "$rs_s" ]] && cmp -s "$tmp/py.out" "$tmp/rs.out"
}

check() { # NAME ROOTDIR ROWS_JSON
    if same "$2" "$3"; then
        pass=$((pass + 1))
    else
        fail=$((fail + 1))
        echo "FAIL: $1" >&2
        diff "$tmp/py.out" "$tmp/rs.out" >&2 || true
    fi
}

# receipt NAME BYTES: write docs/audits/impl-NAME-receipt.md under $R (printf %b escapes)
R="$tmp/root"
mkdir -p "$R/docs/audits"
receipt() { printf '%b' "$2" >"$R/docs/audits/impl-$1-receipt.md"; }
receipt C '---\nstatus: complete\n---\n'
receipt P '---\nstatus: partial\n---\n'
receipt Q '---\nstatus: "complete"\n---\n'
receipt S "---\nstatus: ' com plete '\n---\n"
receipt CR '---\r\nstatus: complete\r\n---\r\n'
receipt BOM '\xef\xbb\xbf---\nstatus: complete\n---\n'
receipt LATE '---\ntitle: x\n---\nstatus: complete\n'
receipt FIRST '---\nstatus: done\nstatus: complete\n---\n'
receipt NOFM 'status: complete\n'
receipt FS '---\nstatus:\x1ccomplete\x1c\n---\n'
receipt UNC '---\nstatus: complete\n'
receipt EMPTY ''
receipt BAD '---\nstatus: \xff\n---\n'
receipt 7 '---\nstatus: complete\n---\n'
receipt True '---\nstatus: complete\n---\n'
mkdir -p "$R/docs/audits/impl-DIR-receipt.md"

# --- the planted mismatch must be caught before any real case counts -------------
"$PY" "$tmp/driver.py" "$ROOT/scripts/lib" "$R" <<<'[["a",{"pmat_id":"C"}]]' >"$tmp/py.out"
printf '{"d7":[],"status":[["a","open"]]}\n' >"$tmp/rs.out"
if cmp -s "$tmp/py.out" "$tmp/rs.out"; then
    echo "FAIL: the planted mismatch (complete vs open) compared identical" >&2
    exit 1
fi

row() { printf '[["r",{"pmat_id":%s}]]' "$1"; }
check "complete" "$R" "$(row '"C"')"
check "partial is open" "$R" "$(row '"P"')"
check "double-quoted value" "$R" "$(row '"Q"')"
check "inner and edge spaces, single quotes" "$R" "$(row '"S"')"
check "CRLF: ---\\r is no fence" "$R" "$(row '"CR"')"
check "BOM before the fence" "$R" "$(row '"BOM"')"
check "status after the front matter" "$R" "$(row '"LATE"')"
check "first status line wins" "$R" "$(row '"FIRST"')"
check "no front matter" "$R" "$(row '"NOFM"')"
check "\\x1c counts as whitespace" "$R" "$(row '"FS"')"
check "unclosed front matter" "$R" "$(row '"UNC"')"
check "empty receipt" "$R" "$(row '"EMPTY"')"
check "receipt path is a directory" "$R" "$(row '"DIR"')"
check "missing receipt" "$R" "$(row '"NONE"')"
check "no pmat_id, empty, null, 0" "$R" '[["a",{}],["b",{"pmat_id":""}],["c",{"pmat_id":null}],["d",{"pmat_id":0}]]'
check "int and bool pmat_id name their receipt" "$R" '[["a",{"pmat_id":7}],["b",{"pmat_id":true}]]'
check "D7: typed agrees, typed disagrees, typed partial" "$R" \
    '[["a",{"pmat_id":"C","status":"complete"}],["b",{"pmat_id":"P","status":"complete"}],["c",{"pmat_id":"P","status":"partial"}]]'
check "D7: typed bool, null, int, list, sorted dict" "$R" \
    '[["a",{"pmat_id":"C","status":true}],["b",{"status":null}],["c",{"pmat_id":"C","status":1}],["d",{"status":["x",1.5]}],["e",{"status":{"a":1,"b":"c"}}]]'
check "D7: no pmat_id prints None" "$R" '[["a",{"status":"complete"}]]'
check "int and non-ASCII row ids, order kept" "$R" '[["z",{"pmat_id":"C"}],[3,{"pmat_id":"P"}],["é—\u001c",{"status":"x"}]]'
check "no rows" "$R" '[]'
check "BAD INPUT: receipt is not UTF-8" "$R" "$(row '"BAD"')"
check "BAD INPUT: row is not a mapping" "$R" '[["a",["pmat_id","C"]]]'
check "BAD INPUT: stdin is not JSON" "$R" '{nope'

ran=$((pass + fail))
echo "ci_tools_dag_status_parity: $pass/$ran identical (declared $EXPECTED_CASES)"
if [[ "$ran" -ne "$EXPECTED_CASES" ]]; then
    echo "FAIL: ran $ran cases, the table declares $EXPECTED_CASES" >&2
    exit 1
fi
[[ "$fail" -eq 0 ]]
