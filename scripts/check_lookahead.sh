#!/usr/bin/env bash
# check_lookahead.sh -- the case table for scripts/lookahead/lookahead.py, the slot
# machinery of APR-LOOKAHEAD-001 (epic #4540). Each row plants a state (or a PR list,
# a budget reading, a rotation) in a throwaway dir, runs the REAL tool on it, and
# asserts the exit code and a line of output. The committed spec's invariants are
# only as real as the rows below that turn RED when they break.
#
# WHY A check_*.sh AND NOT lookahead.py --self-test: guard_tree.sh dispatches every
# `scripts/check_*.sh` in its --no-cargo step, so this table runs on every PR with no
# workflow edit (the same reasoning as check_batch_fold.sh).
#
#   check_lookahead.sh              the case table against the real tool
#   check_lookahead.sh --self-test  the mutants: each wrong tool must turn >=1 row RED
#
# Exit: 0 all rows green, 1 a row RED, 2 cannot run (no python3, tool missing).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
SUBJECT="${LOOKAHEAD_SUBJECT:-$ROOT/scripts/lookahead/lookahead.py}"
export LOOKAHEAD_SCHEMA="$ROOT/docs/lookahead/lookahead-state.schema.json"

rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
command -v python3 >/dev/null 2>&1 || { echo "ENV: python3 not found" >&2; exit 2; }
[ -f "$SUBJECT" ] || { echo "ENV: $SUBJECT not found" >&2; exit 2; }
[ -f "$LOOKAHEAD_SCHEMA" ] || { echo "ENV: $LOOKAHEAD_SCHEMA not found" >&2; exit 2; }

# ---- fixtures -------------------------------------------------------------------
# slot TRAIN WORKER [HEARTBEAT] -- one slot object
slot() {
    printf '{"train": "%s", "worker": "%s", "since": "2026-09-27T10:00:00Z", "heartbeat": "%s", "handoff": "docs/lookahead/%s.md"}' \
        "$1" "$2" "${3:-2026-09-27T11:00:00Z}" "$1"
}
# state CUR L1 L2 L3 -- a whole state, slots given as JSON objects
state() { printf '{"current_train": "%s", "slots": {"L1": %s, "L2": %s, "L3": %s}}\n' "$1" "$2" "$3" "$4"; }
good_state() { state 0.70 "$(slot 0.71 kreg)" "$(slot 0.72 la-72)" "$(slot 0.73 la-73)"; }

# ---- the table ------------------------------------------------------------------
FAILS=0; ROWS=0
# row NAME WANT_RC WANT_REGEX -- runs "$@" after `--`, compares rc and output
row() {
    local name=$1 want_rc=$2 want_re=$3; shift 4
    local out rc
    out=$("$@" 2>&1); rc=$?
    ROWS=$((ROWS + 1))
    if [ "$rc" -eq "$want_rc" ] && printf '%s\n' "$out" | grep -Eq -- "$want_re"; then
        printf 'GREEN %-44s rc=%s\n' "$name" "$rc"
    else
        FAILS=$((FAILS + 1))
        printf 'RED   %-44s rc=%s want rc=%s /%s/\n' "$name" "$rc" "$want_rc" "$want_re"
        printf '%s\n' "$out" | sed 's/^/      | /' | head -8
    fi
}
la() { python3 "$SUBJECT" "$@"; }

run_table() {
    local T=$1
    # LA-00 lookahead-state-v1 ------------------------------------------------------
    good_state > "$T/good.json"
    row "state/good-passes" 0 '^OK ' -- la validate "$T/good.json"
    # two workers, one slot: the key L1 twice (json.load alone keeps the last)
    printf '{"current_train": "0.70", "slots": {"L1": %s, "L1": %s, "L2": %s, "L3": %s}}\n' \
        "$(slot 0.71 kreg)" "$(slot 0.71 other)" "$(slot 0.72 la-72)" "$(slot 0.73 la-73)" > "$T/two-in-l1.json"
    row "state/two-workers-one-slot-fails" 1 'VIOLATION I2' -- la validate "$T/two-in-l1.json"
    # one worker, two slots
    state 0.70 "$(slot 0.71 kreg)" "$(slot 0.72 kreg)" "$(slot 0.73 la-73)" > "$T/one-in-two.json"
    row "state/one-worker-two-slots-fails" 1 'VIOLATION I2' -- la validate "$T/one-in-two.json"
    # a worker field that is a list is two workers in one slot, too
    state 0.70 "$(slot 0.71 kreg)" '{"train": "0.72", "worker": ["a","b"], "since": "2026-09-27T10:00:00Z", "heartbeat": "2026-09-27T10:00:00Z", "handoff": "docs/lookahead/0.72.md"}' "$(slot 0.73 la-73)" > "$T/list-worker.json"
    row "state/worker-list-fails" 1 'VIOLATION SCHEMA' -- la validate "$T/list-worker.json"
    # N+3 missing: L3 absent
    printf '{"current_train": "0.70", "slots": {"L1": %s, "L2": %s}}\n' "$(slot 0.71 kreg)" "$(slot 0.72 la-72)" > "$T/no-l3.json"
    row "state/missing-n+3-slot-fails" 1 "missing required 'L3'" -- la validate "$T/no-l3.json"
    # N+3 missing: three slots, but L3 re-covers N+2
    state 0.70 "$(slot 0.71 kreg)" "$(slot 0.72 la-72)" "$(slot 0.72 la-73)" > "$T/l3-is-n2.json"
    row "state/missing-n+3-train-fails" 1 'VIOLATION I1: L3 holds train 0.72, want 0.73' -- la validate "$T/l3-is-n2.json"
    # a rotation half-done: current moved, slots did not
    state 0.71 "$(slot 0.71 kreg)" "$(slot 0.72 la-72)" "$(slot 0.73 la-73)" > "$T/stale.json"
    row "state/current-moved-slots-stale-fails" 1 'VIOLATION I1' -- la validate "$T/stale.json"
    printf '{"current_train": "0.70", "slots": \n' > "$T/broken.json"
    row "state/unparseable-cannot-judge" 2 'CANNOT-JUDGE' -- la validate "$T/broken.json"
    row "state/absent-cannot-judge" 2 'CANNOT-JUDGE' -- la validate "$T/absent.json"
    # the committed schema is itself JSON, and the committed handoffs name their train
    row "schema/is-json" 0 '.' -- python3 -m json.tool "$LOOKAHEAD_SCHEMA"
    local t
    for t in 0.71 0.72 0.73; do
        row "handoff/$t-skeleton-present" 0 "\"train\": \"$t\"" -- cat "$ROOT/docs/lookahead/$t.md"
    done
}

run_all() {
    local T; T=$(mktemp -d) || return 2
    FAILS=0; ROWS=0
    run_table "$T"
    rmtree "$T"
    printf '%s rows, %s RED\n' "$ROWS" "$FAILS"
    [ "$FAILS" -eq 0 ]
}

# ---- mutants -------------------------------------------------------------------
# mutant NAME OLD NEW -- a copy of the tool with OLD replaced by NEW. The replacement
# must apply exactly once: a mutant that did not mutate would read as "killed".
MUTANTS=(
  "dup-keys-ignored|dup_sink.append(k)|pass"
  "i1-unchecked|if got != want:|if False:"
  "i2-unchecked|if w in owner:|if False:"
  "schema-ignored|out += [(\"SCHEMA\", e) for e in schema_errors(schema, schema, state)]|pass"
  "cannot-judge-passes|        return 2\n\n\nif __name__|        return 0\n\n\nif __name__"
)

self_test() {
    local real_rc killed=0 total=0 spec name old new M out
    run_all >/dev/null; real_rc=$?
    [ "$real_rc" -eq 0 ] || { echo "SELF-TEST: the real tool is not GREEN (rc=$real_rc) — fix that first"; return 1; }
    M=$(mktemp -d) || return 2
    for spec in "${MUTANTS[@]}"; do
        IFS='|' read -r name old new <<<"$spec"
        total=$((total + 1))
        if ! python3 - "$SUBJECT" "$M/$name.py" "$old" "$new" <<'PY'
import sys
src, dst, old, new = sys.argv[1], sys.argv[2], sys.argv[3].encode().decode("unicode_escape"), sys.argv[4].encode().decode("unicode_escape")
text = open(src, encoding="utf-8").read()
if text.count(old) != 1:
    sys.exit(f"mutation site found {text.count(old)} times")
open(dst, "w", encoding="utf-8").write(text.replace(old, new))
PY
        then echo "MUTANT $name: could not apply — the guard no longer knows the code"; rmtree "$M"; return 1; fi
        out=$(LOOKAHEAD_SUBJECT="$M/$name.py" bash "$0" 2>&1)
        if [ $? -ne 0 ]; then
            killed=$((killed + 1)); echo "KILLED   $name ($(printf '%s\n' "$out" | grep -c '^RED') rows RED)"
        else
            echo "SURVIVED $name"
        fi
    done
    rmtree "$M"
    echo "$killed/$total mutants killed"
    [ "$killed" -eq "$total" ]
}

case "${1:-}" in
    --self-test) self_test ;;
    "") run_all ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
