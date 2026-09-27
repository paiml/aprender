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

    # LA-02 lookahead-liveness-v1 ---------------------------------------------------
    # A fixture repo whose handoffs were updated at $UPD; the clock is pinned at $NOW.
    local NOW=2026-09-27T12:00:00Z R="$T/repo" H="$T/hb"
    handoffs() {  # handoffs UPDATED -- write the three handoff files into $R
        mkdir -p "$R/docs/lookahead"
        for t in 0.71 0.72 0.73; do
            printf '# %s\n```json lookahead-handoff\n{"train": "%s", "updated": "%s"}\n```\n' "$t" "$t" "$1" > "$R/docs/lookahead/$t.md"
        done
    }
    handoffs 2026-09-27T11:00:00Z
    cp "$T/good.json" "$T/live.json"
    row "tick/all-live-clean" 0 '"I3": "ok"' -- la tick "$T/live.json" --now "$NOW" --root "$R" --hb-dir "$H"
    # a planted dead worker: L2's heartbeat is 3 h old
    state 0.70 "$(slot 0.71 kreg)" "$(slot 0.72 la-72 2026-09-27T09:00:00Z)" "$(slot 0.73 la-73)" > "$T/dead.json"
    row "tick/dead-3h-worker-respawn-due" 1 'ACTION RESPAWN L2' -- la tick "$T/dead.json" --now "$NOW" --root "$R" --hb-dir "$H"
    row "tick/dead-names-the-handoff" 1 'lookahead-worker.md with SLOT=L2 TRAIN=0.72' -- la tick "$T/dead.json" --now "$NOW" --root "$R" --hb-dir "$H"
    # a heartbeat of exactly 2 h is dead (I3 is "< 2 h")
    state 0.70 "$(slot 0.71 kreg)" "$(slot 0.72 la-72 2026-09-27T10:00:00Z)" "$(slot 0.73 la-73)" > "$T/edge.json"
    row "tick/exactly-2h-is-dead" 1 'ACTION RESPAWN L2' -- la tick "$T/edge.json" --now "$NOW" --root "$R" --hb-dir "$H"
    # respawned within one tick: respawn, then the next tick is clean
    cp "$T/dead.json" "$T/dead2.json"
    row "respawn/dead-slot-refilled" 0 'RESPAWNED L2 train 0.72: la-72 -> la-72b' -- la respawn "$T/dead2.json" --slot L2 --worker la-72b --now "$NOW" --hb-dir "$H"
    row "respawn/next-tick-clean" 0 '"I3": "ok"' -- la tick "$T/dead2.json" --now "$NOW" --root "$R" --hb-dir "$H"
    row "respawn/state-still-legal" 0 '^OK ' -- la validate "$T/dead2.json"
    # a planted duplicate: respawning a LIVE slot is refused and the state is untouched
    cp "$T/good.json" "$T/dup.json"
    row "respawn/live-slot-duplicate-refused" 1 'REFUSED: L1 is held by live worker kreg' -- la respawn "$T/dup.json" --slot L1 --worker intruder --now "$NOW" --hb-dir "$H"
    row "respawn/refusal-left-state-untouched" 0 '^' -- cmp "$T/good.json" "$T/dup.json"
    # a worker already in a slot may not take a second one
    cp "$T/dead.json" "$T/dup2.json"
    row "respawn/worker-in-two-slots-refused" 1 'REFUSED: kreg already holds L1' -- la respawn "$T/dup2.json" --slot L2 --worker kreg --now "$NOW" --hb-dir "$H"
    # the worker's own heartbeat file keeps a slot live; a foreign worker cannot write one
    cp "$T/dead.json" "$T/hb.json"
    row "heartbeat/holder-writes" 0 'HEARTBEAT L2 la-72' -- la heartbeat "$T/hb.json" --slot L2 --worker la-72 --now 2026-09-27T11:50:00Z --hb-dir "$H"
    row "heartbeat/file-keeps-slot-live" 0 '"I3": "ok"' -- la tick "$T/hb.json" --now "$NOW" --root "$R" --hb-dir "$H"
    row "heartbeat/foreign-worker-refused" 1 'REFUSED: intruder does not hold L2' -- la heartbeat "$T/hb.json" --slot L2 --worker intruder --now "$NOW" --hb-dir "$H"
    # a replaced worker's heartbeat file does not keep its successor's slot live
    printf '{"slot": "L2", "worker": "ghost", "at": "%s"}\n' "$NOW" > "$H/L2.json"
    row "heartbeat/replaced-worker-file-ignored" 1 'ACTION RESPAWN L2' -- la tick "$T/hb.json" --now "$NOW" --root "$R" --hb-dir "$H"
    # I4: a handoff not updated within 24 h, and a missing one
    handoffs 2026-09-26T11:00:00Z
    row "tick/handoff-25h-stale" 1 'ACTION HANDOFF-STALE L1' -- la tick "$T/live.json" --now "$NOW" --root "$R" --hb-dir "$T/nohb"
    handoffs 2026-09-27T11:00:00Z
    rm -f "${R:?}/docs/lookahead/0.73.md"
    row "tick/handoff-missing" 1 'ACTION HANDOFF-STALE L3' -- la tick "$T/live.json" --now "$NOW" --root "$R" --hb-dir "$T/nohb"
    row "tick/illegal-state-repair-first" 1 'ACTION REPAIR I2' -- la tick "$T/one-in-two.json" --now "$NOW" --root "$R" --hb-dir "$T/nohb"

    # LA-03 the standing prompt ------------------------------------------------------
    row "prompt/renders-for-slot" 0 'aprender train 0.72 \(slot L2\)' -- la prompt --slot L2 --train 0.72 --root "$ROOT"
    row "prompt/no-placeholder-left" 0 '^[^$]*$' -- sh -c "python3 '$SUBJECT' prompt --slot L3 --train 0.73 --root '$ROOT' | tr -d '\n'"
    row "prompt/tick-launch-line-matches" 1 'Run docs/prompts/lookahead-worker.md with SLOT=L2 TRAIN=0.72' -- la tick "$T/dead.json" --now "$NOW" --root "$R" --hb-dir "$T/nohb"
    # a planted drifted prompt file (step 5 deleted) is refused
    local P="$T/prompt-root"
    mkdir -p "$P/docs/prompts" "$P/docs/specifications"
    cp "$ROOT/docs/specifications/APR-LOOKAHEAD-001-rolling-epic-workers.md" "$P/docs/specifications/"
    grep -v 'Respect your open-PR budget' "$ROOT/docs/prompts/lookahead-worker.md" > "$P/docs/prompts/lookahead-worker.md"
    row "prompt/drifted-from-spec-refused" 1 'VIOLATION PROMPT: .* differs' -- la prompt --slot L1 --train 0.71 --root "$P"
    # a prompt file without the launch line tick prints is refused
    grep -v 'SLOT=L1 TRAIN=0.71 autonomously' "$ROOT/docs/prompts/lookahead-worker.md" > "$P/docs/prompts/lookahead-worker.md"
    row "prompt/no-launch-line-refused" 1 'lacks the launch line' -- la prompt --slot L1 --train 0.71 --root "$P"
    rm -f "${P:?}/docs/prompts/lookahead-worker.md"
    row "prompt/absent-cannot-judge" 2 'CANNOT-JUDGE' -- la prompt --slot L1 --train 0.71 --root "$P"

    # LA-04 lookahead-pr-budget-v1 --------------------------------------------------
    # pr NUM MILESTONE DRAFT CREATED-HOUR FILE -- one PR in the plain snapshot shape
    pr() { printf '{"number": %s, "milestone": "%s", "draft": %s, "created": "2026-09-27T%s:00:00Z", "files": ["%s"]}' "$@"; }
    snap() { local cap=$1 q=$2; shift 2; local IFS=,; printf '{"cap": %s, "queue": [%s], "prs": [%s]}\n' "$cap" "$q" "$*"; }
    snap 10 "" "$(pr 1 0.71.0 false 08 crates/a.rs)" "$(pr 2 0.71.0 false 09 crates/b.rs)" \
        "$(pr 3 0.72.0 false 09 docs/x.md)" "$(pr 4 0.73.0 false 09 docs/y.md)" "$(pr 5 0.70.0 false 07 crates/c.rs)" > "$T/budget-ok.json"
    row "budget/within-budget-clean" 0 '^OK: every look-ahead PR' -- la pr-budget "$T/good.json" --prs "$T/budget-ok.json"
    # a planted third L1 PR: the newest one is blocked, the older two stand
    snap 10 "" "$(pr 1 0.71.0 false 08 crates/a.rs)" "$(pr 2 0.71.0 false 09 crates/b.rs)" "$(pr 6 0.71.0 false 10 crates/d.rs)" > "$T/l1-three.json"
    row "budget/third-l1-pr-blocked" 1 '^BLOCK #6: L1 holds 3 open PRs, budget 2' -- la pr-budget "$T/good.json" --prs "$T/l1-three.json"
    row "budget/older-l1-prs-stand" 0 '^[^#]*#6[^#]*$' -- sh -c "python3 '$SUBJECT' pr-budget '$T/good.json' --prs '$T/l1-three.json' | tr '\n' ' '"
    snap 10 "" "$(pr 3 0.72.0 false 09 docs/x.md)" "$(pr 7 0.72.0 true 10 docs/z.md)" > "$T/l2-two.json"
    row "budget/second-l2-pr-blocked" 1 '^BLOCK #7: L2 holds 2' -- la pr-budget "$T/good.json" --prs "$T/l2-two.json"
    # what a slot may land
    snap 10 "" "$(pr 3 0.72.0 false 09 crates/spike.rs)" > "$T/l2-code.json"
    row "budget/l2-code-must-be-draft" 1 '^DRAFT #3' -- la pr-budget "$T/good.json" --prs "$T/l2-code.json"
    snap 10 "" "$(pr 3 0.72.0 true 09 crates/spike.rs)" > "$T/l2-draft.json"
    row "budget/l2-draft-code-ok" 0 '^OK' -- la pr-budget "$T/good.json" --prs "$T/l2-draft.json"
    snap 10 "" "$(pr 4 0.73.0 true 09 crates/spike.rs)" > "$T/l3-code.json"
    row "budget/l3-code-blocked" 1 '^BLOCK #4: L3 lands specs and docs only' -- la pr-budget "$T/good.json" --prs "$T/l3-code.json"
    # a planted look-ahead PR ahead of a train PR in the queue is re-queued
    snap 10 "3,5" "$(pr 3 0.72.0 false 09 docs/x.md)" "$(pr 5 0.70.0 false 07 crates/c.rs)" > "$T/queue-ahead.json"
    row "budget/lookahead-ahead-of-train-requeued" 1 '^REQUEUE #3: L2 PR queued ahead of a train-0.70 PR' -- la pr-budget "$T/good.json" --prs "$T/queue-ahead.json"
    snap 10 "5,3" "$(pr 3 0.72.0 false 09 docs/x.md)" "$(pr 5 0.70.0 false 07 crates/c.rs)" > "$T/queue-behind.json"
    row "budget/lookahead-behind-train-ok" 0 '^OK' -- la pr-budget "$T/good.json" --prs "$T/queue-behind.json"
    snap 10 "5,1,2" "$(pr 1 0.71.0 false 08 crates/a.rs)" "$(pr 2 0.71.0 false 09 crates/b.rs)" "$(pr 5 0.70.0 false 07 crates/c.rs)" > "$T/queue-two.json"
    row "budget/two-same-train-in-queue-requeued" 1 '^REQUEUE #2: a second train-0.71' -- la pr-budget "$T/good.json" --prs "$T/queue-two.json"
    # the repo cap: L3 yields first, then L2; train and unslotted PRs never yield
    snap 3 "" "$(pr 5 0.70.0 false 07 crates/c.rs)" "$(pr 8 "" false 07 crates/e.rs)" "$(pr 1 0.71.0 false 08 crates/a.rs)" \
        "$(pr 3 0.72.0 false 09 docs/x.md)" "$(pr 4 0.73.0 false 09 docs/y.md)" > "$T/cap.json"
    row "budget/at-cap-l3-yields-first" 1 '^YIELD #4: .*L3 yields' -- la pr-budget "$T/good.json" --prs "$T/cap.json"
    row "budget/at-cap-then-l2" 1 '^YIELD #3: .*L2 yields' -- la pr-budget "$T/good.json" --prs "$T/cap.json"
    row "budget/at-cap-train-never-yields" 0 '^$' -- sh -c "python3 '$SUBJECT' pr-budget '$T/good.json' --prs '$T/cap.json' | grep -E 'YIELD #(5|8|1):' || true"
    # raw `gh pr list --json` rows are accepted as they come
    printf '[{"number": 1, "milestone": {"title": "0.71.0"}, "isDraft": false, "createdAt": "2026-09-27T08:00:00Z", "files": [{"path": "a.rs"}]}, {"number": 2, "milestone": {"title": "0.71.0"}, "isDraft": false, "createdAt": "2026-09-27T09:00:00Z", "files": []}, {"number": 6, "milestone": {"title": "0.71.0"}, "isDraft": false, "createdAt": "2026-09-27T10:00:00Z", "files": []}]\n' > "$T/gh.json"
    row "budget/gh-json-accepted" 1 '^BLOCK #6' -- la pr-budget "$T/good.json" --prs "$T/gh.json" --cap 10
    printf '{"prs": []}\n' > "$T/nocap.json"
    row "budget/no-cap-cannot-judge" 2 'CANNOT-JUDGE' -- la pr-budget "$T/good.json" --prs "$T/nocap.json"

    # LA-05 lookahead-readiness-v1 --------------------------------------------------
    # ready_repo DIR [MUTATION] -- a fixture repo whose 0.71 handoff is fully Ready;
    # MUTATION plants one defect (see the python below).
    ready_repo() {
        python3 - "$1" "${2:-}" <<'FIX'
import json, os, sys
root, mut = sys.argv[1], sys.argv[2]
def w(rel, text):
    os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
    open(os.path.join(root, rel), "w").write(text)
w("docs/specifications/s.md", "# spec\n")
w("contracts/c-v1.yaml", "falsification_tests:\n  - id: FALSIFY-C-001\n")
w("docs/baselines/b.json", "{}\n")
w("docs/lookahead/risks-0.73.md", "# risks\n")
w("docs/specifications/06x-release-schedule.md",
  "| 0.71 | **Verbs Are Fast** | #3598 |\n| 0.72 | **Train What You Serve** | #4000 |\n")
row = lambda i, ev: {"id": f"R{i}", "ev": ev, "spec": "docs/specifications/s.md#r", "contract": "contracts/c-v1.yaml",
                     "falsifier": "FALSIFY-C-001", "baseline": "docs/baselines/b.json", "owner": "kreg"}
rows = [row(i, 100 - i) for i in range(10)] + [{"id": "R-low", "ev": 1}]  # an unready 11th row is outside the top 10
h1 = {"train": "0.71", "epic": 3598, "theme": "Verbs Are Fast", "updated": "2026-09-27T11:00:00Z",
      "exit_criteria": {"V1": "open", "V5": {"status": "met", "receipt": "docs/baselines/b.json"}},
      "rows": rows, "first5_prs": [{"title": f"p{i}", "owner": "kreg"} for i in range(5)],
      "rulings_pending": [{"q": "scope?", "request": "#9999"}]}
h2 = {"train": "0.72", "theme": "Train What You Serve", "updated": "2026-09-27T11:00:00Z", "exit_criteria": {},
      "rows": [row(i, 50 - i) for i in range(20)]}
h3 = {"train": "0.73", "theme": "Runs Everywhere", "updated": "2026-09-27T11:00:00Z", "exit_criteria": {},
      "risk_register": "docs/lookahead/risks-0.73.md", "measurement_plan": "docs/lookahead/absent.md"}
if mut == "no-contract":
    rows[3]["contract"] = "contracts/missing-v1.yaml"
if mut == "falsifier-absent":
    rows[3]["falsifier"] = "FALSIFY-C-999"
if mut == "met-no-receipt":
    h1["exit_criteria"]["V5"] = "met"
if mut == "question-unfiled":
    h1["rulings_pending"].append({"q": "theme?"})
if mut == "four-prs":
    h1["first5_prs"].pop()
if mut == "spec-outside-tree":
    rows[0]["spec"] = "../../etc/passwd"
for t, h in (("0.71", h1), ("0.72", h2), ("0.73", h3)):
    w(f"docs/lookahead/{t}.md", "```json lookahead-handoff\n" + json.dumps(h) + "\n```\n")
w("epics.json", json.dumps({"3598": {"tickets": 12, "groomed": 11 if mut == "epic-11-of-12" else 12}}))
FIX
    }
    local D m
    D="$T/ready"; ready_repo "$D"
    row "ready/fully-ready-n+1" 0 '^READY 0.71$' -- la readiness "$T/good.json" --root "$D" --epics "$D/epics.json"
    row "ready/top10-rendered" 0 '"top10_ready": "10/10"' -- la readiness "$T/good.json" --root "$D" --epics "$D/epics.json"
    row "ready/n+2-derived" 0 '"top5_specified": "5/5"' -- la readiness "$T/good.json" --root "$D" --epics "$D/epics.json"
    row "ready/n+2-theme-ruled-from-schedule" 0 '"theme_ruled": true' -- la readiness "$T/good.json" --root "$D" --epics "$D/epics.json"
    row "ready/n+3-absent-plan-is-false" 0 '"measurement_plan": false' -- la readiness "$T/good.json" --root "$D" --epics "$D/epics.json"
    row "ready/met-with-receipt-counts" 0 '"V5": "met"' -- la readiness "$T/good.json" --root "$D" --epics "$D/epics.json"
    # the planted top-10 row without a contract shows NOT READY, naming the row
    D="$T/r-nc"; ready_repo "$D" no-contract
    row "ready/top10-row-without-contract-not-ready" 1 '^NOT READY 0.71: row R3: contract' -- la readiness "$T/good.json" --root "$D" --epics "$D/epics.json"
    for m in falsifier-absent met-no-receipt question-unfiled four-prs spec-outside-tree epic-11-of-12; do
        D="$T/r-$m"; ready_repo "$D" "$m"
        row "ready/$m-not-ready" 1 '^NOT READY 0.71' -- la readiness "$T/good.json" --root "$D" --epics "$D/epics.json"
    done
    row "ready/epics-absent-unverified" 1 'NOT READY 0.71: epic grooming unverified' -- la readiness "$T/good.json" --root "$T/ready"
    row "ready/handoff-missing-cannot-judge" 2 'CANNOT-JUDGE' -- la readiness "$T/good.json" --root "$T/empty-root"
    # train-start latency receipt
    printf '[{"number": 11, "milestone": {"title": "0.71.0"}, "mergedAt": "2026-09-27T19:30:00Z"}, {"number": 12, "milestone": {"title": "0.71.0"}, "mergedAt": "2026-09-27T18:40:00Z"}, {"number": 13, "milestone": {"title": "0.70.0"}, "mergedAt": "2026-09-27T18:10:00Z"}, {"number": 14, "milestone": {"title": "0.71.0"}, "mergedAt": "2026-09-27T17:00:00Z"}]\n' > "$T/merged.json"
    row "latency/first-n+1-pr-after-tag" 0 '"first_pr": 12, "latency_min": 40' -- la latency --tag v0.70.0 --tag-at 2026-09-27T18:00:00Z --train 0.71 --merged "$T/merged.json"
    printf '[{"number": 11, "milestone": {"title": "0.71.0"}, "mergedAt": "2026-09-27T19:30:00Z"}]\n' > "$T/merged-late.json"
    row "latency/over-target-not-met" 1 '"latency_min": 150, "target_min": 120, "met": false' -- la latency --tag v0.70.0 --tag-at 2026-09-27T17:00:00Z --train 0.71 --merged "$T/merged-late.json"
    row "latency/none-yet-pending" 1 '"first_pr": null' -- la latency --tag v0.70.0 --tag-at 2026-09-27T20:00:00Z --train 0.71 --merged "$T/merged.json"
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
  "i3-age-unchecked|        if age >= HEARTBEAT_MAX_MIN:|        if False:"
  "live-slot-respawnable|    if age < HEARTBEAT_MAX_MIN:|    if False:"
  "foreign-heartbeat-accepted|    if opts[\"worker\"] != owner:|    if False:"
  "hb-file-any-worker|        if isinstance(hb, dict) and hb.get(\"worker\") == rec[\"worker\"]:|        if isinstance(hb, dict):"
  "i4-unchecked|            stale = ho_age_h >= HANDOFF_MAX_H or ho.get(\"train\") != rec[\"train\"]|            stale = False"
  "prompt-drift-unchecked|    if block != spec_block:|    if False:"
  "prompt-launch-unchecked|    if launch_prompt(\"L1\", \"0.71\") not in text:|    if False:"
  "slot-budget-unchecked|        for p in mine[PR_BUDGET[s]:]:|        for p in mine[99:]:"
  "l2-draft-unchecked|        if s == \"L2\" and code and not p.get(\"draft\"):|        if False:"
  "queue-order-unchecked|        if i < last_train:|        if False:"
  "yield-order-reversed|YIELD_ORDER = [\"L3\", \"L2\", \"L1\"]|YIELD_ORDER = [\"L1\", \"L2\", \"L3\"]"
  "row-contract-unchecked|        out.append(f\"contract {contract!r} not a file under contracts/\")|        pass"
  "falsifier-unchecked|                out.append(f\"falsifier {fid!r} not in {contract}\")|                pass"
  "met-trusted|        elif status == \"met\" and not|        elif False and not"
  "epic-unchecked|    if groomed != 100:|    if False:"
  "latency-before-tag-counted|        if when >= tag_at and|        if True and"
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
