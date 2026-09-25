#!/usr/bin/env bash
# check_merge_queue_ruleset.sh — the main merge queue is what was RULED, and stays so (#4435).
#
# Ruling (FLOW-001 R-4, operator 2026-09-25 16:58 Madrid, verbatim): "Merge queue starts at
# B=8 the moment FLOW-04 merges and verifies on a real merge_group; K = all full runner sets;
# auto-rollback to B=1 if split rate > 2p or any unverified merge."
# R-2: "B = min(8, round(1/p))".
#
# The declaration is .github/merge-queue-declared.json. Three modes:
#
#   check [--live FILE]   D1 the declaration obeys the ruling: B = min(B_cap, round(runs/failures)),
#                            B_cap = 8, K = |runner_sets|, rollback = {to_B 1, factor 2, unverified on}.
#                         L1 the LIVE ruleset (gh api, or FILE) matches it: max_entries_to_merge = B
#                            (to_B when state is rolled_back), max_entries_to_build = B*K,
#                            grouping_strategy and merge_method as declared.
#   rollback-eval [--runs FILE]
#                         R1 over the last window_runs merge_group CI runs: split rate > factor*p
#                            (a failed group is a split) -> ROLLBACK.
#                         R2 a group CI passed (it merged) while pr-review-quorum on the SAME
#                            head_sha did not succeed -> an unverified merge -> ROLLBACK.
#   --self-test           both polarities of D1, L1, R1, R2 on fixtures; no network.
#
# Exit: 0 OK | 10 DRIFT or ROLLBACK REQUIRED | 3 NOT MEASURED (no gh, API refused, bad JSON:
# unmeasured is never green, and never a code a crash also produces) | 2 usage.
set -uo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
DECL=${MQ_DECLARED:-$REPO_ROOT/.github/merge-queue-declared.json}
REPO=${MQ_REPO:-paiml/aprender}

need() { command -v "$1" >/dev/null 2>&1 || { echo "NOT MEASURED: $1 is not on PATH" >&2; exit 3; }; }
need jq

# declared_ok <decl> -> prints each violation; status 0 when none
declared_ok() {
    jq -r '
      def rnd: (. + 0.5 | floor);
      . as $d
      | [ (if ($d.p.failures|type) != "number" or ($d.p.runs|type) != "number" or $d.p.runs < 1
             then "p: failures/runs missing or runs < 1" else empty end),
          (if $d.B_cap != 8 then "B_cap is \($d.B_cap), the ruling caps B at 8" else empty end),
          (($d.p.failures) as $f | ($d.p.runs) as $n
            | (if $f == 0 then $d.B_cap else ([$d.B_cap, ($n / $f | rnd)] | min) end) as $want
            | if $d.B != $want then "B is \($d.B), min(\($d.B_cap), round(1/p)) with p=\($f)/\($n) is \($want)" else empty end),
          (if ($d.runner_sets|type) != "array" or ($d.runner_sets|length) < 1 then "runner_sets empty"
           elif $d.K != ($d.runner_sets|length) then "K is \($d.K), runner_sets lists \($d.runner_sets|length)" else empty end),
          (if $d.rollback.to_B != 1 then "rollback.to_B is \($d.rollback.to_B), the ruling rolls back to 1" else empty end),
          (if $d.rollback.split_rate_factor_of_p != 2 then "rollback factor is \($d.rollback.split_rate_factor_of_p), the ruling is 2p" else empty end),
          (if $d.rollback.on_unverified_merge != true then "rollback.on_unverified_merge is not true" else empty end),
          (if ($d.rollback.window_runs|type) != "number" or $d.rollback.window_runs < 1 then "rollback.window_runs missing" else empty end),
          (if ($d.state != "active" and $d.state != "rolled_back") then "state is \($d.state), not active|rolled_back" else empty end)
        ] | .[]' "$1" | sed 's/^/  D1  /' | { grep . && return 1; return 0; }
}

# live_ok <decl> <live ruleset json> -> prints each drift; status 0 when none
live_ok() {
    jq -r --slurpfile d "$1" '
      ($d | first) as $d
      | ([.rules[]? | select(.type == "merge_queue") | .parameters] | first) as $q
      | (if $d.state == "rolled_back" then $d.rollback.to_B else $d.B end) as $b
      | if $q == null then "  L1  the live ruleset has no merge_queue rule"
        else
          (if $q.max_entries_to_merge != $b then "  L1  max_entries_to_merge is \($q.max_entries_to_merge), declared B is \($b) (state \($d.state))" else empty end),
          (if $q.max_entries_to_build != ($b * $d.K) then "  L1  max_entries_to_build is \($q.max_entries_to_build), declared B*K is \($b * $d.K)" else empty end),
          (if $q.grouping_strategy != $d.grouping_strategy then "  L1  grouping_strategy is \($q.grouping_strategy), declared \($d.grouping_strategy)" else empty end),
          (if $q.merge_method != $d.merge_method then "  L1  merge_method is \($q.merge_method), declared \($d.merge_method)" else empty end)
        end' "$2" | { grep . && return 1; return 0; }
}

# rollback_ok <decl> <runs json: [{name,head_sha,conclusion,created_at}]> -> prints the trigger; 0 when none
rollback_ok() {
    jq -r --slurpfile d "$1" '
      . as $all | ($d | first) as $d
      | ($d.p.failures / $d.p.runs) as $p
      | [ .[] | select(.name == "CI" and (.conclusion == "success" or .conclusion == "failure")) ]
      | sort_by(.created_at) | reverse | .[: $d.rollback.window_runs] as $ci
      | ($ci | length) as $n
      | ([($ci | .[]) | select(.conclusion == "failure")] | length) as $splits
      | (if $n > 0 and ($splits / $n) > ($d.rollback.split_rate_factor_of_p * $p)
           then "  R1  split rate \($splits)/\($n) > \($d.rollback.split_rate_factor_of_p)p = \($d.rollback.split_rate_factor_of_p * $p) -> ROLLBACK to B=\($d.rollback.to_B)"
           else empty end),
        ( ($ci | .[]) | select(.conclusion == "success") | .head_sha as $h
          | select(([($all | .[]) | select(.name == "pr-review-quorum" and .head_sha == $h and .conclusion == "success")] | length) == 0)
          | "  R2  merge_group \($h | .[:9]) passed CI with no successful pr-review-quorum on the same sha -> unverified merge -> ROLLBACK to B=\($d.rollback.to_B)")
      ' "$2" | { grep . && return 1; return 0; }
}

mode_check() {
    local live=${1:-} tmp rc=0
    declared_ok "$DECL" || rc=10
    if [ -z "$live" ]; then
        need gh
        tmp=$(mktemp) || exit 2
        live=$tmp
        gh api "repos/$REPO/rulesets/$(jq -r .ruleset_id "$DECL")" > "$live" 2>/dev/null \
            || { rm -f -- "${tmp:?}"; echo "NOT MEASURED: gh api could not read ruleset $(jq -r .ruleset_id "$DECL")" >&2; exit 3; }
    fi
    jq -e . "$live" >/dev/null 2>&1 || { echo "NOT MEASURED: $live is not JSON" >&2; exit 3; }
    live_ok "$DECL" "$live" || rc=10
    [ -n "${tmp:-}" ] && rm -f -- "${tmp:?}"
    if [ "$rc" = 0 ]; then
        echo "merge queue OK: B=$(jq .B "$DECL") K=$(jq .K "$DECL") state=$(jq -r .state "$DECL"), live ruleset matches"
    else
        echo "merge queue DRIFT (#4435): fix the ruleset or the declaration, with a ruling" >&2
    fi
    return "$rc"
}

mode_rollback() {
    local runs=${1:-} tmp
    if [ -z "$runs" ]; then
        need gh
        tmp=$(mktemp) || exit 2
        runs=$tmp
        gh api "repos/$REPO/actions/runs?event=merge_group&per_page=100" \
            --jq '[.workflow_runs[] | {name, head_sha, conclusion, created_at}]' > "$runs" 2>/dev/null \
            || { rm -f -- "${tmp:?}"; echo "NOT MEASURED: gh api could not list merge_group runs" >&2; exit 3; }
    fi
    jq -e 'type == "array"' "$runs" >/dev/null 2>&1 || { echo "NOT MEASURED: $runs is not a JSON array" >&2; exit 3; }
    local rc=0
    rollback_ok "$DECL" "$runs" || rc=10
    [ -n "${tmp:-}" ] && rm -f -- "${tmp:?}"
    [ "$rc" = 0 ] && echo "rollback-eval: no trigger (split rate within 2p, every merge verified)"
    return "$rc"
}

self_test() {
    local st fail=0 n=0
    st=$(mktemp -d) || exit 2
    trap 'rm -rf -- "${st:?}"' RETURN
    cp "$DECL" "$st/good.json"
    edit() { jq "$1" "$st/good.json" > "$st/$2.json"; }
    edit '.B = 7' b7;  edit '.B = 9' b9;  edit '.B_cap = 10 | .B = 10' cap10
    edit '.K = 3' k3;  edit '.runner_sets = ["intel","yoga","gx10"]' sets3
    edit '.rollback.to_B = 2' rb2;  edit '.rollback.split_rate_factor_of_p = 3' rbf3
    edit 'del(.rollback.on_unverified_merge)' rbnone;  edit 'del(.rollback)' rbgone
    edit '.p.failures = 20 | .p.runs = 50' p40;  edit '.p.failures = 20 | .p.runs = 50 | .B = 3' p40b3
    edit '.state = "rolled_back"' rolled
    live() { # <merge> <build> <strategy> <name>
        jq -n --argjson m "$1" --argjson b "$2" --arg g "$3" \
          '{rules: [{type: "merge_queue", parameters: {max_entries_to_merge: $m, max_entries_to_build: $b,
             grouping_strategy: $g, merge_method: "SQUASH", min_entries_to_merge: 1}}]}' > "$st/live-$4.json"
    }
    live 8 32 ALLGREEN ok;  live 7 32 ALLGREEN b7;  live 8 24 ALLGREEN k3;  live 1 3 ALLGREEN old
    live 8 32 HEADGREEN hg;  live 1 4 ALLGREEN rb;  live 3 12 ALLGREEN p40
    echo '{"rules":[{"type":"required_status_checks"}]}' > "$st/live-noq.json"
    runs() { # <name> <spec: oldest-first groups like F11,S39> <quorum: all|last-fail|last-missing>
        python3 - "$st/runs-$1.json" "$2" "$3" <<'PY2'
import json, sys
out, spec, q = sys.argv[1:]
ci = [("failure" if g[0] == "F" else "success") for g in spec.split(",") for _ in range(int(g[1:]))]
rows = []
for i, c in enumerate(ci):
    t = f"2026-09-25T{i // 60:02d}:{i % 60:02d}:00Z"
    rows.append({"name": "CI", "head_sha": f"{i:040x}", "conclusion": c, "created_at": t})
    last = i == len(ci) - 1
    if c == "success" and not (last and q == "last-missing"):
        rows.append({"name": "pr-review-quorum", "head_sha": f"{i:040x}",
                     "conclusion": "failure" if (last and q == "last-fail") else "success", "created_at": t})
json.dump(rows, open(out, "w"))
PY2
    }
    runs calm  F5,S45  all            # 5/50  = 0.10 <= 2p = 0.208
    runs edge  F10,S40 all            # 10/50 = 0.20 <= 0.208, the last green count
    runs split F11,S39 all            # 11/50 = 0.22 >  0.208
    runs old   F11,S50 all            # the 11 splits fell out of the 50-run window
    runs unver S50     last-fail      # the newest merge's quorum FAILED
    runs noq   S50     last-missing   # the newest merge has no quorum run at all
    row() { # <name> <want> <mode> <decl> <file>
        local got
        n=$((n + 1))
        MQ_DECLARED="$st/$4.json" bash "$0" "$3" "$([ "$3" = check ] && echo --live || echo --runs)" "$st/$5.json" >/dev/null 2>&1
        got=$?
        if [ "$got" = "$2" ]; then printf 'ok    %-26s rc=%s\n' "$1" "$got"
        else printf 'FAIL  %-26s rc=%s (wanted %s)\n' "$1" "$got" "$2"; fail=$((fail + 1)); fi
    }
    row declared-live-match       0  check good live-ok
    row B-edited-to-7             10 check b7 live-ok
    row B-edited-to-9             10 check b9 live-ok
    row B-cap-raised-to-10        10 check cap10 live-ok
    row B-follows-p-at-40pct      0  check p40b3 live-p40
    row p-40pct-keeps-B8          10 check p40 live-ok
    row K-edited-to-3             10 check k3 live-ok
    row runner-set-dropped        10 check sets3 live-ok
    row rollback-to-B2            10 check rb2 live-ok
    row rollback-factor-3p        10 check rbf3 live-ok
    row rollback-unverified-gone  10 check rbnone live-ok
    row rollback-rule-deleted     10 check rbgone live-ok
    row live-B-7                  10 check good live-b7
    row live-build-not-BxK        10 check good live-k3
    row live-still-pre-ruling     10 check good live-old
    row live-headgreen            10 check good live-hg
    row live-no-merge-queue       10 check good live-noq
    row live-rolled-back-undeclared 10 check good live-rb
    row live-rolled-back-declared 0  check rolled live-rb
    row rolled-declared-live-B8   10 check rolled live-ok
    row splits-within-2p          0  rollback-eval good runs-calm
    row splits-at-2p-edge         0  rollback-eval good runs-edge
    row splits-above-2p           10 rollback-eval good runs-split
    row unverified-merge          10 rollback-eval good runs-unver
    row merge-with-no-quorum-run  10 rollback-eval good runs-noq
    row splits-outside-window     0  rollback-eval good runs-old
    if [ "$fail" != 0 ]; then echo "--- $fail of $n rows wrong ---" >&2; return 1; fi
    echo "--- $n/$n rows, both polarities ---"
}

case "${1:-}" in
    check)         [ "${2:-}" = --live ] && { [ -n "${3:-}" ] || exit 2; }; mode_check "${3:-}"; exit $? ;;
    rollback-eval) [ "${2:-}" = --runs ] && { [ -n "${3:-}" ] || exit 2; }; mode_rollback "${3:-}"; exit $? ;;
    --self-test)   self_test; exit $? ;;
    *) echo "usage: check_merge_queue_ruleset.sh check [--live FILE] | rollback-eval [--runs FILE] | --self-test" >&2; exit 2 ;;
esac
