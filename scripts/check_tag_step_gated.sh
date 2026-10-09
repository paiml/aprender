#!/usr/bin/env bash
# check_tag_step_gated.sh -- the release train's tag step must not be reachable without its
# gate (PMAT-3459, which first placed the milestone gate inside cut_tag).
# #4688: the milestone cut and the coverage receipt (#4691) left the release path; what this guard
# still runs is cut_tag's release-policy / #3715 readiness gate ahead of `git tag`.
#
# THE DEFECT, measured 2026-09-17. v0.68.1 was tagged at 15:06:29Z by a per-train
# autopilot copy living OUTSIDE the repository, six minutes after #3455 merged the
# milestone gate. `grep -c check_milestone_cut autopilot.sh` was 0: the tag step read
# no milestone at all. The spec required the read; no code path that cuts a tag made it.
#
# WHY THIS IS BEHAVIOURAL AND NOT A grep. "The file mentions the gate" is satisfied by
# a comment. What must hold is that `git tag` is UNREACHABLE unless the gate returned
# 0, so this guard EXTRACTS cut_tag() from the autopilot and RUNS it against stubs,
# once per gate outcome, and asserts on whether a tag was cut:
#     all green -> tag is cut; readiness or policy refused -> no tag
# --self-test then builds MUTANTS (gate requirements deleted, verdicts discarded)
# and requires this guard to go RED on each. It also runs the carry script's own
# case table, which lives in scripts/release/ where guard_tree cannot discover it.
#
#   check_tag_step_gated.sh              judge scripts/release/autopilot.sh
#   check_tag_step_gated.sh --self-test  case table + the gate-removed mutant
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2

# SEC011: never let an empty or '/' value reach `rm -rf`. One validated helper, used
# by every call site in this file, rather than the check repeated at each one.
rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
SUBJECT="$ROOT/scripts/release/autopilot.sh"

# run_cut_tag <autopilot> <strict-rc> [<must-carry-rc> [<carry-rc> [<readiness> [<covjob-rc> [<policy> [<models>]]]]]] -- extract cut_tag(), run
# it with stubs, print a transcript (SAY/DIE/GIT-TAG/GIT-PUSH lines, then the CALL order).
# Returns 2 if the function is missing.
run_cut_tag() {
    local ap=$1 grc=$2 mrc=${3:-0} crc=${4:-0} rdy=${5:-pass} jrc=${6:-0} pol=${7:-none} mdl=${8:-crux} arc=${9:-0} d fn pfn
    d=$(mktemp -d) || return 2
    fn=$(awk '/^cut_tag\(\) \{/,/^\}/' "$ap")
    [ -n "$fn" ] || { rmtree "$d"; return 2; }
    mkdir -p "$d/scripts/hooks" "$d/scripts/release" "$d/ap" "$d/scripts/lib" "$d/contracts"
    pfn=$(awk '/^ap_policy_applies\(\) \{/,/^\}/' "$ap")
    # the standing release policy: the release worktree's own reader (this checkout's copy) and a ladder
    # with no policy (none), one covering 0.0.0 (covers) or one the reader refuses (bad)
    cp -- "$ROOT/scripts/lib/release_policy.sh" "$ROOT"/scripts/lib/release_policy_*.awk "$d/scripts/lib/" || { rmtree "$d"; return 2; }
    {   printf 'ladder:\n'
        case "$pol" in
            covers|bad) printf '  release_policy:\n    name: crux-smoke\n    since: "0.0.0"\n    date: "d"\n    quote: "q"\n'
                printf '    hosts: [lambda, gx10]\n    thinking: ["off"]\n    larger_rows: nightly\n    red_row_needs: ticket\n    ticket_owner: "#1"\n'
                [ "$pol" = bad ] || printf '    release_notes: known_failures\n' ;;
        esac
        printf '  emergency_scopes:\n'
    } > "$d/contracts/model-capability-ladder-v1.yaml"
    # the models lane's log: a CRUX-smoke GO for this commit (crux), for another (stale), or none
    case "$mdl" in
        crux)  printf 'MODELS GO (CRUX smoke) on lambda and gx10 at deadbeef: the judge passed both receipts (apr 0.0.0 (deadbeef))\n' > "$d/ap/models-t1.log" ;;
        stale) printf 'MODELS GO (CRUX smoke) on lambda and gx10 at cafef00d: the judge passed both receipts (apr 0.0.0 (cafef00d))\n' > "$d/ap/models-t1.log" ;;
        ladder) printf 'MODELS GO on lambda and gx10 at deadbeef: the judge passed both receipts (apr 0.0.0 (deadbeef))\n' > "$d/ap/models-t1.log" ;;
        absent) : ;;
    esac
    # #3715 B1: the readiness step's log, as the T-1 `readiness` step leaves it (or does not)
    case "$rdy" in
        pass)   printf 'ok    R8 release-readiness-v1 for 0.0.0: Pass\nok    R8 #3715 ENFORCE PASS version=0.0.0 commit=deadbeef pv=pv_x out_sha256=0\n' > "$d/ap/readiness-t1.log" ;;
        report) printf 'WARN  R8 REPORT-ONLY release-readiness-v1 for 0.0.0: Fail, 3 violation(s)\n' > "$d/ap/readiness-t1.log" ;;
        stale)  printf 'ok    R8 #3715 ENFORCE PASS version=0.0.0 commit=cafef00d pv=pv_x out_sha256=0\n' > "$d/ap/readiness-t1.log" ;;
        absent) : ;;
    esac
    printf '#!/usr/bin/env bash\nif [ "${2:-}" = --must-carry ]; then echo CALL-MUST-CARRY >> %q; exit %s; fi\necho CALL-STRICT >> %q; exit %s\n' \
        "$d/calls" "$mrc" "$d/calls" "$grc" > "$d/scripts/check_milestone_cut.sh"
    printf '#!/usr/bin/env bash\necho CALL-CARRY >> %q\nexit %s\n' "$d/calls" "$crc" > "$d/scripts/release/carry_milestone_items.sh"
    printf '#!/usr/bin/env bash\necho CALL-COVJOB >> %q\nexit %s\n' "$d/calls" "$jrc" > "$d/scripts/release/tag_coverage_gate.sh"
    # #4950 G1: the pre-push tag guard's --arm-release, for exactly the tag being cut
    printf '#!/usr/bin/env bash\n[ "$1 $2" = "--arm-release v0.0.0" ] && [ "$3" -le 300 ] 2>/dev/null && echo CALL-ARM >> %q\nexit %s\n' "$d/calls" "$arc" > "$d/scripts/hooks/pre-push-tags.sh"
    {
        printf 'set -uo pipefail\n'
        printf 'REPO_ROOT=%q\nLOG=%q\nAP=%q\n' "$d" "$d/log" "$d/ap"
        printf 'say() { printf "SAY %%s\\n" "$*"; }\n'
        printf 'die() { printf "DIE %%s\\n" "$*"; exit 1; }\n'
        printf 'git() { printf "GIT-%%s %%s\\n" "$(printf %%s "$1" | tr "a-z" "A-Z")" "$*"; echo "CALL-GIT-$1" >> %q; }\n' "$d/calls"
        printf '%s\n' "$pfn"
        printf '%s\n' "$fn"
        printf 'cut_tag 0.0.0 v0.0.0 deadbeef\n'
    } > "$d/harness.sh"
    (cd "$d" && bash "$d/harness.sh" 2>&1)
    cat "$d/log" 2>/dev/null
    printf 'ORDER %s\n' "$(tr '\n' ' ' 2>/dev/null < "$d/calls")"
    rmtree "$d"
}

# run_casc <autopilot> <fn> <k=v ...> -- #4950: extract cascade_cleanroom_at_tag() and
# cascade_no_secret_green(), run <fn> against stubs, print a transcript. Knobs (defaults = all green):
#   rec=1 tagc=mc|other|none job=success sha=mc|other|both|none            (cleanroom premise)
#   budget=0 lib=1 seven=1 onlist=1 chkfile=1 chk=0                       (no-secret check)
run_casc() {
    local ap=$1 fn=$2 d body rec=1 tagc=mc job=success sha=mc budget=0 lib=1 seven=1 onlist=1 chkfile=1 chk=0 kv
    shift 2; for kv in "$@"; do local "${kv?}"; done
    d=$(mktemp -d) || return 2
    body=$(awk '/^cascade_cleanroom_at_tag\(\) \{/,/^\}/; /^cascade_no_secret_green\(\) \{/,/^\}/' "$ap")
    [ -n "$body" ] || { rmtree "$d"; return 2; }
    mkdir -p "$d/scripts/release" "$d/scripts/lib" "$d/contracts" "$d/ap" "$d/bin"
    [ "$rec" = 1 ] && printf '4242\n' > "$d/ap/cleanroom-run-id"
    case "$tagc" in mc) tc=deadbeef ;; other) tc=cafef00d ;; *) tc='' ;; esac
    M=deadbeefdeadbeefdeadbeefdeadbeefdeadbeef O=cafef00dcafef00dcafef00dcafef00dcafef00d
    case "$sha" in mc) lg="tested-sha: $M" ;; other) lg="tested-sha: $O" ;; both) lg="tested-sha: $M
tested-sha: $O" ;; *) lg='' ;; esac
    printf '%s\n' "$lg" > "$d/runlog"
    {   printf '#!/usr/bin/env bash\n'
        printf 'case "$1 $2" in\n  "run view") case " $* " in *" --log "*) echo "CALL-LOG $3 $4" >> %q; [ "$3 $4" = "--job 77" ] && cat %q ;;\n' "$d/calls" "$d/runlog"
        printf '    *) echo "CALL-JOBS $3" >> %q; echo %q ;; esac ;;\nesac\n' "$d/calls" "${job:+$job 77}"
    } > "$d/bin/gh"; chmod +x "$d/bin/gh"
    printf '#!/usr/bin/env bash\necho CALL-BUDGET >> %q\nexit %s\n' "$d/calls" "$budget" > "$d/scripts/release/release_ready.sh"
    [ "$lib" = 1 ] && printf 'rp_entries() { RP_IDS="RR-T01 RR-P20"; return 0; }\n' > "$d/scripts/lib/release_policy.sh"
    [ "$lib" = 1 ] || printf ': no reader\n' > "$d/scripts/lib/release_policy.sh"
    id=RR-P20; [ "$onlist" = 1 ] || id=RR-P99
    s='seven: no-secret-in-crates, '; [ "$seven" = 1 ] || s=''
    {   printf '  - {id: RR-T01, applies_to: [tag, publish], seven: provenance, checker: "scripts/x.sh"}\n'
        printf '  - {id: %s, applies_to: [publish], %schecker: "scripts/check_no_secret.sh#main", provenance: "x"}\n' "$id" "$s"
    } > "$d/contracts/release-ready-v1.yaml"
    [ "$chkfile" = 1 ] && printf '#!/usr/bin/env bash\necho CALL-NOSECRET >> %q\nexit %s\n' "$d/calls" "$chk" > "$d/scripts/check_no_secret.sh"
    {   printf 'set -uo pipefail\nV=0.0.0 T=v0.0.0 MC=%s REPO=paiml/aprender INFRA=paiml/infra\n' "$M"
        printf 'AP=%q LOG=%q\n' "$d/ap" "$d/log"
        printf 'say() { printf "SAY %%s\\n" "$*"; }\ndie() { printf "DIE %%s\\n" "$*"; exit 1; }\n'
        printf 'git() { [ "$1" = ls-remote ] && echo "CALL-LSREMOTE $3" >> %q && [ -n "%s" ] && printf "%%s\\trefs/tags/x\\n" "%s"; return 0; }\n' \
            "$d/calls" "$tc" "$( [ -n "$tc" ] && { [ "$tc" = deadbeef ] && echo "$M" || echo "$O"; } )"
        printf '%s\n%s\n' "$body" "$fn"
    } > "$d/harness.sh"
    (cd "$d" && PATH="$d/bin:$PATH" bash "$d/harness.sh" 2>&1)
    printf 'ORDER %s\n' "$(tr '\n' ' ' 2>/dev/null < "$d/calls")"
    printf 'REC %s\n' "$(jq -c . "$d/ap/cascade-cleanroom.json" 2>/dev/null)"
    rmtree "$d"
}

# judge_casc <autopilot> -- the #4950 cascade-premise case table. Returns the number of wrong rows.
judge_casc() {
    local ap=$1 out w=0 cb
    # row RC DESC -- RC is the status of the check just run (no eval: each check is plain code above its row)
    row() { if [ "$1" = 0 ]; then printf 'ok    %s\n' "$2"; else printf 'FAIL  %s\n%s\n' "$2" "$out" >&2; w=$((w + 1)); fi; }
    C=cascade_cleanroom_at_tag N=cascade_no_secret_green
    out=$(run_casc "$ap" $C)
    grep -q "^REC {\"cleanroom_run\":\"4242\",\"tag\":\"v0.0.0\",\"sha\":\"deadbeefdeadbeefdeadbeefdeadbeefdeadbeef\"}$" <<< "$out" && ! grep -q "^DIE" <<< "$out"
    row $? "cascade premise: tag at MC, job green, tested exactly MC -> record run id + sha"
    for c in "rec=0|no clean-room run recorded" "tagc=other|on origin is at .cafef00d" "tagc=none|on origin is at .absent" \
             "job=failure|concluded .failure" "job=|concluded .absent" "sha=other|tested .tested-sha: cafef00d" \
             "sha=both|not exactly" "sha=none|tested .., not exactly"; do
        out=$(run_casc "$ap" $C "${c%%|*}")
        grep -qE "^DIE .*${c#*|}" <<< "$out" && grep -q "^REC $" <<< "$out"
        row $? "cascade premise: ${c%%|*} -> stop, nothing recorded"
    done
    out=$(run_casc "$ap" $N)
    grep -q "^ORDER CALL-BUDGET CALL-NOSECRET $" <<< "$out" && grep -q "^SAY CASCADE no-secret-in-crates: on the publish list and green" <<< "$out"
    row $? "no-secret: budget clean, on the publish list, checker green -> continue"
    for c in "budget=1|release_ready.sh --budget is not clean" "lib=0|cannot be read" "seven=0|no publish entry carries" \
             "onlist=0|no publish entry carries" "chkfile=0|does not exist" "chk=1|is red"; do
        out=$(run_casc "$ap" $N "${c%%|*}")
        grep -qE "^DIE .*${c#*|}" <<< "$out"
        row $? "no-secret: ${c%%|*} -> stop"
    done
    db=$(awk '/^if run_step dryrun; then$/,/^fi$/' "$ap")
    out=$db
    awk "/^  bash scripts\\/release\\/check_crate_contents.sh .*\\\\\$/ { s = NR } s && NR == s + 1 && /^    \\|\\| die / { ok = 1 } END { exit !ok }" <<< "$db"
    row $? "dryrun: every rehearsal runs check_crate_contents.sh, and anything but rc 0 stops it"
    cb=$(awk '/^if run_step cascade; then$/,/^fi$/' "$ap")
    out=$cb
    awk "/^  cascade_cleanroom_at_tag\$/ { a = NR } /^  cascade_no_secret_green\$/ { b = NR } /cascade-drain.sh/ && !d { d = NR } END { exit !(a && b && d > a && d > b) }" <<< "$cb"
    row $? "cascade: both premises run before the first cascade-drain"
    out=$(awk '/^if run_step ledger; then$/,/^fi$/' "$ap")
    grep -q "cascade-cleanroom.json\" \] || die" <<< "$out" && grep -q "cascade_cleanroom: \$c\[0\]" <<< "$out"
    row $? "ledger: refuses without the cascade record and folds it into the ledger record"
    return "$w"
}

# judge <autopilot> -- 0 when every gate outcome behaves; 1 otherwise. Prints rows.
judge() {
    local ap=$1 out bad=0
    if ! grep -q '^cut_tag() {' "$ap"; then
        printf 'FAIL  %s has no cut_tag() -- the tag path moved and this guard is judging nothing\n' "$ap" >&2
        return 2
    fi
    out=$(run_cut_tag "$ap" 0) || true
    if grep -q 'GIT-TAG' <<< "$out"; then printf 'ok    gate rc=0 -> the tag is cut\n'
    else printf 'FAIL  gate rc=0 -> NO tag was cut\n%s\n' "$out" >&2; bad=1; fi
    # #4950 G1: the tag, then its one-shot arm for the pre-push guard (for 300 s at most), then the push
    if grep -q '^ORDER CALL-GIT-tag CALL-ARM CALL-GIT-push $' <<< "$out"; then printf 'ok    all clean -> the tag, arm it, push it\n'
    else printf 'FAIL  all clean did not run tag -> arm -> push\n%s\n' "$out" >&2; bad=1; fi

    # #3715 B1: no ENFORCED readiness Pass for exactly this version+commit -> no tag, nothing carried
    for r in absent report stale; do
        out=$(run_cut_tag "$ap" 0 0 0 "$r") || true
        if grep -q 'GIT-TAG' <<< "$out" || grep -q 'CALL-' <<< "$out"; then
            printf 'FAIL  readiness %s -> a tag was cut or the milestone was touched without an enforced #3715 Pass\n%s\n' "$r" "$out" >&2; bad=1
        else printf 'ok    readiness %s -> no tag, nothing carried\n' "$r"; fi
    done
    # the standing release policy (contracts/model-capability-ladder-v1.yaml `ladder.release_policy`):
    # a covered version tags on the models lane's CRUX-smoke GO for exactly this commit, with no readiness run
    out=$(run_cut_tag "$ap" 0 0 0 absent 0 covers crux) || true
    if grep -q 'GIT-TAG' <<< "$out" && grep -q '^SAY POLICY-GATE ' <<< "$out"; then
        printf 'ok    policy covers, CRUX-smoke GO at this commit, readiness not run -> the tag is cut\n'
    else printf 'FAIL  policy covers + CRUX-smoke GO -> NO tag (or no POLICY-GATE line)\n%s\n' "$out" >&2; bad=1; fi
    for m in stale ladder absent; do
        out=$(run_cut_tag "$ap" 0 0 0 pass 0 covers "$m") || true
        if grep -q 'GIT-TAG' <<< "$out" || grep -q 'CALL-' <<< "$out"; then
            printf 'FAIL  policy covers, models log %s -> a tag was cut or the milestone was touched without a CRUX-smoke GO\n%s\n' "$m" "$out" >&2; bad=1
        else printf 'ok    policy covers, models log %s -> no tag, nothing carried (a readiness Pass does not stand in)\n' "$m"; fi
    done
    out=$(run_cut_tag "$ap" 0 0 0 pass 0 bad crux) || true
    if grep -q 'GIT-TAG' <<< "$out" || grep -q 'CALL-' <<< "$out" || ! grep -q '^DIE the standing release policy cannot be judged' <<< "$out"; then
        printf 'FAIL  unreadable policy block -> a tag was cut, the milestone was touched, or the refusal named another cause\n%s\n' "$out" >&2; bad=1
    else printf 'ok    unreadable policy block -> no tag, nothing carried (not measured is not a pass)\n'; fi
    # #4950 G1: arming the tag for the pre-push guard fails -> the tag is never pushed
    out=$(run_cut_tag "$ap" 0 0 0 pass 0 none crux 2) || true
    if grep -q 'GIT-PUSH' <<< "$out" || ! grep -q '^DIE arming v0.0.0 .*pre-push tag guard failed' <<< "$out"; then
        printf 'FAIL  arm rc=2 -> the tag was pushed unarmed, or the refusal named another cause\n%s\n' "$out" >&2; bad=1
    else printf 'ok    arm rc=2 -> tag not pushed\n'; fi
    # #4950 G2, over the WHOLE autopilot: GitHub polls share one hourly budget, so every wait is
    # AP_POLL (at least 300 s) or the one AP_SETTLE after a dispatch; the crates.io index settle is not a poll
    out=$(awk '/^[[:space:]]*#/ { next }
        /^AP_POLL=/ { s = $0; sub(/^AP_POLL=/, "", s); sub(/[^0-9].*$/, "", s); p = s }
        { l = $0
          while (match(l, /(^|[^_[:alnum:]])sleep[[:space:]]+[^;&|[:space:]]+/)) {
            a = substr(l, RSTART, RLENGTH); sub(/^.*sleep[[:space:]]+/, "", a)
            if (a != "\"$AP_POLL\"" && a != "\"$AP_SETTLE\"" && a != "\"${RELEASE_INDEX_SETTLE_S:-180}\"") w = w NR ":sleep " a " "
            l = substr(l, RSTART + RLENGTH) } }
        END { if (p == "" || p + 0 < 300) w = w "AP_POLL=" p; printf "%s", w }' "$ap")
    if [ -n "$out" ]; then printf 'FAIL  a GitHub poll faster than 300 s, or AP_POLL below 300: %s\n' "$out" >&2; bad=1
    else printf 'ok    every autopilot wait is AP_POLL (>= 300 s) or the one dispatch settle\n'; fi
    judge_casc "$ap" || bad=1
    return "$bad"
}

# guard_tree.sh decides a guard "advertises --self-test" by looking for the literal
# substring `self-test` in its OWN `--help` output (guard_tree.sh:47, advertises_self_test).
# Without this handler the self-test below is discovered by nothing and runs nowhere --
# a facility with a self-test and no caller. With it, guard_tree gives this guard TWO
# rows, `[self-test]` and `[run]`, inside the guard-tree job.
case "${1:-}" in -h|--help) sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== the tag-gate guard must still turn RED (mutants) ==="
    d=$(mktemp -d) || exit 2
    trap 'rmtree "${d:-}"' EXIT
    bad=0
    ok()  { printf 'ok    %s\n' "$*"; }
    nok() { printf 'FAIL  %s\n' "$*" >&2; bad=1; }

    # M6 (#3715 B1): the readiness requirement deleted -> a skipped or report-mode readiness step tags.
    sed '/ENFORCE PASS for\|index(\$0, n) == 1/d; /no .#3715 ENFORCE PASS/d' "$SUBJECT" > "$d/m6.sh"
    if cmp -s "$SUBJECT" "$d/m6.sh"; then
        nok "MUTANT 6 could not be built -- the readiness check line did not match; vacuous"
    elif judge "$d/m6.sh" > "$d/m6.out" 2>&1; then
        nok "MUTANT 6 (readiness requirement deleted) PASSED"
    else
        ok "mutant 6: #3715 readiness requirement deleted -> RED"
    fi
    # M9: under the policy, the CRUX-smoke GO requirement discarded -> a stale or absent models GO tags.
    sed 's/|| die "the standing release policy covers \$v but/|| true; : "/' "$SUBJECT" > "$d/m9.sh"
    if cmp -s "$SUBJECT" "$d/m9.sh"; then
        nok "MUTANT 9 could not be built -- the CRUX-smoke GO die line did not match; vacuous"
    elif judge "$d/m9.sh" > "$d/m9.out" 2>&1; then
        nok "MUTANT 9 (CRUX-smoke GO requirement discarded) PASSED"
    else
        ok "mutant 9: CRUX-smoke GO requirement discarded -> RED"
    fi
    # M10: the policy verdict discarded -> an unreadable policy block reads as "no policy".
    sed 's/|| die "the standing release policy cannot be judged/|| true; : "/' "$SUBJECT" > "$d/m10.sh"
    if cmp -s "$SUBJECT" "$d/m10.sh"; then
        nok "MUTANT 10 could not be built -- the policy-judge die line did not match; vacuous"
    elif judge "$d/m10.sh" > "$d/m10.out" 2>&1; then
        nok "MUTANT 10 (policy verdict discarded) PASSED"
    else
        ok "mutant 10: policy verdict discarded -> RED"
    fi
    # M11: the policy branch never taken -> a covered release still demands the readiness run it replaced.
    sed 's/^    if \[ "\$pol" = 1 \]; then$/    if false; then/' "$SUBJECT" > "$d/m11.sh"
    if cmp -s "$SUBJECT" "$d/m11.sh"; then
        nok "MUTANT 11 could not be built -- the policy branch line did not match; vacuous"
    elif judge "$d/m11.sh" > "$d/m11.out" 2>&1; then
        nok "MUTANT 11 (policy branch never taken) PASSED"
    else
        ok "mutant 11: policy branch never taken -> RED"
    fi
    # M12: the arm verdict discarded -> an unarmed tag is pushed and the guard refuses it on release day.
    sed 's/|| die "arming \$t for the pre-push tag guard failed/|| true; : "/' "$SUBJECT" > "$d/m12.sh"
    if cmp -s "$SUBJECT" "$d/m12.sh"; then
        nok "MUTANT 12 could not be built -- the arm die line did not match; vacuous"
    elif judge "$d/m12.sh" > "$d/m12.out" 2>&1; then
        nok "MUTANT 12 (arm verdict discarded) PASSED"
    else
        ok "mutant 12: arm verdict discarded -> RED"
    fi
    # M13: the arm call deleted -> the ORDER row must see it missing.
    grep -v 'pre-push-tags.sh" --arm-release' "$SUBJECT" > "$d/m13.sh"
    if cmp -s "$SUBJECT" "$d/m13.sh"; then
        nok "MUTANT 13 could not be built -- the arm call line did not match; vacuous"
    elif judge "$d/m13.sh" > "$d/m13.out" 2>&1; then
        nok "MUTANT 13 (arm call deleted) PASSED"
    else
        ok "mutant 13: arm call deleted -> RED"
    fi
    # M14/M15: the poll interval lowered, once globally and once at a single wait.
    sed 's/^AP_POLL=300;/AP_POLL=60;/' "$SUBJECT" > "$d/m14.sh"
    awk '!d && /sleep "\$AP_POLL"/ { sub(/sleep "\$AP_POLL"/, "sleep 60"); d = 1 } { print }' "$SUBJECT" > "$d/m15.sh"
    for mu in 14 15; do
        if cmp -s "$SUBJECT" "$d/m$mu.sh"; then
            nok "MUTANT $mu could not be built -- the poll line did not match; vacuous"
        elif judge "$d/m$mu.sh" > "$d/m$mu.out" 2>&1; then
            nok "MUTANT $mu (GitHub poll under 300 s) PASSED"
        else
            ok "mutant $mu: GitHub poll under 300 s -> RED"
        fi
    done
    # M22-M32 (#4950; M31, the coverage push, left with the coverage dispatch): the cascade premises, the ledger fold, the arm lifetime and the rehearsal secrets check, each weakened one way.
    sed 's/\[ "\$tc" = "\$MC" \] || die/true || die/' "$SUBJECT" > "$d/m22.sh"
    sed 's/\[ "\$shas" = "tested-sha: \$MC" \] || die/true || die/' "$SUBJECT" > "$d/m23.sh"
    grep -v '^  cascade_cleanroom_at_tag$' "$SUBJECT" > "$d/m24.sh"
    grep -v '^  cascade_no_secret_green$' "$SUBJECT" > "$d/m25.sh"
    sed 's/bash "\$chk" >> "\$LOG" 2>&1 || die/true || die/' "$SUBJECT" > "$d/m26.sh"
    sed 's| && /seven: no-secret-in-crates\[,}\]/||' "$SUBJECT" > "$d/m27.sh"
    sed 's/\[ -s "\$AP\/cascade-cleanroom.json" \] || die/true || die/' "$SUBJECT" > "$d/m28.sh"
    sed 's/gh run view --job "\$jid"/gh run view "$crun"/' "$SUBJECT" > "$d/m29.sh"
    sed 's/--arm-release "\$t" 120/--arm-release "$t" 100000/' "$SUBJECT" > "$d/m30.sh"
    sed 's/^    || die "check_crate_contents.sh is not green/    || true "check_crate_contents.sh is not green/' "$SUBJECT" > "$d/m32.sh"
    for mu in 22 23 24 25 26 27 28 29 30 32; do
        if cmp -s "$SUBJECT" "$d/m$mu.sh"; then
            nok "MUTANT $mu could not be built -- its cascade line did not match; vacuous"
        elif judge "$d/m$mu.sh" > "$d/m$mu.out" 2>&1; then
            nok "MUTANT $mu (cascade premise or ledger fold weakened) PASSED"
        else
            ok "mutant $mu: a #4950 check weakened -> RED"
        fi
    done
    # the carry script's own case table: it lives in scripts/release/, where guard_tree cannot see it
    if bash "$ROOT/scripts/release/carry_milestone_items.sh" --self-test > "$d/carry.out" 2>&1; then
        ok "carry_milestone_items.sh case table ($(grep -c '^ok ' "$d/carry.out") rows)"
    else
        nok "carry_milestone_items.sh case table FAILED"; cat "$d/carry.out" >&2
    fi
    # M3: cut_tag() removed entirely -> ENV (2), never a pass.
    awk '/^cut_tag\(\) \{/,/^\}/ {next} {print}' "$SUBJECT" > "$d/m3.sh"
    judge "$d/m3.sh" > "$d/m3.out" 2>&1; rc=$?
    [ "$rc" -eq 2 ] && ok "mutant 3: cut_tag() removed -> ENV rc=2, not a pass" \
        || nok "mutant 3: cut_tag() removed gave rc=$rc, wanted 2"

    # and the real subject must be GREEN, or a red subject masquerades as a killed mutant
    if judge "$SUBJECT" > "$d/real.out" 2>&1; then
        ok "the real subject is GREEN, so the REDs above are the mutants'"
    else
        nok "the real subject is itself RED -- the mutations prove nothing"; cat "$d/real.out" >&2
    fi
    [ "$bad" -eq 0 ] && { echo "SELF-TEST PASSED"; exit 0; }
    echo "SELF-TEST FAILED" >&2; exit 1
fi

echo "=== the tag step cannot be reached without its readiness or policy gate (check_tag_step_gated.sh) ==="
judge "$SUBJECT"; rc=$?
[ "$rc" -eq 0 ] && echo "PASS" || echo "FAIL: the tag step is reachable without its gate (rc=$rc)" >&2
exit "$rc"
