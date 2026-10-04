#!/usr/bin/env bash
# check_producers_chain.sh — every nightly producer chains from "Nightly pick" and measures the night's C (#4686).
#
# A producer that starts from its own cron runs whenever GitHub gets round to creating the run (2–6.5h late on
# 09-26..10-03), and measures main's head at that hour: two producers of one night can measure two commits. T44 chains
# them instead: each one starts on the pick's workflow_run and, after every checkout, switches the tree to the commit
# the pick pinned at nightly/<night> (scripts/release/nightly_c_checkout.sh). This guard reads the workflow text and
# is RED when a producer:
#   P1  still has a `schedule:` trigger (a live line in its on: block);
#   P2  lacks the chain: workflow_run with workflows: [<the name nightly-pick.yml declares>], types: [completed],
#       branches: [main];
#   P3  checks this repo out (actions/checkout without `repository:`) and the next step, after an optional
#       "Preflight…" step, is not the C step: run `bash scripts/release/nightly_c_checkout.sh --at "$NIGHTLY_PICK_AT"`,
#       env NIGHTLY_PICK_AT from github.event.workflow_run.run_started_at, and if exactly
#       `github.event_name == 'workflow_run'`, or `github.event_name == 'workflow_run' && (<the checkout's if>)`;
#       the C step carries nothing else (no continue-on-error, no other env), a checkout's if must be a bare
#       expression the C step can copy, a listed producer has at least one checkout of this repo, and no job of it
#       calls a reusable workflow (`uses:` at job level), whose checkout this file cannot see;
#   P4  names github.sha, GITHUB_SHA, github.workflow_sha or GITHUB_WORKFLOW_SHA on a live line that names neither
#       NIGHTLY_C nor workflow_run.head_sha outside a trailing comment;
#   P5  is listed below and missing, or chains from the pick and is not listed (the list is complete both ways).
# Limits: it reads the workflow text, not the scripts a step calls; an origin/main a step fetches as a declared
# comparand (guards-nightly's SATD ceiling) is not a measurement and is not judged here.
#
# Usage:
#   check_producers_chain.sh [--root DIR]     judge the tree (default: the repo this file is in)
#   check_producers_chain.sh --self-test | --mutants
# Exit: 0 every producer chained; 1 RED; 3 caller error.
set -uo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_PATH="$HERE/$(basename -- "${BASH_SOURCE[0]}")"

# The nightly producers: the workflows whose results are read as "the night's" measurement.
PRODUCERS="beat-speed-nightly book book-contracts conleche-nightly coverage-nightly cuda-nightly examples-nightly
guards-nightly install-script mutants-nightly nightly nightly-bench qwen-story-daily silicon-nightly toolchain-ceiling"

caller_error() { printf 'check_producers_chain: caller error: %s\n' "$1" >&2; exit 3; }

# judge_file FILE PICK LISTED -> FAIL lines on stdout; "CHAINED" when the file chains from PICK
judge_file() {
    awk -v pick="$2" -v listed="$3" -v f="$(basename -- "$1")" '
        function ind(s) { match(s, /^ */); return RLENGTH }
        function live(s) { return s !~ /^[[:space:]]*(#|$)/ }
        function bare(s) { sub(/^[^:]*:[[:space:]]*/, "", s); sub(/[[:space:]]+$/, "", s); return s }
        function list1(s) { s = bare(s); if (s !~ /^\[.*\]$/) return "\001"; s = substr(s, 2, length(s) - 2)
            gsub(/^[[:space:]]+|[[:space:]]+$/, "", s); if (s ~ /^".*"$/ || s ~ /^\047.*\047$/) s = substr(s, 2, length(s) - 2)
            return s }
        function fail(m) { printf "FAIL  %s: %s\n", f, m; bad = 1 }
        function key(st, k,   a, m, i, p) {
            m = split(st, a, "\n"); p = sprintf("%" (ind(a[1]) + 2) "s", "")
            if (a[1] ~ ("^ *- " k ":")) return bare(a[1])
            for (i = 2; i <= m; i++) if (index(a[i], p k ":") == 1) return bare(a[i])
            return "\001"
        }
        function unq(s) { if (s ~ /^".*"$/ || s ~ /^\047.*\047$/) s = substr(s, 2, length(s) - 2); return s }
        function stray(st,   a, m, i, p, k, s) { # the first part of the C step that is not name/if/run or the env line
            m = split(st, a, "\n"); p = ind(a[1]) + 2
            for (i = 1; i <= m; i++) {
                s = a[i]; if (!live(s)) continue
                if (i == 1) { k = s; sub(/^ *- /, "", k) }
                else if (ind(s) == p) k = substr(s, p + 1)
                else { gsub(/^[[:space:]]+|[[:space:]]+$/, "", s)
                       if (s == "NIGHTLY_PICK_AT: ${{ github.event.workflow_run.run_started_at }}") continue; return s }
                sub(/:.*/, "", k); if (k !~ /^(name|if|env|run)$/) return k
            }
            return ""
        }
        function hasline(st, want,   a, m, i, s) {
            m = split(st, a, "\n")
            for (i = 1; i <= m; i++) { s = a[i]; gsub(/^[[:space:]]+|[[:space:]]+$/, "", s); if (s == want) return 1 }
            return 0
        }
        function flush(   k, j, cif, want, nm) {
            if (!listed) { n = 0; ii = -1; return }   # only a listed producer owes the C step
            for (k = 1; k <= n; k++) {
                if (unq(key(S[k], "uses")) !~ /^actions\/checkout@/ || S[k] ~ /\n[[:space:]]*repository:/) continue
                checkouts++
                j = k + 1; nm = key(S[j], "name")
                if (j <= n && nm ~ /^Preflight/) j++
                cif = unq(key(S[k], "if"))
                if (cif ~ /\$\{\{|^[>|]/) {
                    fail("P3 checkout in job step " k " has an if: the C step cannot copy (${{ }} or a block): " cif); continue }
                want = (cif == "\001") ? "github.event_name == \047workflow_run\047" \
                                       : "github.event_name == \047workflow_run\047 && (" cif ")"
                if (j > n) { fail("P3 checkout in job step " k " is not followed by the C step (end of steps)"); continue }
                if (key(S[j], "run") != "bash scripts/release/nightly_c_checkout.sh --at \"$NIGHTLY_PICK_AT\"") {
                    fail("P3 checkout in job step " k " is not followed by the C step (step " j " runs something else)"); continue }
                if (!hasline(S[j], "NIGHTLY_PICK_AT: ${{ github.event.workflow_run.run_started_at }}"))
                    fail("P3 the C step after job step " k " does not take NIGHTLY_PICK_AT from workflow_run.run_started_at")
                if (unq(key(S[j], "if")) != want)
                    fail("P3 the C step after job step " k " has if: " key(S[j], "if") " -- want: " want)
                if ((sx = stray(S[j])) != "")
                    fail("P3 the C step after job step " k " carries " sx " -- only name, if, env NIGHTLY_PICK_AT, run")
            }
            n = 0; ii = -1
        }
        { L = $0 }
        # on: block
        /^on:[[:space:]]*$/ { inon = 1; next }
        inon && live(L) && ind(L) == 0 { inon = 0; wr = 0 }
        inon && live(L) {
            if (L ~ /^  schedule:/) sched = 1
            if (L ~ /^  workflow_run:/) { wr = 1; next }
            if (ind(L) <= 2) wr = 0
            if (wr && L ~ /^    workflows:/ && list1(L) == pick) cw = 1
            if (wr && L ~ /^    types:/ && list1(L) == "completed") ct = 1
            if (wr && L ~ /^    branches:/ && list1(L) == "main") cb = 1
        }
        # sha rule, every live line
        live(L) { c = L; sub(/[[:space:]]#.*/, "", c)   # an exemption in a trailing comment exempts nothing
            if (c ~ /github\.(workflow_)?sha|GITHUB_(WORKFLOW_)?SHA/ && c !~ /NIGHTLY_C|workflow_run\.head_sha/) shaline[++ns] = NR }
        # a job that calls a reusable workflow: its checkout is not in this file
        listed && live(L) && L ~ /^    uses:/ { fail("P3 line " NR " is a job-level uses: -- its checkout cannot be judged here") }
        # steps
        insteps && live(L) && ind(L) <= si { flush(); insteps = 0 }
        insteps && live(L) {
            if (L ~ /^ *- / && (ii < 0 || ind(L) == ii)) { ii = ind(L); S[++n] = L }
            else if (n > 0) S[n] = S[n] "\n" L
            next
        }
        /^ *steps:[[:space:]]*$/ && live(L) { insteps = 1; si = ind(L); n = 0; ii = -1; next }
        END {
            if (insteps) flush()
            chained = (cw && ct && cb)
            if (!listed) { if (cw) printf "FAIL  %s: P5 chains from \"%s\" but is not in the producer list\n", f, pick; exit }
            if (sched) fail("P1 still has a schedule: trigger -- a producer starts on the pick, never on its own cron")
            if (!checkouts) fail("P3 no actions/checkout of this repo found -- nothing switches the tree to C")
            if (!chained) fail("P2 no workflow_run on [\"" pick "\"], types [completed], branches [main]" \
                               " (workflows=" cw " types=" ct " branches=" cb ")")
            for (i = 1; i <= ns; i++) fail("P4 line " shaline[i] " names github.sha/GITHUB_SHA -- main at trigger, not C")
            if (!bad) printf "ok    %s: chained from \"%s\", %d checkout(s), each followed by the C step\n", f, pick, checkouts
        }' "$1"
}

judge() { # ROOT
    local root="$1" wf pick p rc=0 out listed
    wf="$root/.github/workflows"
    [ -f "$wf/nightly-pick.yml" ] || { printf 'FAIL  nightly-pick.yml is missing: there is no pick to chain from\n'; return 1; }
    pick="$(sed -n -e 's/^name:[[:space:]]*//p' "$wf/nightly-pick.yml" | head -n 1 | sed -e 's/[[:space:]]*$//' -e 's/^"\(.*\)"$/\1/' -e "s/^'\(.*\)'\$/\1/")"
    [ -n "$pick" ] || { printf 'FAIL  nightly-pick.yml declares no name:\n'; return 1; }
    for p in $PRODUCERS; do
        [ -f "$wf/$p.yml" ] || { printf 'FAIL  %s.yml: P5 listed producer is missing\n' "$p"; rc=1; }
    done
    for f in "$wf"/*.yml; do
        [ -f "$f" ] || continue
        p="$(basename -- "$f" .yml)"
        [ "$p" = nightly-pick ] && continue
        listed=0
        case " $(printf '%s' "$PRODUCERS" | tr '\n' ' ') " in *" $p "*) listed=1 ;; esac
        out="$(judge_file "$f" "$pick" "$listed")"
        [ -n "$out" ] && printf '%s\n' "$out"
        printf '%s\n' "$out" | grep -q -e '^FAIL' && rc=1
    done
    if [ "$rc" -eq 0 ]; then printf 'PASS  every nightly producer chains from "%s" and measures its C\n' "$pick"
    else printf 'RED   a nightly producer is not chained to the pick, or does not measure its C\n'; fi
    return "$rc"
}

# ------------------------------------------------------------------------------------ the case table ----
CASES=0; FAILED=0
REAL_ROOT="$(cd -- "$HERE/../.." && pwd)"
fixture() { # DIR: the real producers + nightly-pick.yml
    rm -rf -- "${1:?}"; mkdir -p "$1/.github/workflows" || return 1
    local p
    for p in $PRODUCERS nightly-pick; do cp "$REAL_ROOT/.github/workflows/$p.yml" "$1/.github/workflows/" || return 1; done
}
row() { # NAME WANT-RC MUST -- MUTATION-FN (run inside the fixture's workflows dir)
    local name="$1" want="$2" must="$3" fx out rc
    shift 4
    fx="$TMP_ST/fx"
    fixture "$fx" || { printf 'RED   %-40s fixture failed\n' "$name"; FAILED=$((FAILED + 1)); CASES=$((CASES + 1)); return; }
    (cd "$fx/.github/workflows" && "$@") || { printf 'RED   %-40s mutation did not apply\n' "$name"; FAILED=$((FAILED + 1)); CASES=$((CASES + 1)); return; }
    out="$(judge "$fx" 2>&1)"; rc=$?
    CASES=$((CASES + 1))
    if [ "$rc" != "$want" ]; then printf 'RED   %-40s rc=%s want %s\n%s\n' "$name" "$rc" "$want" "$(printf '%s\n' "$out" | grep -e '^FAIL' | head -n 3)"; FAILED=$((FAILED + 1)); return; fi
    if [ -n "$must" ] && ! printf '%s\n' "$out" | grep -qF -e "$must"; then printf 'RED   %-40s missing: %s\n' "$name" "$must"; FAILED=$((FAILED + 1)); return; fi
    printf 'ok    %s\n' "$name"
}
changed() { # FILE SED-EXPR: apply, and fail when nothing changed (a no-op mutation is an error, never a pass)
    local before; before="$(cat "$1")"; sed -i -e "$2" "$1" || return 1; [ "$before" != "$(cat "$1")" ]
}
CSTEP='bash scripts/release/nightly_c_checkout.sh --at "$NIGHTLY_PICK_AT"'
synth() { # FILE: a minimal chained producer, built by the caller-supplied steps on stdin
    { printf 'name: Synth\non:\n  workflow_run:\n    workflows: ["Nightly pick"]\n    types: [completed]\n    branches: [main]\n  workflow_dispatch:\n'
      printf 'jobs:\n  a:\n    runs-on: x\n    steps:\n'; cat; } > "$1"
}
cstep() { # [IF]: the C step text, with the checkout's if when given
    local cif="github.event_name == 'workflow_run'"; [ -n "${1:-}" ] && cif="$cif && ($1)"
    printf '      - name: Measure the night'"'"'s C, not main'"'"'s head (T44)\n        if: %s\n        env:\n          NIGHTLY_PICK_AT: ${{ github.event.workflow_run.run_started_at }}\n        run: %s\n' "$cif" "$CSTEP"
}
m_none()          { :; }
m_sched_back()    { changed nightly.yml 's/^  workflow_run:$/  schedule:\n    - cron: "0 3 * * *"\n  workflow_run:/'; }
m_sched_comment() { changed nightly.yml 's/^  workflow_run:$/  # schedule:\n  #   - cron: "0 3 * * *"\n  workflow_run:/'; }
m_chain_other()   { changed book.yml 's/workflows: \["Nightly pick"\]/workflows: ["Nightly"]/'; }
m_chain_two()     { changed book.yml 's/workflows: \["Nightly pick"\]/workflows: ["Nightly pick", "CI"]/'; }
m_chain_type()    { changed book.yml '0,/types: \[completed\]/s//types: [requested]/'; }
m_chain_branch()  { changed cuda-nightly.yml '/^    branches: \[main\]$/d'; }
m_chain_branch2() { changed cuda-nightly.yml 's/^    branches: \[main\]$/    branches: [release]/'; }
m_chain_cmt()     { changed cuda-nightly.yml 's/^    workflows:/    # workflows:/'; }
m_pick_renamed()  { changed nightly-pick.yml 's/^name: Nightly pick$/name: Nightly Pick/'; }
m_missing()       { rm -f -- silicon-nightly.yml; }
m_unlisted()      { synth extra.yml <<< '      - run: true'; }
m_no_cstep()      { changed toolchain-ceiling.yml '/- name: Measure the night.s C, not main.s head (T44)/,/nightly_c_checkout.sh/d'; }
m_cstep_or()      { changed install-script.yml "0,/if: github.event_name == 'workflow_run'\$/s//if: github.event_name == 'workflow_run' || true/"; }
m_cstep_ifdrop()  { changed cuda-nightly.yml "s/if: github.event_name == 'workflow_run' && (.*)\$/if: github.event_name == 'workflow_run'/"; }
m_cstep_noat()    { changed nightly-bench.yml '0,/nightly_c_checkout.sh --at "\$NIGHTLY_PICK_AT"/s//nightly_c_checkout.sh/'; }
m_cstep_wrongat() { changed coverage-nightly.yml '0,/workflow_run.run_started_at/s//workflow_run.head_commit.timestamp/'; }
m_step_between()  { changed examples-nightly.yml '0,/- name: Measure the night.s C/s//- run: make bench\n      - name: Measure the night'"'"'s C/'; }
m_repo_exempt()   { synth nightly.yml <<< "$(printf '      - uses: actions/checkout@v7\n'; cstep; printf '      - uses: actions/checkout@v7\n        with:\n          repository: paiml/other\n')"; }
m_preflight_ok()  { synth nightly.yml <<< "$(printf '      - uses: actions/checkout@v7\n      - name: Preflight — tools\n        run: true\n'; cstep)"; }
m_flow_if_ok()    { synth nightly.yml <<< "$(printf '      - uses: actions/checkout@v7\n        if: a || b\n'; cstep 'a || b')"; }
m_flow_if_bare()  { synth nightly.yml <<< "$(printf '      - uses: actions/checkout@v7\n        if: a || b\n'; cstep | sed -e "s/'workflow_run'\$/'workflow_run' \&\& a || b/")"; }
m_last_step()     { synth nightly.yml <<< '      - uses: actions/checkout@v7'; }
m_sha_plain()     { changed qwen-story-daily.yml '0,/^    steps:$/s//    steps:\n      - run: echo "${{ github.sha }}"/'; }
m_sha_env()       { changed beat-speed-nightly.yml '0,/^    steps:$/s//    steps:\n      - run: git diff "$GITHUB_SHA"/'; }
m_sha_comment()   { changed beat-speed-nightly.yml '0,/^    steps:$/s//    steps:\n      # measured at $GITHUB_SHA, before T44/'; }
m_sha_c()         { changed beat-speed-nightly.yml '0,/^    steps:$/s//    steps:\n      - run: echo "${NIGHTLY_C:-$GITHUB_SHA}"/'; }
m_cstep_coe()     { changed install-script.yml '0,/nightly_c_checkout.sh --at "\$NIGHTLY_PICK_AT"$/s//&\n        continue-on-error: true/'; }
m_cstep_envx()    { changed nightly-bench.yml '0,/NIGHTLY_PICK_AT: \${{ github.event.workflow_run.run_started_at }}$/s//&\n          GIT_DIR: \/elsewhere/'; }
m_cstep_qif_ok()  { synth nightly.yml <<< "$(printf '      - uses: actions/checkout@v7\n'; cstep | sed -e "s/if: \(.*\)\$/if: \"\1\"/")"; }
m_quoted_uses()   { synth nightly.yml <<< "$(printf '      - uses: "actions/checkout@v7"\n      - run: make bench\n')"; }
m_no_checkout()   { synth nightly.yml <<< '      - run: make bench'; }
m_job_uses()      { synth nightly.yml <<< "$(printf '      - uses: actions/checkout@v7\n'; cstep; printf '  b:\n    uses: ./.github/workflows/bench.yml\n')"; }
m_checkout_ifexp(){ synth nightly.yml <<< "$(printf '      - uses: actions/checkout@v7\n        if: ${{ inputs.x }}\n'; cstep '${{ inputs.x }}')"; }
m_sha_workflow()  { changed qwen-story-daily.yml '0,/^    steps:$/s//    steps:\n      - run: echo "${{ github.workflow_sha }}"/'; }
m_sha_cmt_exempt(){ changed qwen-story-daily.yml '0,/^    steps:$/s//    steps:\n      - run: git diff "${{ github.sha }}" # NIGHTLY_C/'; }
m_bystander()     { printf 'name: Other\non:\n  schedule:\n    - cron: "0 1 * * *"\njobs:\n  a:\n    runs-on: x\n    steps:\n      - uses: actions/checkout@v7\n      - run: echo "$GITHUB_SHA"\n' > other.yml; }

self_test() {
    TMP_ST="$(mktemp -d "${TMPDIR:-/tmp}/pc-st.XXXXXX")" || caller_error "no temp dir"
    echo "=== nightly producers chain: each rule must turn RED when its link breaks ==="
    row base_real_producers         0 'PASS'                                   -- m_none
    row p1_schedule_back            1 'P1 still has a schedule'                -- m_sched_back
    row p1_schedule_commented       0 'PASS'                                   -- m_sched_comment
    row p2_other_workflow           1 'book.yml: P2'                           -- m_chain_other
    row p2_two_workflows            1 'book.yml: P2'                           -- m_chain_two
    row p2_requested_type           1 'book.yml: P2'                           -- m_chain_type
    row p2_no_branches              1 'cuda-nightly.yml: P2'                   -- m_chain_branch
    row p2_other_branch             1 'cuda-nightly.yml: P2'                   -- m_chain_branch2
    row p2_commented_workflows      1 'cuda-nightly.yml: P2'                   -- m_chain_cmt
    row p2_pick_renamed             1 'P2 no workflow_run on ["Nightly Pick"]' -- m_pick_renamed
    row p5_missing_producer         1 'silicon-nightly.yml: P5 listed'         -- m_missing
    row p5_unlisted_chain           1 'extra.yml: P5 chains'                   -- m_unlisted
    row p3_no_cstep                 1 'toolchain-ceiling.yml: P3'              -- m_no_cstep
    row p3_if_or_true               1 'install-script.yml: P3'                 -- m_cstep_or
    row p3_if_drops_checkout_if     1 'cuda-nightly.yml: P3'                   -- m_cstep_ifdrop
    row p3_no_at                    1 'nightly-bench.yml: P3'                  -- m_cstep_noat
    row p3_at_from_head_commit      1 'does not take NIGHTLY_PICK_AT'         -- m_cstep_wrongat
    row p3_step_between             1 'examples-nightly.yml: P3'               -- m_step_between
    row p3_last_step_checkout       1 'end of steps'                           -- m_last_step
    row p3_other_repo_exempt        0 'PASS'                                   -- m_repo_exempt
    row p3_preflight_between        0 'nightly.yml: chained'                   -- m_preflight_ok
    row p3_checkout_if_with_or      0 'nightly.yml: chained'                   -- m_flow_if_ok
    row p3_unparenthesised_or       1 'nightly.yml: P3'                        -- m_flow_if_bare
    row p4_github_sha               1 'qwen-story-daily.yml: P4'               -- m_sha_plain
    row p4_GITHUB_SHA               1 'beat-speed-nightly.yml: P4'             -- m_sha_env
    row p4_comment_ignored          0 'PASS'                                   -- m_sha_comment
    row p4_nightly_c_fallback       0 'PASS'                                   -- m_sha_c
    row p5_unlisted_scheduled_ok    0 'PASS'                                   -- m_bystander
    row p3_cstep_continue_on_error  1 'carries continue-on-error'              -- m_cstep_coe
    row p3_cstep_extra_env          1 'carries GIT_DIR'                        -- m_cstep_envx
    row p3_cstep_quoted_if          0 'nightly.yml: chained'                   -- m_cstep_qif_ok
    row p3_quoted_checkout_uses     1 'is not followed by the C step'          -- m_quoted_uses
    row p3_no_checkout              1 'no actions/checkout of this repo'       -- m_no_checkout
    row p3_job_level_uses           1 'job-level uses'                         -- m_job_uses
    row p3_checkout_if_expression   1 'cannot copy'                            -- m_checkout_ifexp
    row p4_workflow_sha             1 'qwen-story-daily.yml: P4'               -- m_sha_workflow
    row p4_exemption_in_comment     1 'qwen-story-daily.yml: P4'               -- m_sha_cmt_exempt
    rm -rf -- "${TMP_ST:?}"
    if [ "$FAILED" -eq 0 ]; then printf 'SELF-TEST PASSED: %s rows\n' "$CASES"; return 0; fi
    printf 'SELF-TEST FAILED: %s of %s rows\n' "$FAILED" "$CASES"; return 1
}

MUTANTS='m01_no_schedule_rule	s/if (sched) fail/if (0) fail/
m02_any_branch	s/list1(L) == "main") cb = 1/1) cb = 1/
m03_any_type	s/list1(L) == "completed") ct = 1/1) ct = 1/
m04_any_workflow	s/list1(L) == pick) cw = 1/1) cw = 1/
m05_no_preflight_skip	s/nm ~ \/^Preflight\/) j++/0) j++/
m06_no_repo_exempt	s/ || S\[k\] ~ \/\\n\[\[:space:\]\]\*repository:\/) continue/) continue/
m07_any_if	s/if (unq(key(S\[j\], "if")) != want)/if (0)/
m08_any_at_source	s/if (!hasline(S\[j\], "NIGHTLY_PICK_AT/if (0 \&\& !hasline(S[j], "NIGHTLY_PICK_AT/
m09_no_sha_rule	s|if (c ~ /github|if (0 \&\& c ~ /github|
m10_sha_no_exemption	s| \&\& c !~ /NIGHTLY_C\|workflow_run\\.head_sha/)|)|
m11_no_missing_rule	/P5 listed producer is missing/s/; rc=1; }/; }/
m12_no_unlisted_rule	s/if (cw) printf "FAIL/if (0) printf "FAIL/
m13_comments_live	s/function live(s) { return s !~ \/^\[\[:space:\]\]\*(#|\$)\/ }/function live(s) { return s !~ \/^[[:space:]]*$\/ }/
m14_end_of_steps_ok	s/if (j > n) { fail(/if (0) { fail(/
m15_any_run	s/if (key(S\[j\], "run") != /if (0 \&\& key(S[j], "run") != /
m16_judge_bystanders	/only a listed producer owes the C step/d
m17_cstep_any_key	s/if ((sx = stray(S\[j\])) != "")/if (0)/
m18_quoted_uses_unseen	s/if (unq(key(S\[k\], "uses")) !~/if (key(S[k], "uses") !~/
m19_zero_checkouts_ok	s/if (!checkouts) fail/if (0) fail/
m20_job_uses_ok	s/listed \&\& live(L) \&\& L ~ \/^    uses:\//0 \&\& \/^    uses:\//
m21_checkout_if_any	s/if (cif ~ \/\\\$\\{\\{|^\[>|\]\/) {/if (0) {/
m22_comment_exempts	s/c = L; sub(\/\[\[:space:\]\]#\.\*\/, "", c)/c = L/
m23_no_workflow_sha	s/github\\\.(workflow_)?sha/github\\.sha/
m24_env_any_line	s/if (s == "NIGHTLY_PICK_AT: \${{ github.event.workflow_run.run_started_at }}") continue; return s }/continue }/
m25_cstep_if_quoted_no	s/if (unq(key(S\[j\], "if")) != want)/if (key(S[j], "if") != want)/'
mutants() {
    local tmp name expr killed=0 total=0 errors=0 out cut reds
    cut="$(grep -n -m1 -e "^# -* the case table" "$SCRIPT_PATH" | cut -d: -f1)"
    [ -n "$cut" ] || caller_error "no case-table marker"
    tmp="$(mktemp -d "${TMPDIR:-/tmp}/pc-mu.XXXXXX")" || caller_error "no temp dir"
    mkdir -p "$tmp/scripts/release" "$tmp/.github/workflows"
    cp "$REAL_ROOT"/.github/workflows/*.yml "$tmp/.github/workflows/" || caller_error "cannot copy the workflows"
    while IFS="$(printf '\t')" read -r name expr; do
        [ -n "$name" ] || continue
        total=$((total + 1))
        if ! head -n "$((cut - 1))" "$SCRIPT_PATH" | sed -e "$expr" > "$tmp/top"; then
            printf 'ERROR %-28s sed rejected the patch (an error, never a kill)\n' "$name"; errors=$((errors + 1)); continue
        fi
        { cat "$tmp/top"; tail -n "+$cut" "$SCRIPT_PATH"; } > "$tmp/scripts/release/check_producers_chain.sh"
        if cmp -s "$SCRIPT_PATH" "$tmp/scripts/release/check_producers_chain.sh"; then
            printf 'ERROR %-28s the patch did not apply (an error, never a survivor)\n' "$name"; errors=$((errors + 1)); continue
        fi
        if out="$(bash "$tmp/scripts/release/check_producers_chain.sh" --self-test 2>&1)"; then
            printf 'SURVIVED %s\n' "$name"
        else
            reds="$(printf '%s\n' "$out" | grep -c -e '^RED ')"
            if [ "$reds" -eq 0 ]; then
                printf 'ERROR %-28s failed with no RED row (a broken script, not a kill)\n' "$name"; errors=$((errors + 1)); continue
            fi
            killed=$((killed + 1))
            printf 'killed   %-28s %s\n' "$name" "$reds"
        fi
    done <<< "$MUTANTS"
    rm -rf -- "${tmp:?}"
    printf 'MUTANTS killed=%s total=%s errors=%s\n' "$killed" "$total" "$errors"
    [ "$killed" -eq "$total" ] && [ "$errors" -eq 0 ]
}

# ------------------------------------------------------------------------------------ main ----
main() {
    local root="$REAL_ROOT"
    case "${1:-}" in
        --self-test) self_test; return $? ;;
        --mutants) mutants; return $? ;;
        -h | --help) sed -n '2,28p' "$SCRIPT_PATH"; return 0 ;;
        --root) root="${2:-}"; [ -d "$root" ] || caller_error "--root DIR" ;;
        '') ;;
        *) caller_error "unknown argument '$1' (--root DIR|--self-test|--mutants)" ;;
    esac
    judge "$root"
}
main "$@"; exit $?
