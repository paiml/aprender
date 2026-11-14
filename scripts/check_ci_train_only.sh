#!/usr/bin/env bash
# check_ci_train_only.sh -- the heavy tier runs once per merge train; a PR lands only via the train (#7001).
#
# THE RULE (quorum receipt QUORUM-GATE-7001-1113, verdict b; fix round 1 after the non-author FAIL on #7006). In ci.yml:
#   TRAIN = pull_request AND head repo == this repo AND head ref starts with `train/aprender-` AND PR author == the fleet App.
#   1. STRUCTURE, not substrings. The TRAIN expression and each heavy job's `if:` must EQUAL a canonical string held once
#      in this file (whitespace-normalised). The App login is held once here (BOT); every `user.login ==` in the workflow
#      must name it. Each verdict job's job-level `if:` must be exactly `always()` (a verdict that can be skipped is a pass).
#      A reordered, OR-ed, extra-clause or renamed condition is RED, not just a removed one.
#   2. BEHAVIOUR. The first run-step of each verdict job (workspace-test, ci-gate, gate) is executed WHOLE with a stubbed env:
#        draft PR                       -> RED, "draft: not mergeable" (never green, never skipped)
#        non-train ready PR             -> RED, the exact refusal message
#        workflow_dispatch off main     -> RED;  workflow_dispatch on refs/heads/main -> green
#        train behind main / tip unread / pr-light red / a heavy result skipped -> RED
#        train on main's tip, everything green                                  -> green
#        gate on a train with x86-main not success (skipped/missing too) -> RED (fix round 3, finding 1: the train owes the
#          whole x86-main job, where vendored-schemas, guard-cargo-shard and the bin filter run, as merge_group owes `guards`)
#        push off refs/heads/main (v* tag, master) at the head sha of an open PR, or the PR list unreadable -> RED;
#          push to main, and a tag push no open PR heads -> green. The step's own jq filter runs over fixtures: an open PR
#          heading the sha -> RED; a closed one, or one whose BASE is the sha -> green.
#   3. the verdict needs every heavy job (a missing `needs` could let a skipped heavy job pass unseen); nothing in a
#      verdict job is continue-on-error (no third state). gate reads X86R from needs.x86-main.result; x86-main runs the
#      bin-filter step on pull_request (it ran in `guards` on merge_group). The heavy jobs (x86-main, determinism,
#      workspace-test-shard) carry no job-level continue-on-error, and a continue-on-error step in them must have an if:
#      confined to a push to refs/heads/main with no `||` (else x86-main reports success over a failed guard).
#   4. REQUIRED CHECK NAMES (fix round 3, finding 3). Branch protection keys on check-run NAMES, from any job of any
#      workflow. Each required name is produced by exactly its verdict job and by no other job:
#        `ci / gate` <- ci-gate,  `gate` <- gate,  `workspace-test` <- workspace-test   (REQ below).
#      A job's check name is its `name:` or else its id; a reusable-workflow call job (`uses:`) yields "<name> / <inner>",
#      so a call job whose name/id prefixes a required name is a producer. A `${{` in a job name is unknowable: RED.
#      Other workflows in .github/workflows produce none of the three, except OTHER_OK (nightly.yml's `gate`: schedule
#      and workflow_dispatch only; harmless alone, all three names are required, the other two stay ci.yml's).
#
# Usage:  bash scripts/check_ci_train_only.sh [--self-test] [WORKFLOW_FILE]
# Wired:  ci/explicit-test-commands.d/847-ci-train-only.cmd (--self-test) and guard_tree.sh (no args).
set -euo pipefail

BOT='paiml-fleet[bot]'   # the fleet App's bot login (merged_by on train PRs, e.g. paiml/aprender#6855). ONE source.
MSG='not mergeable alone: lands only via train/aprender-* (heavy tier runs once per train, #7001)'
DRAFT_MSG='draft: not mergeable; mark ready, then it lands only via train/aprender-*'
EVENT_PART="github.event_name != 'pull_request' || (github.event.pull_request.head.repo.full_name == github.repository && startsWith(github.event.pull_request.head.ref, 'train/aprender-') && github.event.pull_request.user.login == '$BOT')"
CAN_X86="if: \${{ !cancelled() && needs.mg-reuse.outputs.x86 != '1' && github.event.pull_request.draft != true && ($EVENT_PART) }}"
CAN_DET="if: \${{ !cancelled() && needs.mg-reuse.outputs.det != '1' && ($EVENT_PART) }}"
CAN_SHARD="if: \${{ !cancelled() && needs.mg-reuse-wt.outputs.wt != '1' && github.event.pull_request.draft != true && ($EVENT_PART) }}"
CAN_TRAIN="TRAIN: \${{ (github.event_name == 'pull_request' && github.event.pull_request.head.repo.full_name == github.repository && startsWith(github.event.pull_request.head.ref, 'train/aprender-') && github.event.pull_request.user.login == '$BOT') && '1' || '0' }}"
CAN_REF='REF: ${{ github.ref }}'
CAN_X86R='X86R: ${{ needs.x86-main.result }}'
CAN_BINF_IF="if: \${{ github.event_name == 'pull_request' }}"
REQ=('ci / gate:ci-gate' 'gate:gate' 'workspace-test:workspace-test')   # required check name : the one job that may produce it
OTHER_OK='nightly.yml:gate'   # the only other-workflow producer of a required name (see 4. above)

check_names() { # FILE -> lines "id<TAB>check-name<TAB>uses(0|1)", one per job under `jobs:`
    awk -v q="'" '
        function flush() { if (id != "") print id "\t" (nm != "" ? nm : id) "\t" u; id = ""; nm = ""; u = 0 }
        /^jobs:[[:space:]]*$/ { on = 1; next }
        on && /^[^ #]/ { flush(); on = 0; next }
        on && /^  [A-Za-z0-9_-]+:[[:space:]]*$/ { flush(); id = $0; sub(/^  /, "", id); sub(/:.*$/, "", id); next }
        on && id != "" && /^    name:/ {
            nm = $0; sub(/^    name:[[:space:]]*/, "", nm); sub(/[[:space:]]+#.*$/, "", nm); sub(/[[:space:]]+$/, "", nm)
            c = substr(nm, 1, 1)
            if ((c == "\"" || c == q) && substr(nm, length(nm), 1) == c) nm = substr(nm, 2, length(nm) - 2)
            next }
        on && id != "" && /^    uses:/ { u = 1 }
        END { flush() }' "$1"
}

names_check() { # CI_FILE OTHER_DIR -> one FAIL line per required name not produced by exactly its verdict job; rc 1 when any
    local f="$1" od="$2" bad=0 r nm want ids o base id cn u
    local tab=$'\t'
    while IFS="$tab" read -r id cn u; do
        case "$cn" in *'${{'*) echo "FAIL  job $id: check name '$cn' is an expression (unknowable): RED"; bad=1 ;; esac
    done < <(check_names "$f")
    for r in "${REQ[@]}"; do
        nm="${r%:*}" want="${r##*:}"
        ids="$(check_names "$f" | awk -F '\t' -v n="$nm" '$2 == n {print $1}' | sort | tr '\n' ' ' | sed 's/ $//')"
        [ "$ids" = "$want" ] || { echo "FAIL  required check '$nm' is produced by job(s) [${ids:-none}] in $(basename "$f"), want exactly [$want]"; bad=1; }
        ids="$(check_names "$f" | awk -F '\t' -v n="$nm" '$3 == 1 && index(n, $2 " / ") == 1 {print $1}' | tr '\n' ' ' | sed 's/ $//')"
        [ -z "$ids" ] || { echo "FAIL  required check '$nm' can come from reusable-workflow call job(s) [$ids]"; bad=1; }
        for o in "$od"/*.yml "$od"/*.yaml; do
            [ -f "$o" ] || continue
            base="$(basename "$o")"
            [ "$base" != ci.yml ] || continue   # the file under test stands in for ci.yml (a self-test mutant lives elsewhere)
            while IFS="$tab" read -r id cn u; do
                if [ "$cn" = "$nm" ] || { [ "$u" = 1 ] && [ "${nm#"$cn / "}" != "$nm" ]; }; then
                    [ "$base:$id" = "$OTHER_OK" ] && [ "$nm" = gate ] && continue
                    echo "FAIL  required check '$nm' is also produced by $base job $id"; bad=1
                fi
            done < <(check_names "$o")
        done
    done
    return "$bad"
}

# has PATTERN -> rc 0 when stdin has a matching line; reads ALL of stdin (grep -q + pipefail reads a present line as absent).
has() { /usr/bin/grep "$@" > /dev/null; }
norm() { tr -s '[:space:]' ' ' | sed -e 's/^ //' -e 's/ $//'; }

job_block() { # FILE JOB -> the lines of `  JOB:` up to the next two-space-indented key
    awk -v j="$2" '
        $0 ~ "^  " j ":[[:space:]]*$" { on = 1; print; next }
        on && /^  [A-Za-z0-9_-]+:/ { exit }
        on { print }' "$1"
}

needs_of() { # BLOCK -> the words of the job's `needs: [..]` line
    printf '%s\n' "$1" | sed -n 's/^    needs: *\[\(.*\)\].*/\1/p' | tr ',' ' '
}

first_step() { # BLOCK -> the first run script of the job, de-indented (the whole step, not a fragment)
    printf '%s\n' "$1" | awk '/^        run: \|$/ {r = 1; next} r && /^      - / {exit} r {sub(/^          /, ""); print}'
}

line_eq() { # BLOCK INDENT PREFIX CANONICAL -> rc 0 when exactly one line at exactly INDENT spaces starts with PREFIX and equals CANONICAL
    local n pat="^ {$2}$3"
    n="$(printf '%s\n' "$1" | /usr/bin/grep -c -E "$pat" || true)"
    [ "$n" = 1 ] || return 1
    [ "$(printf '%s\n' "$1" | /usr/bin/grep -E "$pat" | norm)" = "$4" ]
}

# runstep TD SCRIPT BEHIND ENV... -> executes the whole step script with a stubbed gh (prints BEHIND, or fails when BEHIND=fail).
runrc=0; runout=""
runstep() {
    local td="$1" scr="$2" behind="$3"; shift 3
    # With PULLS_JSON set, the commits/<sha>/pulls read applies the step's OWN --jq filter to that fixture (round 3 r2:
    # a stub that ignored --jq let a filter on .base.sha or state "closed" pass every row).
    { printf '#!/bin/sh\nB=%s\n' "'$behind'"
      cat <<'STUB'
[ "$B" = fail ] && exit 1
case "$*" in *"/pulls"*) if [ -n "${PULLS_JSON:-}" ]; then
    while [ $# -gt 0 ]; do if [ "$1" = --jq ]; then printf '%s' "$PULLS_JSON" | jq -r "$2"; exit $?; fi; shift; done
    echo "stub: /pulls read without --jq" >&2; exit 1; fi ;; esac
echo "$B"
STUB
    } > "$td/bin/gh"; chmod +x "$td/bin/gh"
    runrc=0
    runout="$(env -i PATH="$td/bin:/usr/bin:/bin:/usr/local/bin" REPO=x/y BASE_REF=main HEAD_SHA=abc GITHUB_SHA=abc "$@" bash -c "$scr" 2>&1)" || runrc=$?
}

# the per-job env for a fully green heavy tier (so only the tested condition can turn the step red)
green_env() { # JOB -> env assignments, one per line
    case "$1" in
        workspace-test) printf '%s\n' SHARD_RESULT=success WT_REUSE= WT_CITE= ;;
        ci-gate) printf '%s\n' 'RESULTS={"sov.gate":{"result":"success"}}' MG=skipped MG_X86= MG_DET= MG_CITE= ADV=skipped ;;
        gate) printf '%s\n' 'X86={"guard-tree":{"result":"success"},"guard-cargo":{"result":"success"},"sov.gate":{"result":"success"}}' \
                'DET={"determinism-compare":{"result":"success"}}' GRD=null GRDR=success X86R=success WT=success MST=success \
                MG=skipped MG_X86= MG_DET= MG_CITE= ADV=skipped ;;
    esac
}

bot_pin() { # SCRIPT -> rc 0 when its TRAIN_BOT pin equals BOT; prints one FAIL line otherwise
    local got
    got="$(sed -n "s/^TRAIN_BOT='\\([^']*\\)'.*/\\1/p" "$1" 2>/dev/null)"
    [ "$got" = "$BOT" ] && return 0
    echo "FAIL  $(basename "$1"): TRAIN_BOT is not the App login $BOT (got: ${got:-none})"; return 1
}

check() { # FILE -> one FAIL line per violated condition; rc 1 when any
    local f="$1" bad=0 j blk n w scr td c1 want
    local -a cfg base
    td="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/train-only.XXXXXX")"
    mkdir -p "$td/bin"
    # --- 1. structure: canonical strings ---
    for w in "x86-main:$CAN_X86" "determinism:$CAN_DET" "workspace-test-shard:$CAN_SHARD"; do
        j="${w%%:*}"; want="${w#*:}"
        blk="$(job_block "$f" "$j")"
        if [ -z "$blk" ]; then echo "FAIL  heavy job $j not found"; bad=1; continue; fi
        if ! line_eq "$blk" 4 'if:' "$want"; then echo "FAIL  heavy job $j: its if: is not the canonical TRAIN-gated condition"; bad=1; fi
    done
    n="$(/usr/bin/grep -c 'user\.login ==' "$f" || true)"
    c1="$(/usr/bin/grep -c "user\\.login == '$(printf '%s' "$BOT" | sed 's/[][]/\\&/g')'" "$f" || true)"
    if [ "$n" != "$c1" ] || [ "$n" != 6 ]; then echo "FAIL  user.login == appears $n times, $c1 name the App $BOT (want 6 and 6: 3 heavy if + 3 verdict TRAIN)"; bad=1; fi
    # --- 2/3. the verdict jobs ---
    for j in workspace-test ci-gate gate; do
        blk="$(job_block "$f" "$j")"
        if [ -z "$blk" ]; then echo "FAIL  verdict job $j not found"; bad=1; continue; fi
        if [ "$(printf '%s\n' "$blk" | /usr/bin/grep -E '^    if:' | norm)" != 'if: always()' ]; then echo "FAIL  verdict job $j: job-level if: is not exactly always()"; bad=1; fi
        if ! line_eq "$blk" 6 'TRAIN:' "$CAN_TRAIN"; then echo "FAIL  verdict job $j: TRAIN is not the canonical expression"; bad=1; fi
        if ! line_eq "$blk" 6 'REF:' "$CAN_REF"; then echo "FAIL  verdict job $j: REF is not github.ref"; bad=1; fi
        if printf '%s\n' "$blk" | has 'continue-on-error'; then echo "FAIL  verdict job $j uses continue-on-error (no third state)"; bad=1; fi
        # the first step must carry no step-level if: (a skipped verdict step reads as success)
        if printf '%s\n' "$blk" | awk '/^    steps:/{s=1;next} s&&/^        run: \|$/{exit} s&&/^        if:/{f=1} END{exit !f}'; then
            echo "FAIL  verdict job $j: the verdict step has a step-level if:"; bad=1; fi
        scr="$(first_step "$blk")"
        if [ -z "$scr" ]; then echo "FAIL  verdict job $j has no run script"; bad=1; continue; fi
        mapfile -t cfg < <(green_env "$j")
        base=(DRAFT= EVT=pull_request PRL=success TRAIN=1 REF=refs/pull/1/merge "${cfg[@]}")
        row() { # NAME WANT(0|red) MESSAGE BEHIND OVERRIDES... -> run the whole step
            local nm="$1" wantrc="$2" m="$3" bh="$4"; shift 4
            runstep "$td" "$scr" "$bh" "${base[@]}" "$@"
            if [ "$wantrc" = 0 ] && [ "$runrc" != 0 ]; then echo "FAIL  $j: $nm should be green, rc=$runrc"; bad=1
            elif [ "$wantrc" = red ] && [ "$runrc" = 0 ]; then echo "FAIL  $j: $nm passed (must be RED)"; bad=1
            elif [ -n "$m" ] && ! printf '%s\n' "$runout" | has -F -- "$m"; then echo "FAIL  $j: $nm is red without the message: $m"; bad=1; fi
        }
        row 'draft PR (even a train, pr-light green)' red "$DRAFT_MSG" 0 DRAFT=true
        row 'draft PR from a non-train' red "$DRAFT_MSG" 0 DRAFT=true TRAIN=0
        row 'non-train ready PR' red "$MSG" 0 TRAIN=0
        row 'ready PR, TRAIN unset' red "$MSG" 0 TRAIN=
        row 'workflow_dispatch on a branch' red 'workflow_dispatch on' 0 EVT=workflow_dispatch TRAIN=0 REF=refs/heads/some-branch
        row 'workflow_dispatch with an empty ref' red 'workflow_dispatch on' 0 EVT=workflow_dispatch TRAIN=0 REF=
        row 'workflow_dispatch on main, all green' 0 '' 0 EVT=workflow_dispatch TRAIN=0 REF=refs/heads/main PRL=skipped
        row 'train behind main' red 'not on main' 3
        row 'train tip unreadable' red 'not on main' fail
        row 'train, pr-light red' red 'pr-light' 0 PRL=failure
        row 'train on main tip, all green' 0 '' 0
        # push off main (finding 1): BEHIND doubles as the stubbed open-PR count (the PR block is not entered on push)
        row 'push to main, all green' 0 '' 1 EVT=push TRAIN=0 REF=refs/heads/main PRL=skipped
        row 'push of a v* tag no open PR heads' 0 '' 0 EVT=push TRAIN=0 REF=refs/tags/v9.9.9 PRL=skipped
        row 'push of a v* tag at an open PR head' red 'at the head of an open PR' 1 EVT=push TRAIN=0 REF=refs/tags/v9.9.9 PRL=skipped
        row 'push to master at an open PR head' red 'at the head of an open PR' 2 EVT=push TRAIN=0 REF=refs/heads/master PRL=skipped
        row 'push of a v* tag, open PRs unreadable' red 'at the head of an open PR' fail EVT=push TRAIN=0 REF=refs/tags/v9.9.9 PRL=skipped
        # the step's own jq filter over a fixture (GITHUB_SHA=abc): only an OPEN PR whose HEAD is this sha refuses
        row 'push of a v* tag, fixture: open PR heads this sha' red 'at the head of an open PR' 0 EVT=push TRAIN=0 REF=refs/tags/v9.9.9 PRL=skipped \
            'PULLS_JSON=[{"state":"open","head":{"sha":"abc"},"base":{"sha":"def"}}]'
        row 'push of a v* tag, fixture: closed PR heads this sha' 0 '' 0 EVT=push TRAIN=0 REF=refs/tags/v9.9.9 PRL=skipped \
            'PULLS_JSON=[{"state":"closed","head":{"sha":"abc"},"base":{"sha":"def"}}]'
        row 'push of a v* tag, fixture: open PR, this sha is its base' 0 '' 0 EVT=push TRAIN=0 REF=refs/tags/v9.9.9 PRL=skipped \
            'PULLS_JSON=[{"state":"open","head":{"sha":"def"},"base":{"sha":"abc"}}]'
        case "$j" in
            workspace-test) row 'train with the shard matrix skipped' red '' 0 SHARD_RESULT=skipped ;;
            ci-gate) row 'train with x86-main skipped (no sov.gate)' red '' 0 'RESULTS=null' ;;
            gate) row 'train with the determinism section missing' red '' 0 'DET=null'
                  row 'train with workspace-test not green' red '' 0 WT=skipped
                  row 'train with x86-main failed, its sections green' red 'x86-main is' 0 X86R=failure
                  row 'train with x86-main skipped, its sections green' red 'x86-main is' 0 X86R=skipped
                  row 'train with x86-main result missing' red 'x86-main is' 0 X86R=
                  row 'merge_group ignores X86R (guards carries it)' 0 '' 0 EVT=merge_group TRAIN=0 PRL=skipped X86R=skipped \
                      'GRD={"guard-tree":{"result":"success"},"guard-cargo":{"result":"success"}}'
                  if ! line_eq "$blk" 10 'X86R:' "$CAN_X86R"; then echo "FAIL  gate: X86R is not needs.x86-main.result"; bad=1; fi ;;
        esac
    done
    # the bin-filter step (merge_group runs it in `guards`) runs in x86-main on pull_request: present, and gated only on the event
    blk="$(job_block "$f" x86-main)"
    w="$(printf '%s\n' "$blk" | awk '/^      - name: Binary Release bin filter/{s=1;print;next} s&&/^      - /{exit} s{print}')"
    if ! printf '%s\n' "$w" | has -x -F -- '          bash scripts/tests/release_bin_filter_test.sh'; then
        echo "FAIL  x86-main has no bin-filter step (release_bin_filter_test.sh): a train would skip a guard merge_group ran"; bad=1
    elif ! line_eq "$w" 8 'if:' "$CAN_BINF_IF"; then
        echo "FAIL  x86-main bin-filter step: its if: is not exactly the pull_request condition"; bad=1
    fi
    # gate reads needs.x86-main.result and workspace-test reads the shard matrix result: a continue-on-error job, or a
    # continue-on-error step that can run on a pull_request, reports success over a failed guard (no third state). Only a
    # step whose if: confines it to a push to refs/heads/main (codecov, attest) may carry it.
    for j in x86-main determinism workspace-test-shard; do
        blk="$(job_block "$f" "$j")"
        if printf '%s\n' "$blk" | /usr/bin/grep -q -E '^    continue-on-error:'; then echo "FAIL  heavy job $j is continue-on-error at job level"; bad=1; fi
        # confined = the step if: has the push-to-main conjuncts and no `||` (an OR-ed clause could re-admit a PR)
        n="$(printf '%s\n' "$blk" | awk '
            function done() { if (c && !m) bad++; c = 0; m = 0 }
            /^      - / { done() }
            /^        continue-on-error:/ { c = 1 }
            /^        if:/ && index($0, "github.event_name == '\''push'\'' && github.ref == '\''refs/heads/main'\''") && !index($0, "||") { m = 1 }
            END { done(); print bad + 0 }')"
        if [ "$n" != 0 ]; then echo "FAIL  heavy job $j: $n continue-on-error step(s) not confined to a push to main"; bad=1; fi
    done
    names_check "$f" "${OTHER_WF_DIR:-$root/.github/workflows}" || bad=1
    rm -rf "${td:?}"
    for w in "gate:x86-main" "gate:determinism" "gate:workspace-test" "workspace-test:workspace-test-shard" "ci-gate:x86-main"; do
        j="${w%%:*}"; n="${w#*:}"
        blk="$(job_block "$f" "$j")"
        if ! needs_of "$blk" | tr ' ' '\n' | has -x -F -- "$n"; then echo "FAIL  job $j does not need heavy job $n"; bad=1; fi
    done
    # the PR-body lints judge a train body only for the same App login (one pin per consumer, all equal to BOT)
    bot_pin "${LINT_SCRIPT:-$root/scripts/check_pr_ont_delta.sh}" || bad=1
    return "$bad"
}

self_test() {
    local src="$1" td fails=0 got name jb
    td="$(mktemp -d "${RUNNER_TEMP:-${TMPDIR:-/tmp}}/train-only-st.XXXXXX")"; trap 'rm -rf "${td:?}"' RETURN
    if got="$(check "$src")"; then echo "ok    green the workflow as committed"; else echo "FAIL  green: the workflow as committed is RED: $got"; fails=1; fi
    # a long job block (past one pipe buffer) must stay GREEN
    awk '{print} /^  ci-gate:[[:space:]]*$/{on=1} on && /BASE_REF: /{for(i=0;i<20000;i++) print "      # pad pad pad pad pad pad"; on=0}' "$src" > "$td/long.yml"
    if got="$(check "$td/long.yml")"; then echo "ok    green a job block longer than one pipe buffer"; else echo "FAIL  green: a long job block is RED: $got"; fails=1; fi
    mutant() { # NAME SED_SCRIPT -> the mutated workflow must be RED
        name="$1"; sed -E "$2" "$src" > "$td/m.yml"
        if cmp -s "$src" "$td/m.yml"; then echo "FAIL  mutant $name changed nothing"; fails=1; return; fi
        if got="$(check "$td/m.yml")"; then echo "FAIL  mutant $name stayed GREEN"; fails=1; else echo "ok    red   mutant $name"; fi
    }
    local LG='github\.event\.pull_request\.user\.login == '
    # --- the author identity: renamed, in one place, in all places ---
    mutant author-renamed-everywhere     "s/paiml-fleet\\[bot\\]/noahgift/g"
    mutant author-renamed-x86-main       "/^  x86-main:\$/,/runs-on:/{s/paiml-fleet\\[bot\\]/noahgift/}"
    mutant author-renamed-determinism    "/^  determinism:\$/,/runs-on:/{s/paiml-fleet\\[bot\\]/noahgift/}"
    mutant author-renamed-shard          "/^  workspace-test-shard:\$/,/runs-on:/{s/paiml-fleet\\[bot\\]/noahgift/}"
    # --- survivors of round 0 (substring checks passed these): operator and extra-clause rewrites ---
    mutant x86-main-conjuncts-ored       '/^  x86-main:$/,/runs-on:/{s/ && startsWith\(github\.event\.pull_request\.head\.ref, .train\/aprender-.\) && / || startsWith(github.event.pull_request.head.ref, '"'"'train\/aprender-'"'"') \&\& /}'
    mutant det-train-or-anything         '/^  determinism:$/,/runs-on:/{s/\)\) \}\}$/)) || true }}/}'
    mutant shard-extra-or-clause         '/^  workspace-test-shard:$/,/runs-on:/{s/\)\) \}\}$/)) || github.event.pull_request.user.login == '"'"'x'"'"' }}/}'
    mutant verdict-train-or-true         '/^  gate:$/,/^    steps:/{s/(TRAIN: \$\{\{ )\(/\1true || (/}'
    mutant ci-gate-train-always-one      "/^  ci-gate:\$/,/^    steps:/{s/&& '1' \|\| '0' \}\}/\&\& '1' || '1' }}/}"
    # --- TRAIN conjuncts removed, each place ---
    mutant x86-main-no-ref-prefix  '/^  x86-main:$/,/runs-on:/{s/ && startsWith\(github\.event\.pull_request\.head\.ref, .train\/aprender-.\)//}'
    mutant x86-main-no-same-repo   '/^  x86-main:$/,/runs-on:/{s/github\.event\.pull_request\.head\.repo\.full_name == github\.repository && //}'
    mutant x86-main-no-author      "/^  x86-main:\$/,/runs-on:/{s/ && ${LG}'paiml-fleet\\[bot\\]'//}"
    mutant x86-main-runs-on-any-pr '/^  x86-main:$/,/runs-on:/{s/github\.event_name != .pull_request. \|\| //}'
    mutant det-no-author           "/^  determinism:\$/,/runs-on:/{s/ && ${LG}'paiml-fleet\\[bot\\]'//}"
    mutant shard-no-same-repo      '/^  workspace-test-shard:$/,/runs-on:/{s/github\.event\.pull_request\.head\.repo\.full_name == github\.repository && //}'
    mutant shard-no-ref-prefix     '/^  workspace-test-shard:$/,/runs-on:/{s/ && startsWith\(github\.event\.pull_request\.head\.ref, .train\/aprender-.\)//}'
    for jb in workspace-test ci-gate gate; do
        mutant "$jb-env-no-author"   "/^  $jb:\$/,/^    steps:/{s/ && ${LG}'paiml-fleet\\[bot\\]'//}"
        mutant "$jb-env-no-prefix"   "/^  $jb:\$/,/^    steps:/{s/ && startsWith\\(github\\.event\\.pull_request\\.head\\.ref, .train\\/aprender-.\\)//}"
        mutant "$jb-env-no-samerepo" "/^  $jb:\$/,/^    steps:/{s/ && github\\.event\\.pull_request\\.head\\.repo\\.full_name == github\\.repository//}"
        mutant "$jb-env-no-event"    "/^  $jb:\$/,/^    steps:/{s/TRAIN: \\$\{\{ \(github\.event_name == .pull_request. && /TRAIN: \${{ (/}"
        mutant "$jb-job-if-skippable" "/^  $jb:\$/,/^    steps:/{s/^    if: always\\(\\)/    if: \${{ github.event_name != 'pull_request' }}/}"
        mutant "$jb-job-if-needs-ok"  "/^  $jb:\$/,/^    steps:/{s/^    if: always\\(\\)/    if: \${{ success() }}/}"
        mutant "$jb-ref-env-gone"    "/^  $jb:\$/,/^    steps:/{/^      REF: /d}"
        mutant "$jb-non-train-passes" "/^  $jb:\$/,/^  [a-z-]+:\$/{s/if \[ \"\\$\{TRAIN:-0\}\" != 1 \]; then/if false; then/}"
        mutant "$jb-message-changed" "/^  $jb:\$/,/^  [a-z-]+:\$/{s/not mergeable alone/mergeable alone/}"
        mutant "$jb-tip-check-gone"  "/^  $jb:\$/,/^  [a-z-]+:\$/{/behind_by=\\$/d}"
        mutant "$jb-unread-tip-passes" "/^  $jb:\$/,/^  [a-z-]+:\$/{s/\|\| behind=unread/|| behind=0/}"
        mutant "$jb-prl-ignored"     "/^  $jb:\$/,/^  [a-z-]+:\$/{s/\[ \"\\$\{PRL:-\}\" = success \] \|\| /true || /}"
        mutant "$jb-exit0-before-block" "/^  $jb:\$/,/^  [a-z-]+:\$/{s/^( *)# --- PR-LIGHT-VERDICT-BEGIN/\1[ \"\\$\{EVT:-\}\" != pull_request ] || exit 0\n&/}"
        mutant "$jb-continue-on-error" "s/^  $jb:\$/  $jb:\n    continue-on-error: true/"
        # draft (B1): green again, or skipped, or the message changed
        mutant "$jb-draft-exit-0"    "/^  $jb:\$/,/^  [a-z-]+:\$/{s/(draft: not mergeable; mark ready[^\"]*\"; )exit 1/\1exit 0/}"
        mutant "$jb-draft-line-gone" "/^  $jb:\$/,/^  [a-z-]+:\$/{/\\[ \"\\$\{DRAFT:-\}\" != true \\] \|\| \\{ echo \"::error::draft/d}"
        mutant "$jb-draft-msg-changed" "/^  $jb:\$/,/^  [a-z-]+:\$/{s/draft: not mergeable;/draft: ok;/}"
        # dispatch (B5): the main-only line gone, or inverted
        mutant "$jb-dispatch-line-gone" "/^  $jb:\$/,/^  [a-z-]+:\$/{/workflow_dispatch on \\$\{REF/d}"
        mutant "$jb-dispatch-any-ref"   "/^  $jb:\$/,/^  [a-z-]+:\$/{s/\|\| \[ \"\\$\{REF:-\}\" = refs\/heads\/main \]/|| true/}"
        mutant "$jb-dispatch-wrong-ref" "/^  $jb:\$/,/^  [a-z-]+:\$/{s/= refs\/heads\/main \]/= refs\/heads\/dev ]/}"
        # push off main at an open PR head (fix round 3, finding 1): the check gone, or an unread PR list passing
        mutant "$jb-push-check-gone"     "/^  $jb:\$/,/^  [a-z-]+:\$/{/at the head of an open PR/d}"
        mutant "$jb-push-unread-passes"  "/^  $jb:\$/,/^  [a-z-]+:\$/{s/\|\| open=unread/|| open=0/}"
        mutant "$jb-push-jq-base-sha"    "/^  $jb:\$/,/^  [a-z-]+:\$/{s/and \\.head\\.sha == /and .base.sha == /}"
        mutant "$jb-push-jq-state-closed" '/^  '"$jb"':$/,/^  [a-z-]+:$/{s/\.state == \\"open\\"/.state == \\"closed\\"/}'
        mutant "$jb-push-tags-exempt"   "/^  $jb:\$/,/^  [a-z-]+:\$/{s/\[ \"\\$\{REF:-\}\" != refs\/heads\/main \]; then/[ \"\${REF:-}\" = refs\/heads\/master ]; then/}"
    done
    # --- fix round 3, finding 1: the train owes the whole x86-main job ---
    mutant gate-x86-requirement-removed   "/^  gate:\$/,/^  [a-z-]+:\$/{/x86-main is '\\$\{X86R/d}"
    mutant gate-x86-requirement-off-train '/^  gate:$/,/^  [a-z-]+:$/{s/\[ "\$EVT" != pull_request \] \|\| \[ "\$\{X86R/[ "$EVT" != dispatch ] || [ "${X86R/}'
    mutant gate-x86-requirement-any-result "/^  gate:\$/,/^  [a-z-]+:\$/{s/\[ \"\\$\{X86R:-\}\" = success \]/[ -n \"\${X86R:-}\" ]/}"
    mutant gate-x86r-reads-pr-light        's/X86R: \$\{\{ needs\.x86-main\.result/X86R: ${{ needs.pr-light.result/'
    mutant gate-x86r-env-gone              '/^          X86R: /d'
    mutant bin-filter-step-gone            '/^  x86-main:$/,/^  [a-z-]+:$/{/release_bin_filter_test\.sh/d}'
    mutant bin-filter-step-if-false        "/^  x86-main:\$/,/^  [a-z-]+:\$/{s/if: \\$\{\{ github\.event_name == 'pull_request' \}\}\$/if: false/}"
    # --- round 3 r2: a continue-on-error heavy job or PR-reachable step reports success over a failed guard ---
    mutant x86-main-job-coe        's/^  x86-main:$/  x86-main:\n    continue-on-error: true/'
    mutant determinism-job-coe     's/^  determinism:$/  determinism:\n    continue-on-error: true/'
    mutant shard-job-coe           's/^  workspace-test-shard:$/  workspace-test-shard:\n    continue-on-error: true/'
    mutant bin-filter-step-coe     's/^(      - name: Binary Release bin filter.*)$/\1\n        continue-on-error: true/'
    mutant codecov-coe-if-ored     "s/(contains\\(steps\\.wait\\.outputs\\.deferred, 'codecov'\\))\$/\\1 || github.event_name == 'pull_request'/"
    # --- fix round 3, finding 3: required check names come from exactly their verdict job ---
    mutant name-ci-gate-moved-to-pr-light  's/^    name: ci \/ gate$/    name: ci-gate/; s/^  pr-light:$/  pr-light:\n    name: ci \/ gate/'
    mutant name-gate-moved-to-pr-light     's/^  pr-light:$/  pr-light:\n    name: gate/; s/^  gate:$/  gate:\n    name: final verdict/'
    mutant name-wt-duplicated-on-pr-light  's/^  pr-light:$/  pr-light:\n    name: workspace-test/'
    mutant name-ci-gate-dup-quoted-comment 's/^  pr-light:$/  pr-light:\n    name: "ci \/ gate"   # second producer/'
    mutant name-gate-dup-on-x86-main       's/^  x86-main:$/  x86-main:\n    name: gate/'
    mutant name-reusable-call-named-ci     's/^  runner-smells:$/  runner-smells:\n    name: ci/'
    mutant name-verdict-renamed            's/^    name: ci \/ gate$/    name: ci \/ gate2/'
    mutant name-gate-job-given-a-name      's/^  gate:$/  gate:\n    name: the gate/'
    mutant name-wt-job-given-a-name        's/^  workspace-test:$/  workspace-test:\n    name: tests/'
    mutant name-is-an-expression           "s/^    name: ci \\/ gate\$/    name: \${{ 'ci \\/ gate' }}/"
    # another workflow producing a required name (nightly.yml's schedule-only `gate` is the one allowed producer)
    mkdir -p "$td/wf"; cp "$root"/.github/workflows/*.yml "$td/wf/"
    if got="$(OTHER_WF_DIR="$td/wf" check "$src")"; then echo "ok    green the workflow set as committed (copied)"; else echo "FAIL  green: the copied workflow set is RED: $got"; fails=1; fi
    for name in workspace-test:workspace-test 'ci / gate:ci-gate-x' gate:gate; do
        rm -rf "$td/wf2"; mkdir -p "$td/wf2"; cp "$td/wf/"*.yml "$td/wf2/"
        printf 'name: planted\non: [pull_request]\njobs:\n  %s:\n    name: %s\n    runs-on: ubuntu-latest\n    steps:\n      - run: "true"\n' \
            "${name##*:}" "${name%:*}" > "$td/wf2/planted.yml"
        if got="$(OTHER_WF_DIR="$td/wf2" check "$src")"; then echo "FAIL  mutant other-workflow-produces '${name%:*}' stayed GREEN"; fails=1
        else echo "ok    red   mutant other-workflow-produces '${name%:*}'"; fi
    done
    # --- the five gate-opening mutants the round-0 non-author review (Opus, f6c869bf3) found surviving ---
    mutant M2-verdict-if-wrapped-skippable "/^  (workspace-test|ci-gate|gate):\$/,/^    steps:/{s/^    if: always\\(\\)/    if: \${{ always() \&\& (github.event_name != 'pull_request' || startsWith(github.event.pull_request.head.ref, 'train\\/aprender-')) }}/}"
    mutant M5-draft-line-always-exit-0     '/draft: not mergeable; mark ready/{s/^( *)\[ "\$\{DRAFT:-\}" != true \] \|\| \{ echo "[^"]*"; exit 1; \}/\1[ "${DRAFT:-}" = x ] || { echo "::error::draft: not mergeable; mark ready, then it lands only via train\/aprender-*"; exit 0; }/}'
    mutant M6-heavy-if-or-any-pr           "/^  (x86-main|determinism|workspace-test-shard):\$/,/runs-on:/{s/\\)\\) \\}\\}\$/)) || github.event_name == 'pull_request' }}/}"
    mutant M6-train-env-or-any-pr          "/^  (workspace-test|ci-gate|gate):\$/,/^    steps:/{s/\\) \&\& '1' \\|\\| '0' \\}\\}/ || github.event_name == 'pull_request') \&\& '1' || '0' }}/}"
    mutant M4-author-is-actor              "s/user\\.login == 'paiml-fleet\\[bot\\]'/user.login == github.actor/g"
    mutant M3-ref-prefix-contains-or       "s/startsWith\\(github\\.event\\.pull_request\\.head\\.ref, 'train\\/aprender-'\\)/(contains(github.event.pull_request.head.ref, 'train\\/aprender-') || startsWith(github.event.pull_request.head.ref, 'train\\/aprender-'))/g"
    # --- needs ---
    mutant gate-drops-x86-main     '/^  gate:$/,/^    if:/{s/needs: \[x86-main, /needs: [/}'
    mutant gate-drops-determinism  '/^  gate:$/,/^    if:/{s/ determinism,//}'
    mutant gate-drops-workspace-test '/^  gate:$/,/^    if:/{s/ workspace-test,//}'
    mutant wt-drops-shard          '/^  workspace-test:$/,/^    if:/{s/needs: \[workspace-test-shard, /needs: [/}'
    mutant ci-gate-drops-x86-main  '/^  ci-gate:$/,/^    if:/{s/needs: \[x86-main, /needs: [/}'
    # --- the lint consumer's bot pin (B3/B4): a different login, or no pin, is RED ---
    sed "s/^TRAIN_BOT='[^']*'/TRAIN_BOT='noahgift'/" "$root/scripts/check_pr_ont_delta.sh" > "$td/lint-wrong.sh"
    sed "/^TRAIN_BOT=/d" "$root/scripts/check_pr_ont_delta.sh" > "$td/lint-none.sh"
    for name in lint-wrong lint-none; do
        if got="$(LINT_SCRIPT="$td/$name.sh" check "$src")"; then echo "FAIL  mutant bot-pin-$name stayed GREEN"; fails=1; else echo "ok    red   mutant bot-pin-$name"; fi
    done
    return "$fails"
}

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1  # script-relative: no repo query (PMAT-6479)
case "${1:-}" in
    --self-test) self_test "${2:-$root/.github/workflows/ci.yml}"; exit $? ;;
    *)
        out="$(check "${1:-$root/.github/workflows/ci.yml}")" && { echo "ok    a pull_request lands only via the train; the heavy tier is train-only"; exit 0; }
        printf '%s\n' "$out"; exit 1 ;;
esac
