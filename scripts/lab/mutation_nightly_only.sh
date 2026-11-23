#!/usr/bin/env bash
# scripts/lab/mutation_nightly_only.sh -- infra#3828. Operator ruling, Noah 2026-11-10 16:50Z, verbatim:
#   "NO build can do mutation outside of nightly; the end. FOREVER"
# Successor of mutants_advisory_scope.sh (#6601), whose "Rust-source mutation is ADVISORY on a PR" row the ruling ends:
# there is no advisory third state, mutation is simply not on a PR, push, merge-queue, train or local-seat path.
#
# THE CHECKS (each RED line names its rule):
#   MN1  SKILL.md §3.D "Rust source" row names mutants-nightly.yml and carries neither `cargo mutants --in-diff`,
#        MUTANT_BUDGET nor "advisory": no process rule demands a mutation score before a PR is opened or merged.
#   MN2  every .github/workflows file that RUNS mutation (`cargo mutants ...` other than --version/--list, or
#        scripts/mutants_diff_gate.sh other than --self-test/--exempt-ref) has an `on:` block whose ONLY trigger
#        is `schedule` (no pull_request, push, merge_group, workflow_call, workflow_dispatch, workflow_run, ...).
#   MN3  ci/sections.yml and ci/vendor/*.yml (the bodies ci.yml runs on pull_request / push / merge_group) run no mutation.
#   MN4  no Makefile recipe runs mutation (a local seat never does; `cargo mutants --list` runs nothing and is allowed).
#   MN5  at least one schedule-only workflow runs mutation: the nightly still reports the score (not vacuous).
#
# TIER. LAB (C324): wired into guards-nightly.yml, never a PR, merge-queue or release job. A red opens or updates one
# ticket with one owner and stays silent. Promotion to a GATE needs the operator's yes (C324 rule 7); the fleet-wide
# zero-tolerance gate for the same rule is runner-smells RS-12 in paiml/.github.
#
# Usage: scripts/lab/mutation_nightly_only.sh [repo-root] | --self-test      Exit: 0 green, 1 RED, 2 cannot measure
set -euo pipefail

SKILL=.claude/skills/pr-review/SKILL.md
WF=.github/workflows

# Non-comment lines of a file that RUN mutation, as FILE:LINE on stdout.
mutation_lines() {
    awk '
      { q = $0; sub(/^[[:space:]]*#.*/, "", q)
        gsub(/cargo[ -]mutants (--version|--list)/, "", q)
        gsub(/mutants_diff_gate\.sh --(self-test|exempt-ref)/, "", q)
        gsub(/cargo install cargo-mutants/, "", q) }
      q ~ /cargo mutants([[:space:]]|$)/ || q ~ /\$\{?cargo\}?"? mutants[[:space:]]/ || q ~ /mutants_diff_gate\.sh/ { print FILENAME ":" FNR }
    ' "$1"
}

# The trigger keys of a workflow `on:` block, one per line (block, inline-list or scalar form).
triggers() {
    awk '
      /^on:[[:space:]]*$/ { on=1; next }
      /^on:[[:space:]]*\[/ { s=$0; sub(/^on:[[:space:]]*\[/, "", s); sub(/\].*/, "", s); n=split(s, a, /[[:space:],]+/); for (i=1;i<=n;i++) if (a[i]!="") print a[i]; next }
      /^on:[[:space:]]*[A-Za-z_]/ { s=$0; sub(/^on:[[:space:]]*/, "", s); sub(/[[:space:]#].*/, "", s); print s; next }
      on && /^[^[:space:]#]/ { on=0 }
      on && /^  [A-Za-z_][A-Za-z0-9_-]*:/ { k=$0; sub(/^  /, "", k); sub(/:.*/, "", k); print k }
    ' "$1"
}

scan() { # root -> RED lines on stdout; rc 2 when it cannot measure
    local root=$1 f rust t bad nightly=0
    [ -f "$root/$SKILL" ] || { echo "CANNOT-MEASURE: $SKILL missing under $root" >&2; return 2; }
    [ -d "$root/$WF" ] || { echo "CANNOT-MEASURE: $WF missing under $root" >&2; return 2; }
    rust="$(grep -m1 -E '^\|[[:space:]]*Rust source[[:space:]]*\|' "$root/$SKILL" || true)"
    if [ -z "$rust" ]; then echo "RED MN1 $SKILL: no §3.D \"Rust source\" row"
    elif grep -qiE 'advisory|MUTANT_BUDGET|cargo mutants --in-diff' <<<"$rust" || ! grep -q 'mutants-nightly\.yml' <<<"$rust"; then
        echo "RED MN1 $SKILL: the Rust source row still asks for mutation on a PR (or omits mutants-nightly.yml): $rust"
    fi
    for f in "$root"/$WF/*.yml "$root"/$WF/*.yaml; do
        [ -f "$f" ] || continue
        [ -n "$(mutation_lines "$f")" ] || continue
        t="$(triggers "$f" | sort -u | tr '\n' ' ')"
        bad="$(triggers "$f" | grep -vx 'schedule' | sort -u | tr '\n' ' ' || true)"
        if [ -z "$t" ]; then echo "RED MN2 ${f#"$root"/}: runs mutation and names no trigger"
        elif [ -n "$bad" ]; then echo "RED MN2 ${f#"$root"/}: runs mutation on a non-schedule trigger: ${bad% }"
        else nightly=$((nightly + 1)); fi
    done
    for f in "$root"/ci/sections.yml "$root"/ci/vendor/*.yml; do
        [ -f "$f" ] || continue
        mutation_lines "$f" | sed "s|^$root/|RED MN3 |; s|\$|: ci.yml runs this body on PR/push/merge_group|"
    done
    if [ -f "$root/Makefile" ]; then
        mutation_lines "$root/Makefile" | sed "s|^$root/|RED MN4 |; s|\$|: a local seat must not run mutation|"
    fi
    [ "$nightly" -gt 0 ] || echo "RED MN5 $WF: no schedule-only workflow runs mutation; the nightly no longer reports it"
    return 0
}

self_test() {
    local t out rc n=0
    t=$(mktemp -d "${TMPDIR:-/tmp}/mutnightly.XXXXXX") || exit 2
    trap 'rm -rf -- "${t:?}"' RETURN
    good() {
        rm -rf -- "${t:?}/.github" "${t:?}/.claude" "${t:?}/ci" "${t:?}/Makefile"
        mkdir -p "$t/$WF" "$t/.claude/skills/pr-review" "$t/ci/vendor"
        printf '| Rust source | not run on a PR: runs only in `.github/workflows/mutants-nightly.yml` | — |\n' > "$t/$SKILL"
        printf 'on:\n  schedule:\n    - cron: "0 1 * * 0"\njobs:\n  m:\n    steps:\n      - run: cargo install cargo-mutants --locked\n      - run: cargo mutants --version\n      - run: bash scripts/mutants_diff_gate.sh day.diff --cap 0\n' > "$t/$WF/mutants-nightly.yml"
        printf 'on:\n  pull_request:\n  push:\n  merge_group:\njobs:\n  a:\n    steps:\n      - run: bash scripts/mutants_diff_gate.sh --self-test\n      # cargo mutants --in-diff was here once\n      - run: bash scripts/release/x.sh --mutants\n' > "$t/$WF/ci.yml"
        printf 'jobs:\n  s:\n    steps:\n      - run: bash scripts/check_mutants_memory_cap.sh\n' > "$t/ci/sections.yml"
        printf 'mutants:\n\t@echo REFUSED; exit 1\nmutants-list:\n\t@cargo mutants --list | head\n' > "$t/Makefile"
    }
    expect() { # label want-substring ("" = green)
        n=$((n + 1))
        out=$(scan "$t" 2>&1) && rc=0 || rc=$?
        if [ -z "$2" ]; then [ -z "$out" ] || { echo "SELF-TEST FAIL $1: expected green, got: $out"; exit 1; }
        else case $out in *"$2"*) ;; *) echo "SELF-TEST FAIL $1: expected '$2', got: ${out:-green}"; exit 1 ;; esac; fi
        echo "  pass  $1"
    }
    good; expect "real-shape tree is green" ""
    good; printf '| Rust source | `cargo mutants --in-diff <diff>`, capped at `MUTANT_BUDGET=40` | advisory |\n' > "$t/$SKILL"; expect "MN1 advisory PR mutation row" "RED MN1 "
    good; printf '| Rust source | `cargo mutants --in-diff d` | **yes** |\n' > "$t/$SKILL"; expect "MN1 blocking PR mutation row" "RED MN1 "
    good; printf '| Rust source | not run on a PR | — |\n' > "$t/$SKILL"; expect "MN1 row that omits the nightly" "RED MN1 "
    good; : > "$t/$SKILL"; expect "MN1 row missing" "RED MN1 "
    good; sed -i 's/^  schedule:/  workflow_dispatch:\n  schedule:/' "$t/$WF/mutants-nightly.yml"; expect "MN2 nightly gains workflow_dispatch" "RED MN2 .github/workflows/mutants-nightly.yml: runs mutation on a non-schedule trigger: workflow_dispatch"
    good; sed -i 's/^  schedule:/  push:\n    branches: [main, "train\/**"]\n  schedule:/' "$t/$WF/mutants-nightly.yml"; expect "MN2 nightly gains push main + train/**" "non-schedule trigger: push"
    good; printf '      - run: cargo mutants --in-diff pr.diff -- --lib\n' >> "$t/$WF/ci.yml"; expect "MN2 cargo mutants in a PR workflow" "RED MN2 .github/workflows/ci.yml"
    good; printf '      - run: bash scripts/mutants_diff_gate.sh pr.diff --cap 40\n' >> "$t/$WF/ci.yml"; expect "MN2 diff gate in a PR workflow" "RED MN2 .github/workflows/ci.yml"
    good; printf 'on: [push, schedule]\njobs:\n  m:\n    steps:\n      - run: cargo mutants\n' > "$t/$WF/inline.yml"; expect "MN2 inline on: list with push" "RED MN2 .github/workflows/inline.yml"
    good; printf 'on: merge_group\njobs:\n  m:\n    steps:\n      - run: "$cargo" mutants --workspace\n' > "$t/$WF/scalar.yml"; expect "MN2 scalar on: + quoted \$cargo" "RED MN2 .github/workflows/scalar.yml"
    good; printf 'on:\n  workflow_call:\njobs:\n  m:\n    steps:\n      - run: cargo mutants -j 4\n' > "$t/$WF/called.yml"; expect "MN2 reusable workflow_call" "non-schedule trigger: workflow_call"
    good; printf 'jobs:\n  m:\n    steps:\n      - run: cargo mutants\n' > "$t/$WF/none.yml"; expect "MN2 no on: block" "names no trigger"
    good; printf '      - run: cargo mutants --in-diff x\n' >> "$t/ci/sections.yml"; expect "MN3 sections.yml body" "RED MN3 ci/sections.yml"
    good; printf '      - run: cargo mutants\n' > "$t/ci/vendor/sovereign-ci.yml"; expect "MN3 vendored reusable" "RED MN3 ci/vendor/sovereign-ci.yml"
    good; printf 'mutants-fast:\n\t@cargo mutants --shard 1/10 -- --lib\n' >> "$t/Makefile"; expect "MN4 Makefile recipe" "RED MN4 Makefile"
    good; rm -f -- "${t:?}/.github/workflows/mutants-nightly.yml"; expect "MN5 no nightly left (vacuous)" "RED MN5 "
    good; rm -f -- "${t:?}/.claude/skills/pr-review/SKILL.md"; out=$(scan "$t" 2>&1) && rc=0 || rc=$?
    [ "$rc" -eq 2 ] || { echo "SELF-TEST FAIL: a missing skill file must be 'cannot measure' (rc 2), got rc $rc"; exit 1; }
    echo "  pass  missing skill is cannot-measure"
    echo "self-test: $((n + 1)) cases ok"
}

case ${1-} in
    --self-test) self_test; exit $? ;;
esac
root=${1:-$(cd "$(dirname "$0")/../.." && pwd)}
out=$(scan "$root") || exit 2
if [ -n "$out" ]; then printf '%s\n' "$out"; echo "RED mutation-nightly-only"; exit 1; fi
echo "GREEN mutation-nightly-only (MN1-MN5): mutation runs only on schedule"
