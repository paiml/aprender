#!/usr/bin/env bash
# check_coverage_has_producers.sh -- every consumer of aprender's coverage number
# still has a producer (#3676).
#
# Operator 2026-09-21: "YES, coverage on tags release only." `ci / coverage` left
# the PR / merge-group / push path: it had gated nothing and measured 0 tests on
# the facade root. A deletion like that can starve a gate that READ the number
# from CI -- rmedia's release gate did exactly that ("could not look"). So the
# chain is asserted, not remembered:
#
#   R1  `make -n coverage-check` reaches llvm-cov -- the PRODUCER the
#       pre-publish dogfood uses measures coverage itself, reads nothing from CI.
#   R2  scripts/dogfood.sh still runs `make ... coverage-check` -- the release
#       CONSUMER exists (without it the chain would be vacuously intact).
#   R3  .github/workflows/coverage-nightly.yml triggers on `schedule:`, NOT on `v*`
#       tags, and runs scripts/coverage_receipt.sh on a live line -- the nightly is
#       the CI producer: it leaves coverage-receipt-<sha>.json for the commit measured.
#   R4  the Makefile defines a numeric COV_FLOOR -- the floor `make coverage` enforces.
#   R5  ci/sections.yml (#4433) hands sovereign-ci `coverage_on: tag` (no `skip_coverage: true`)
#       AND ci.yml's own `on:` has NO `v*` tag trigger (T39). `tag` keeps coverage off the
#       PR / queue / branch-push path: sovereign-ci's gate prints "coverage: NOT MEASURED"
#       there. `skip_coverage: true` was the first spelling (#3688): that gate counts a
#       skipped coverage job as a mandatory failure, so it turned `ci / gate` red on every
#       PR. The tag trigger is gone because nothing read the tag run and it could not
#       pass: a tag create has no base sha, and its coverage section built only the facade.
#   R6  the release gate reads coverage-nightly (`WF='coverage-nightly.yml'` on a live
#       line of scripts/release/tag_coverage_gate.sh), and no script under scripts/
#       lists ci.yml runs for a branch held in a variable -- the shape that read a tag's
#       ci.yml run, which no longer exists and would wait forever.
# Any link missing is FAIL; a file that cannot be read is ENV rc=2.
#
#   check_coverage_has_producers.sh              judge the tree
#   check_coverage_has_producers.sh --self-test  each rule against fixtures that break it
#
# Test seams: COVCHAIN_ROOT (the tree judged).
set -uo pipefail
ROOT="${COVCHAIN_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

# on_block <workflow> -> the lines of its top-level `on:` block, comments stripped
on_block() {
    awk '/^on:/{f=1; next} f && /^[^[:space:]#]/{exit} f' "$1" | sed 's/#.*$//'
}
# live <file> -> the file with comments stripped (a setting in a comment is not a setting)
live() { sed 's/#.*$//' "$1"; }

# tag_push_fires <workflow> -> rc 0 when a push of a tag would start it (GitHub: a `push:`
# filtered to branches only never fires for a tag; `tags:`, no filter at all, or an inline
# `on: [push]` does). Block or flow lists both count -- the old pattern saw only `[... v* ...]`.
# Case table (self-test rows R3/R5): branches only -> no; branches + tags [v*] -> yes;
# tags as a block list -> yes; bare `push:` -> yes; `on: [push, pull_request]` -> yes;
# branches-ignore only -> no; no push at all -> no.
tag_push_fires() {
    local f=$1 inline blk
    inline=$(sed -n 's/#.*$//; s/[[:space:]]*$//; s/^on:[[:space:]]*//p' "$f" | head -n 1)
    if [ -n "$inline" ]; then [[ "$inline" =~ (^|[^a-z_-])push([^a-z_-]|$) ]]; return; fi
    on_block "$f" | grep -qE '^  push:' || return 1
    blk=$(on_block "$f" | awk '/^  push:/{f=1; next} f && /^  [^[:space:]]/{exit} f')
    grep -qE '^[[:space:]]+tags:' <<<"$blk" && return 0
    grep -qE '^[[:space:]]+branches(-ignore)?:' <<<"$blk" && return 1
    return 0
}
# R6: a `gh run list` of ci.yml (or "CI") whose --branch is a variable -- how a tag's
# ci.yml run was read (tag_coverage_gate.sh before #4734). A literal branch (main) is fine.
CI_RUN_BY_VAR_RE='run list.*--workflow[= ].?(ci\.yml|CI).?([[:space:]]|$).*--branch[= ].?[$]'

judge() { # judge <root> -> 0 all links hold, 1 a link broke, 2 ENV
    local r=$1 bad=0 dry blk ci tg hits
    local mk="$r/Makefile" df="$r/scripts/dogfood.sh" wf="$r/.github/workflows/coverage-nightly.yml" cy="$r/.github/workflows/ci.yml" sy="$r/ci/sections.yml"
    for f in "$mk" "$df" "$wf" "$cy" "$sy"; do [ -r "$f" ] || { printf 'ENV   %s is not readable -- cannot judge, not a pass\n' "$f"; return 2; }; done

    dry=$(make -n -C "$r" coverage-check 2>/dev/null) || dry=""
    if [[ $dry == *"llvm-cov"* ]]; then printf 'ok    R1 make -n coverage-check reaches llvm-cov (the release producer measures, it reads nothing from CI)\n'
    else printf 'FAIL  R1 make -n coverage-check does not reach llvm-cov -- the dogfood coverage gate would measure nothing\n'; bad=1; fi

    # WIDENED (#3844): R2 asks "does the release still CHECK coverage", and it used to
    # test for one INVOCATION SHAPE -- `gate coverage make ... coverage-check`. When the
    # coverage row moved off the `gate` helper to a `mark` row (so a miss could be an
    # owed DEFER carrying its measured percentage rather than a NO-GO, #3839), coverage
    # was still run by the very next line and this rule reported "nothing checks
    # coverage at the release". A guard that names a helper cannot answer a question
    # about a behaviour.
    #
    # The new pattern requires an UNCOMMENTED line that invokes `make ... coverage-check`,
    # whatever wraps it. The `[^#[:space:]]` is load-bearing: it keeps the m_r2_comment
    # mutant RED, which a bare `.*` would have admitted. Case table, measured:
    #     gate coverage make -C … coverage-check          MATCH
    #     cov_out=$(make -C … coverage-check 2>&1); …      MATCH
    #   # gate coverage make -C x coverage-check           no
    #      # cov_out=$(make -C x coverage-check)           no
    if grep -qE '^[[:space:]]*[^#[:space:]].*make[[:space:]].*coverage-check' "$df"; then
        printf 'ok    R2 scripts/dogfood.sh runs make ... coverage-check (the release consumer exists)\n'
    else printf 'FAIL  R2 scripts/dogfood.sh no longer runs coverage-check -- nothing checks coverage at the release\n'; bad=1; fi

    blk=$(on_block "$wf")
    if ! grep -qE '^[[:space:]]+schedule:' <<<"$blk"; then
        printf 'FAIL  R3 coverage-nightly.yml must trigger on schedule (on: block, comments ignored)\n'; bad=1
    elif tag_push_fires "$wf"; then
        printf 'FAIL  R3 coverage-nightly.yml also triggers on a tag push -- the release reads the receipt for its commit; no tag needs its own run\n'; bad=1
    elif ! live "$wf" | grep -qE '(^|[[:space:]])scripts/coverage_receipt\.sh([[:space:]]|$)'; then
        printf 'FAIL  R3 coverage-nightly.yml no longer runs scripts/coverage_receipt.sh (on a live line) -- the release gate would find no receipt\n'; bad=1
    else printf 'ok    R3 coverage-nightly.yml triggers on schedule, not on tags, and writes the sha-keyed receipt\n'; fi

    if grep -qE '^COV_FLOOR[[:space:]]*:?=[[:space:]]*[0-9]+' "$mk"; then
        printf 'ok    R4 the Makefile defines a numeric COV_FLOOR\n'
    else printf 'FAIL  R4 the Makefile has no numeric COV_FLOOR -- make coverage enforces nothing\n'; bad=1; fi

    # #4433: the sovereign-ci `with:` block moved to ci/sections.yml (sovereign-ci:);
    # the `on:` trigger stays in ci.yml.
    ci=$(live "$sy")
    if grep -qE '^[[:space:]]+skip_coverage:[[:space:]]*true' <<<"$ci"; then
        printf 'FAIL  R5 ci/sections.yml sets skip_coverage: true -- sovereign-ci'"'"'s gate reads that skip as a mandatory failure (ci / gate red on every PR); use coverage_on: tag\n'; bad=1
    elif ! grep -qE "^[[:space:]]+coverage_on:[[:space:]]*['\"]?tag['\"]?[[:space:]]*$" <<<"$ci"; then
        printf 'FAIL  R5 ci/sections.yml does not hand sovereign-ci coverage_on: tag (on a live line)\n'; bad=1
    elif tag_push_fires "$cy"; then
        printf "FAIL  R5 ci.yml fires on a tag push again -- nothing reads that run, and it cannot pass (no base sha on a tag create)\n"; bad=1
    else printf 'ok    R5 ci/sections.yml: coverage_on: tag, and ci.yml does not fire on a tag push\n'; fi

    tg="$r/scripts/release/tag_coverage_gate.sh"
    if [ ! -r "$tg" ]; then printf 'ENV   %s is not readable -- cannot judge, not a pass\n' "$tg"; return 2; fi
    hits=$(find "$r/scripts" -name '*.sh' ! -name check_coverage_has_producers.sh -print0 2>/dev/null |
        xargs -0 -r grep -HnE "$CI_RUN_BY_VAR_RE" 2>/dev/null | grep -vE '^[^:]+:[0-9]+:[[:space:]]*#')
    hits=${hits//"$r/"/}
    if ! live "$tg" | grep -qE "^[[:space:]]*WF=['\"]?coverage-nightly\.yml['\"]?[[:space:]]*$"; then
        printf "FAIL  R6 scripts/release/tag_coverage_gate.sh does not judge coverage-nightly (no live WF='coverage-nightly.yml')\n"; bad=1
    elif [ "${#hits}" -gt 0 ]; then
        printf 'FAIL  R6 a script lists ci.yml runs for a branch held in a variable (a tag'"'"'s ci.yml run no longer exists):\n'
        printf '        %s\n' "$hits" | head -n 5; bad=1
    else printf 'ok    R6 the release gate reads coverage-nightly, and no script waits for a ci.yml run by a variable branch\n'; fi
    return "$bad"
}

# The header is every leading comment line, found rather than counted.
case "${1:-}" in -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== coverage producer chain: each rule must turn RED when its link breaks ==="
    d=$(mktemp -d "${TMPDIR:-/tmp}/covchain.XXXXXX") || exit 2
    rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
    trap 'rmtree "${d:-}"' EXIT
    bad=0; n=0
    fixture() { # fixture <dir> -> a tree where every link holds
        mkdir -p "$1/scripts/release" "$1/.github/workflows" "$1/ci"
        printf 'COV_FLOOR := 88\nLLVMCOV := llvm-cov\ncoverage:\n\t$(LLVMCOV) report --fail-under-lines $(COV_FLOOR)\ncoverage-check: coverage\n' > "$1/Makefile"
        printf '#!/usr/bin/env bash\n  gate coverage make -C "$d" coverage-check\n' > "$1/scripts/dogfood.sh"
        printf "on:\n  schedule:\n    - cron: '0 22 * * *'\n  workflow_dispatch: {}\njobs:\n  c:\n    steps:\n      - run: |\n          bash scripts/coverage_receipt.sh cov.log \"\$(git rev-parse HEAD)\" 89 out since\n" > "$1/.github/workflows/coverage-nightly.yml"
        printf "on:\n  push:\n    branches: [main, master]\n  pull_request:\n    branches: [main, master]\njobs: {}\n" > "$1/.github/workflows/ci.yml"
        printf "sovereign-ci:\n  uses: paiml/.github/.github/workflows/sovereign-ci.yml@x\n  with:\n    coverage_on: tag\n" > "$1/ci/sections.yml"
        printf "#!/usr/bin/env bash\nWF='coverage-nightly.yml'\n" > "$1/scripts/release/tag_coverage_gate.sh"
        printf '#!/usr/bin/env bash\ngh run list --workflow ci.yml --branch main --limit 1\n' > "$1/scripts/state.sh"
    }
    # one mutation per row, as a function (no eval: the mutation is code, not a string)
    ciyml()           { printf '%b\n  pull_request:\n    branches: [main, master]\njobs: {}\n' "$1" > .github/workflows/ci.yml; }
    m_none()          { :; }
    m_r1()            { sed -i 's/\$(LLVMCOV) report.*/echo nothing/' Makefile; }
    m_r2_gone()       { printf '#!/usr/bin/env bash\n' > scripts/dogfood.sh; }
    m_r2_comment()    { printf '#!/usr/bin/env bash\n# gate coverage make -C x coverage-check\n' > scripts/dogfood.sh; }
    m_r3_nosched()    { sed -i '/schedule:/d; /cron:/d' .github/workflows/coverage-nightly.yml; }
    m_r3_tagsback()   { sed -i "s/^  workflow_dispatch: {}/  push:\n    tags: ['v*']/" .github/workflows/coverage-nightly.yml; }
    m_r3_tagsblock()  { sed -i "s/^  workflow_dispatch: {}/  push:\n    tags:\n      - 'v*'/" .github/workflows/coverage-nightly.yml; }
    m_r3_noreceipt()  { sed -i '/coverage_receipt/d' .github/workflows/coverage-nightly.yml; }
    m_r3_rcptcomment(){ sed -i 's/^\( *\)bash scripts\/coverage_receipt/\1# bash scripts\/coverage_receipt/' .github/workflows/coverage-nightly.yml; }
    m_r4()            { sed -i '/^COV_FLOOR/d' Makefile; }
    m_r5_skip()       { sed -i 's/coverage_on: tag/skip_coverage: true/' ci/sections.yml; }
    m_r5_both()       { printf '    skip_coverage: true\n' >> ci/sections.yml; }
    m_r5_nocov()      { sed -i '/coverage_on:/d' ci/sections.yml; }
    m_r5_comment()    { sed -i 's/    coverage_on: tag/    # coverage_on: tag/' ci/sections.yml; }
    m_r5_always()     { sed -i 's/coverage_on: tag/coverage_on: always/' ci/sections.yml; }
    m_r5_tagsback()   { ciyml 'on:\n  push:\n    branches: [main, master]\n    tags: ["v*"]'; }
    m_r5_tagsblock()  { ciyml 'on:\n  push:\n    branches: [main]\n    tags:\n      - "v[0-9]*"'; }
    m_r5_barepush()   { ciyml 'on:\n  push:'; }
    m_r5_inline()     { printf 'on: [push, pull_request]\njobs: {}\n' > .github/workflows/ci.yml; }
    m_r5_tagcomment() { ciyml 'on:\n  push:\n    branches: [main, master]\n    # tags: ["v*"]'; }
    m_r5_brignore()   { ciyml 'on:\n  push:\n    branches-ignore: [gh-pages]'; }
    m_r5_nopush()     { ciyml 'on:\n  merge_group:'; }
    m_r6_wf()         { sed -i "s/^WF=.*/WF='ci.yml'/" scripts/release/tag_coverage_gate.sh; }
    m_r6_wfcomment()  { sed -i 's/^WF=/# WF=/' scripts/release/tag_coverage_gate.sh; }
    # The planted readers carry a literal '$', passed as %s so the fixture text is not shell code here.
    m_r6_reader()     { printf '    "%sGH" run list --repo "%sREPO" --workflow ci.yml --event push --branch "%s1" --limit 20\n' '$' '$' '$' >> scripts/release/tag_coverage_gate.sh; }
    m_r6_readercmt()  { printf '    # gh run list --workflow ci.yml --branch "%sT"\n' '$' >> scripts/state.sh; }
    m_r6_readerci()   { printf 'gh run list --workflow=CI --branch=%sT\n' '$' >> scripts/state.sh; }
    m_missing()       { rm -f .github/workflows/coverage-nightly.yml; }
    m_missing_ci()    { rm -f .github/workflows/ci.yml; }
    m_missing_sy()    { rm -f ci/sections.yml; }
    m_missing_tg()    { rm -f scripts/release/tag_coverage_gate.sh; }
    row() { # row WANT-RC LABEL MUTATION-FUNCTION
        local want=$1 label=$2 rc=0; n=$((n + 1)); rmtree "$d/t"; fixture "$d/t"
        ( cd "$d/t" && "$3" ) || { printf 'FAIL  row %s fixture mutation failed: %s\n' "$n" "$label"; bad=1; return; }
        judge "$d/t" > "$d/out" 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; sed 's/^/        /' "$d/out"; bad=1; fi
    }
    row 0 "every link holds -> PASS"                                            m_none
    row 1 "R1: coverage no longer runs llvm-cov -> RED"                         m_r1
    row 1 "R2: dogfood stops calling coverage-check -> RED"                     m_r2_gone
    row 1 "R2: the call only in a COMMENT -> RED"                               m_r2_comment
    row 1 "R3: nightly schedule removed -> RED"                                 m_r3_nosched
    row 1 "R3: the nightly fires on v* tags (flow list) -> RED"                 m_r3_tagsback
    row 1 "R3: the nightly fires on tags (block list) -> RED"                   m_r3_tagsblock
    row 1 "R3: the nightly no longer writes the receipt -> RED"                 m_r3_noreceipt
    row 1 "R3: the receipt call only in a COMMENT -> RED"                       m_r3_rcptcomment
    row 1 "R4: COV_FLOOR removed -> RED"                                        m_r4
    row 1 "R5: skip_coverage: true instead of coverage_on (the #3688 red) -> RED" m_r5_skip
    row 1 "R5: skip_coverage: true alongside coverage_on -> RED"                m_r5_both
    row 1 "R5: coverage_on removed -> RED"                                      m_r5_nocov
    row 1 "R5: coverage_on only in a COMMENT -> RED"                            m_r5_comment
    row 1 "R5: coverage_on: always (coverage back on every PR) -> RED"          m_r5_always
    row 1 "R5: ci.yml's v* tag trigger back (flow list) -> RED"                 m_r5_tagsback
    row 1 "R5: ci.yml tag trigger as a block list -> RED"                       m_r5_tagsblock
    row 1 "R5: ci.yml bare push: (fires on tags too) -> RED"                    m_r5_barepush
    row 1 "R5: ci.yml on: [push, pull_request] (inline) -> RED"                 m_r5_inline
    row 0 "R5: ci.yml tag trigger only in a COMMENT -> PASS (not a trigger)"    m_r5_tagcomment
    row 0 "R5: ci.yml push: branches-ignore only (no tag) -> PASS"              m_r5_brignore
    row 0 "R5: ci.yml with no push trigger at all -> PASS"                      m_r5_nopush
    row 1 "R6: the release gate reads ci.yml, not the nightly -> RED"           m_r6_wf
    row 1 "R6: the gate's WF= only in a COMMENT -> RED"                         m_r6_wfcomment
    row 1 "R6: a gh run list of ci.yml by a variable branch (the old main lookup) -> RED" m_r6_reader
    row 1 "R6: the same lookup as --workflow=CI, branch in a variable -> RED"   m_r6_readerci
    row 0 "R6: that lookup only in a COMMENT -> PASS"                           m_r6_readercmt
    row 2 "coverage-nightly.yml missing is ENV rc=2, never a pass"              m_missing
    row 2 "ci.yml missing is ENV rc=2, never a pass"                            m_missing_ci
    row 2 "ci/sections.yml missing is ENV rc=2, never a pass"                   m_missing_sy
    row 2 "scripts/release/tag_coverage_gate.sh missing is ENV rc=2, never a pass" m_missing_tg
    [ "$bad" = 0 ] && { printf 'SELF-TEST PASSED: %s rows\n' "$n"; exit 0; }
    printf 'SELF-TEST FAILED\n' >&2; exit 1
fi

echo "=== every consumer of aprender's coverage number has a producer (check_coverage_has_producers.sh) ==="
judge "$ROOT"; rc=$?
[ "$rc" = 0 ] && echo PASS || echo "FAIL (rc=$rc)" >&2
exit "$rc"
