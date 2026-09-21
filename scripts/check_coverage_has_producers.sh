#!/usr/bin/env bash
# check_coverage_has_producers.sh -- every consumer of aprender's coverage number
# still has a producer (#3676).
#
# Operator 2026-09-21: "YES, coverage on tags release only." `ci / coverage` left
# the PR / merge-group / push path (ci.yml `skip_coverage: true`; it had gated
# nothing and measured 0 tests on the facade root). A deletion like that can
# starve a gate that READ the number from CI -- rmedia's release gate did
# exactly that ("could not look"). So the chain is asserted, not remembered:
#
#   R1  `make -n coverage-check` reaches llvm-cov -- the PRODUCER the
#       pre-publish dogfood uses measures coverage itself, reads nothing from CI.
#   R2  scripts/dogfood.sh still runs `make ... coverage-check` -- the release
#       CONSUMER exists (without it the chain would be vacuously intact).
#   R3  .github/workflows/coverage-nightly.yml triggers on `schedule:` AND on a
#       `push:` of `v*` tags -- the tag gets its own COV_FLOOR run.
#   R4  the Makefile defines a numeric COV_FLOOR -- the floor `make coverage` enforces.
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

judge() { # judge <root> -> 0 all links hold, 1 a link broke, 2 ENV
    local r=$1 bad=0 dry blk
    local mk="$r/Makefile" df="$r/scripts/dogfood.sh" wf="$r/.github/workflows/coverage-nightly.yml"
    for f in "$mk" "$df" "$wf"; do [ -r "$f" ] || { printf 'ENV   %s is not readable -- cannot judge, not a pass\n' "$f"; return 2; }; done

    dry=$(make -n -C "$r" coverage-check 2>/dev/null) || dry=""
    if [[ $dry == *"llvm-cov"* ]]; then printf 'ok    R1 make -n coverage-check reaches llvm-cov (the release producer measures, it reads nothing from CI)\n'
    else printf 'FAIL  R1 make -n coverage-check does not reach llvm-cov -- the dogfood coverage gate would measure nothing\n'; bad=1; fi

    if grep -qE '^[[:space:]]*gate[[:space:]]+coverage[[:space:]]+make[[:space:]].*coverage-check' "$df"; then
        printf 'ok    R2 scripts/dogfood.sh runs make ... coverage-check (the release consumer exists)\n'
    else printf 'FAIL  R2 scripts/dogfood.sh no longer runs coverage-check -- nothing checks coverage at the release\n'; bad=1; fi

    blk=$(on_block "$wf")
    if grep -qE '^[[:space:]]+schedule:' <<<"$blk" && grep -qE "^[[:space:]]+tags:[[:space:]]*\[[^]]*['\"]?v\*['\"]?" <<<"$blk"; then
        printf 'ok    R3 coverage-nightly.yml triggers on schedule AND on v* tags\n'
    else printf 'FAIL  R3 coverage-nightly.yml must trigger on schedule AND on push tags [v*] (on: block, comments ignored)\n'; bad=1; fi

    if grep -qE '^COV_FLOOR[[:space:]]*:?=[[:space:]]*[0-9]+' "$mk"; then
        printf 'ok    R4 the Makefile defines a numeric COV_FLOOR\n'
    else printf 'FAIL  R4 the Makefile has no numeric COV_FLOOR -- make coverage enforces nothing\n'; bad=1; fi
    return "$bad"
}

case "${1:-}" in -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== coverage producer chain: each rule must turn RED when its link breaks ==="
    d=$(mktemp -d "${TMPDIR:-/tmp}/covchain.XXXXXX") || exit 2
    rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
    trap 'rmtree "${d:-}"' EXIT
    bad=0; n=0
    fixture() { # fixture <dir> -> a tree where every link holds
        mkdir -p "$1/scripts" "$1/.github/workflows"
        printf 'COV_FLOOR := 88\nLLVMCOV := llvm-cov\ncoverage:\n\t$(LLVMCOV) report --fail-under-lines $(COV_FLOOR)\ncoverage-check: coverage\n' > "$1/Makefile"
        printf '#!/usr/bin/env bash\n  gate coverage make -C "$d" coverage-check\n' > "$1/scripts/dogfood.sh"
        printf "on:\n  schedule:\n    - cron: '0 22 * * *'\n  push:\n    tags: ['v*']\njobs: {}\n" > "$1/.github/workflows/coverage-nightly.yml"
    }
    # one mutation per row, as a function (no eval: the mutation is code, not a string)
    m_none()        { :; }
    m_r1()          { sed -i 's/\$(LLVMCOV) report.*/echo nothing/' Makefile; }
    m_r2_gone()     { printf '#!/usr/bin/env bash\n' > scripts/dogfood.sh; }
    m_r2_comment()  { printf '#!/usr/bin/env bash\n# gate coverage make -C x coverage-check\n' > scripts/dogfood.sh; }
    m_r3_gone()     { sed -i '/push:/d; /tags:/d' .github/workflows/coverage-nightly.yml; }
    m_r3_comment()  { sed -i "s/^    tags: \['v\*'\]/    # tags: ['v*']/" .github/workflows/coverage-nightly.yml; }
    m_r3_nosched()  { sed -i '/schedule:/d; /cron:/d' .github/workflows/coverage-nightly.yml; }
    m_r3_otherkey() { printf "on:\n  schedule:\n    - cron: '0 22 * * *'\nenv:\n  tags: ['v*']\njobs: {}\n" > .github/workflows/coverage-nightly.yml; }
    m_r4()          { sed -i '/^COV_FLOOR/d' Makefile; }
    m_missing()     { rm -f .github/workflows/coverage-nightly.yml; }
    row() { # row WANT-RC LABEL MUTATION-FUNCTION
        local want=$1 label=$2 rc=0; n=$((n + 1)); rmtree "$d/t"; fixture "$d/t"
        ( cd "$d/t" && "$3" ) || { printf 'FAIL  row %s fixture mutation failed: %s\n' "$n" "$label"; bad=1; return; }
        judge "$d/t" > "$d/out" 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label"; sed 's/^/        /' "$d/out"; bad=1; fi
    }
    row 0 "every link holds -> PASS"                                   m_none
    row 1 "R1: coverage no longer runs llvm-cov -> RED"                m_r1
    row 1 "R2: dogfood stops calling coverage-check -> RED"            m_r2_gone
    row 1 "R2: the call only in a COMMENT -> RED"                      m_r2_comment
    row 1 "R3: the tag trigger removed -> RED"                         m_r3_gone
    row 1 "R3: the tag trigger only in a COMMENT -> RED"               m_r3_comment
    row 1 "R3: schedule removed (tags only) -> RED"                    m_r3_nosched
    row 1 "R3: tags: under a DIFFERENT top-level key, not on: -> RED"  m_r3_otherkey
    row 1 "R4: COV_FLOOR removed -> RED"                               m_r4
    row 2 "a missing file is ENV rc=2, never a pass"                   m_missing
    [ "$bad" = 0 ] && { printf 'SELF-TEST PASSED: %s rows\n' "$n"; exit 0; }
    printf 'SELF-TEST FAILED\n' >&2; exit 1
fi

echo "=== every consumer of aprender's coverage number has a producer (check_coverage_has_producers.sh) ==="
judge "$ROOT"; rc=$?
[ "$rc" = 0 ] && echo PASS || echo "FAIL (rc=$rc)" >&2
exit "$rc"
