#!/usr/bin/env bash
# check_complexity_ratchet.sh - per-function complexity may only fall.
#
# WHY THIS EXISTS
# ---------------
# The only complexity gate in this repository was the LOCAL pmat pre-commit
# hook (.git/hooks/pre-commit, PMAT_MAX_CYCLOMATIC_COMPLEXITY=30,
# PMAT_MAX_COGNITIVE_COMPLEXITY=25, run over STAGED files only).
# .github/workflows/ci.yml ran no complexity check at all. The consequence is
# not "debt accumulates"; it is worse and it is asymmetric:
#
#   * a pull request may LAND a function at cognitive 69, because nothing in
#     CI looks;
#   * and from then on every LOCAL commit that touches that file is refused by
#     the hook, including a commit that has nothing to do with the offending
#     function.
#
# So the merge queue writes the debt and the individual developer pays it, at
# the worst possible moment, on unrelated work. Three live examples on
# 68b059ca:
#
#   crates/apr-cli/src/commands/serve/handlers.rs   cognitive 69
#   crates/apr-cli/src/commands/eval/inference.rs   five functions, 31-40
#   crates/apr-cli/src/commands/tokenize.rs         run_encode_corpus 34/89
#
# WHY A RATCHET AND NOT A GATE
# ----------------------------
# 715 functions in the tree are already over one threshold or the other.
# Turning the hook's rule on in CI outright would red every pull request from
# the first one, and a gate that cannot go green gets disabled - which is how
# this repository lost the last two gates it turned on outright. So the
# existing offenders are recorded, and the recorded numbers may only FALL:
#
#   NEW    a function over a threshold with no row            -> RED
#   GROWN  a recorded function whose number rose              -> RED
#   STALE  a recorded function now under BOTH thresholds      -> RED (delete it)
#
# STALE is the half that makes this a ratchet rather than an allowlist. Without
# it a row survives its own repair, and the next regression at that coordinate
# is admitted for free by a row nobody noticed was already spent.
#
# WHAT THE ROWS MEAN
#
#     <path>::<function> <cyclomatic> <cognitive>
#
# Both numbers are recorded even when only one is over its threshold, because
# the rule is "either", and a function that is legal on cyclomatic today must
# not become illegal on it tomorrow while its row says nothing. Line numbers
# are deliberately NOT part of the key: a file:line baseline drifts the moment
# anything above it is edited, and CI reads the drift as growth (that shape has
# already cost this repository a ratchet).
#
# THE THRESHOLDS ARE THE HOOK'S, AND THEY ARE NOT A DIAL. 30 and 25 are read
# out of .git/hooks/pre-commit. Raising either here would let CI bless code the
# developer's own commit will then refuse, which is the defect this file is
# about, inverted.
#
#   bash scripts/check_complexity_ratchet.sh              # check
#   bash scripts/check_complexity_ratchet.sh --selftest   # case table
#   bash scripts/check_complexity_ratchet.sh --update     # re-baseline
#
# Refs: PMAT-746.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASELINE_REL='scripts/complexity_baseline.txt'
BASELINE="${REPO_ROOT}/${BASELINE_REL}"
ROWS_PY="${REPO_ROOT}/scripts/lib/complexity_rows.py"

# Verbatim from .git/hooks/pre-commit. See the header: these are not a dial.
MAX_CYCLOMATIC=30
MAX_COGNITIVE=25

# Vacuity floor for the real run. A universe that collapsed would report zero
# offenders and read exactly like a clean tree - the failure mode this
# repository has now found in a dozen guards. 10255 .rs files are tracked on
# 68b059ca; the floor is set well below that so an ordinary deletion cannot
# trip it, and far above zero so a broken scan cannot pass.
MIN_RS_FILES=5000

# pmat is fed an explicit file list, and a single argv entry is capped at
# 128 KiB by the kernel (MAX_ARG_STRLEN). The longest tracked path is 129
# bytes, so 400 paths per invocation leaves an order of magnitude of headroom.
CHUNK=400

# ---------------------------------------------------------------------------
# THE UNIVERSE. Tracked UNION working tree, because tracked-only is a free
# pass: an untracked .rs file is invisible to `git ls-files`, and untracked is
# how a new file arrives. That shape has cost this repository four guards.
#
# It is deliberately NOT pmat's own project scan. `.pmatignore` excludes 1336
# tracked .rs files (all of crates/aprender-serve's tests, benches, examples
# and bin entry points, plus a reference monolith), and the pre-commit hook
# does NOT honour it: the hook runs `pmat analyze complexity --file <staged>`,
# which reads whatever it is handed. A CI universe narrower than the hook's
# would leave exactly the files whose debt blocks local commits unguarded.
cx_universe() { # cx_universe <root> -> repo-relative .rs paths, sorted, unique
    local root="$1"
    {
        git -C "$root" ls-files -- '*.rs' 2>/dev/null || true
        find "$root" -type f -name '*.rs' \
            -not -path '*/.git/*' \
            -not -path '*/target/*' \
            -not -path '*/target_disk/*' \
            -not -path '*/node_modules/*' \
            -not -path '*/.claude/worktrees/*' \
            -printf '%P\n' 2>/dev/null || true
    } | LC_ALL=C sort -u | grep -v '^$' || true
}

# Data lines of a row file: comments and blanks are not rows.
cx_data() { # cx_data <file>
    grep -vE '^[[:space:]]*(#|$)' "$1" 2>/dev/null || true
}

# ---------------------------------------------------------------------------
# THE MEASUREMENT. Emits one row per offender, sorted, on stdout.
cx_measure() { # cx_measure <root> <scratch-dir>
    local root="$1" work="$2" list chunk files
    list="$work/universe.txt"
    cx_universe "$root" > "$list"
    files=$(grep -c . "$list" || true)

    # A comma is pmat's own list separator, so a path containing one would be
    # split into two nonexistent files and silently drop from the scan.
    if grep -q ',' "$list"; then
        printf 'FAIL: a .rs path contains a comma, which is pmat --files own separator:\n' >&2
        grep ',' "$list" | sed 's/^/      /' >&2
        return 1
    fi
    if [ "$files" -eq 0 ]; then
        printf 'FAIL: no .rs file found under %s. An empty scan is not a clean scan.\n' "$root" >&2
        return 1
    fi

    rm -rf "${work:?}/chunks"
    mkdir -p "$work/chunks"
    split -l "$CHUNK" -d -a 4 "$list" "$work/chunks/c"
    for chunk in "$work"/chunks/c*; do
        case "$chunk" in *.json | *.err) continue ;; esac
        if ! ( cd "$root" && pmat analyze complexity \
                    --files "$(paste -sd, "$chunk")" \
                    --format json --top-files 0 ) > "$chunk.json" 2> "$chunk.err"; then
            printf 'FAIL: pmat analyze complexity failed on %s\n' "$chunk" >&2
            sed 's/^/      | /' "$chunk.err" >&2
            return 1
        fi
    done

    CX_MAX_CYCLOMATIC="$MAX_CYCLOMATIC" CX_MAX_COGNITIVE="$MAX_COGNITIVE" \
        python3 "$ROWS_PY" "$work"/chunks/c*.json
}

# ---------------------------------------------------------------------------
# THE RATCHET. Pure: two row files in, findings out, rc 1 if any. Everything
# the case table drives goes through here.
#
# The awk is fed PRE-FILTERED data rather than filtering comments itself: an
# awk program carrying a bracket-and-paren regex reads to bashrs as a `[ ` test
# and lands SC1028 error lines in a shrink-only lint baseline.
#
# THE TWO INPUTS ARE TAGGED, NOT COUNTED. The idiomatic `NR == FNR` two-file
# awk is WRONG here and the case table caught it on its first run: when the
# first file is EMPTY -- which is exactly the "no baseline yet" row -- FNR
# restarts at 1 for the second file, `NR == FNR` is true for its first record,
# and every current offender is loaded as though it were the baseline. The
# checker then reported STALE for the two functions it had just measured.
# A tag makes the discrimination explicit and empty-safe.
cx_verdict() { # cx_verdict <baseline-rows> <current-rows>
    local findings
    findings=$( { cx_data "$1" | sed 's/^/B /'
                  cx_data "$2" | sed 's/^/C /'; } | LC_ALL=C awk '
        $1 == "B" { bcyc[$2] = $3; bcog[$2] = $4; base[$2] = 1; next }
        $1 == "C" {
            cur[$2] = 1
            if (!($2 in base)) {
                printf "  NEW    %s  cyclomatic %s cognitive %s\n", $2, $3, $4
            } else {
                if ($3+0 > bcyc[$2]+0) {
                    printf "  GROWN  %s  cyclomatic %s -> %s\n", $2, bcyc[$2], $3
                }
                if ($4+0 > bcog[$2]+0) {
                    printf "  GROWN  %s  cognitive %s -> %s\n", $2, bcog[$2], $4
                }
            }
        }
        END {
            for (k in base) {
                if (!(k in cur)) {
                    printf "  STALE  %s  now under both thresholds; delete the row\n", k
                }
            }
        }
    ' | LC_ALL=C sort)
    if [ -n "$findings" ]; then
        printf '%s\n' "$findings"
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------------------
# THE CASE TABLE. Both polarities for every rule, over a THROWAWAY CRATE whose
# four functions were measured, not guessed:
#
#     tidy             1 / 0    under both
#     nested           7 / 21   under both, and a near miss on cognitive
#     cognitive_only   8 / 28   over cognitive ONLY
#     branchy         35 / 34   over both
#
# `cognitive_only` is the row that makes "either threshold" a measured claim
# rather than a sentence in a header, and `nested` is the control that says the
# detector is not simply reporting everything it sees.
#
# WHAT THIS TABLE DOES NOT COVER, stated rather than left to be found: the real
# run also calls baseline_ratchet_check (the anti-laundering comparison against
# origin/main, whose own case table is scripts/check_baseline_ratchets.sh) and
# the MIN_RS_FILES vacuity floor. Both need a git repository with a protected
# comparand and are exercised there, not here.
cx_selftest() {
    local td fixture clean measured rows fails=0 count=0

    td=$(mktemp -d) || return 1
    # shellcheck disable=SC2064
    trap "rm -rf '${td:?}'" EXIT

    fixture="$td/fixture"
    mkdir -p "$fixture/src"
    printf '[package]\nname = "cx-selftest-fixture"\nversion = "0.0.0"\nedition = "2021"\n' \
        > "$fixture/Cargo.toml"
    cx_write_fixture_lib > "$fixture/src/lib.rs"

    clean="$td/clean"
    mkdir -p "$clean/src"
    printf '[package]\nname = "cx-selftest-clean"\nversion = "0.0.0"\nedition = "2021"\n' \
        > "$clean/Cargo.toml"
    printf 'pub fn tidy(a: i32) -> i32 {\n    a + 1\n}\n' > "$clean/src/lib.rs"

    printf -- '--- case table -----------------------------------------------------\n'

    measured="$td/measured.txt"
    mkdir -p "$td/w1"
    if ! cx_measure "$fixture" "$td/w1" > "$measured" 2> "$td/w1.err"; then
        printf '  FAIL  the measurement itself failed:\n'
        sed 's/^/          | /' "$td/w1.err"
        printf '\nSELF-TEST FAILED\n'
        return 1
    fi

    cx_row() { # cx_row <name> <want-red|want-green> <needle> <baseline> <current>
        local name="$1" want="$2" needle="$3" out rc ok=1
        out=$(cx_verdict "$4" "$5") && rc=0 || rc=$?
        count=$((count + 1))
        case "$want" in
            want-red)   [ "$rc" -ne 0 ] || ok=0 ;;
            want-green) [ "$rc" -eq 0 ] || ok=0 ;;
        esac
        if [ -n "$needle" ] && ! grep -qF -- "$needle" <<< "$out"; then ok=0; fi
        if [ "$ok" -eq 1 ]; then
            printf '  ok    %-10s %s\n' "$want" "$name"
        else
            printf '  BROKE %-10s %s (rc=%s)\n' "$want" "$name" "$rc"
            printf '%s\n' "$out" | sed 's/^/          | /'
            fails=$((fails + 1))
        fi
    }

    cx_assert() { # cx_assert <name> <ok:0|1> <detail>
        count=$((count + 1))
        if [ "$2" -eq 0 ]; then
            printf '  ok    %-10s %s\n' 'measured' "$1"
        else
            printf '  BROKE %-10s %s: %s\n' 'measured' "$1" "$3"
            fails=$((fails + 1))
        fi
    }

    # ROW 1/2. THE MEASUREMENT, both polarities. Without these every row below
    # could pass over an empty file.
    rows=$(grep -c . "$measured" || true)
    if [ "$rows" -eq 2 ] \
       && grep -qF 'src/lib.rs::branchy 35 34' "$measured" \
       && grep -qF 'src/lib.rs::cognitive_only 8 28' "$measured"; then
        cx_assert 'both offenders found, cognitive-only included' 0 ''
    else
        cx_assert 'both offenders found, cognitive-only included' 1 \
            "expected 2 rows, got ${rows}: $(tr '\n' ';' < "$measured")"
    fi
    if grep -qE '::(tidy|nested) ' "$measured"; then
        cx_assert 'a function under both thresholds is NOT reported' 1 \
            'tidy or nested was reported as an offender'
    else
        cx_assert 'a function under both thresholds is NOT reported' 0 ''
    fi

    # ROW 3/4. NEW, both polarities.
    : > "$td/empty.txt"
    cx_row 'a new offender with no row'          want-red   'NEW'   "$td/empty.txt" "$measured"
    cx_row 'a baselined offender'                want-green ''      "$measured"     "$measured"

    # ROW 5/6/7. GROWN, both metrics and the falling case. THIS IS THE
    # MUST-FIRE MUTATION TARGET: delete either comparison in cx_verdict and the
    # matching row BROKEs.
    sed 's|^src/lib.rs::branchy 35 34$|src/lib.rs::branchy 34 34|' "$measured" > "$td/grew_cyc.txt"
    cx_row 'a baselined offender grown on cyclomatic' want-red 'GROWN' "$td/grew_cyc.txt" "$measured"
    sed 's|^src/lib.rs::cognitive_only 8 28$|src/lib.rs::cognitive_only 8 27|' "$measured" > "$td/grew_cog.txt"
    cx_row 'a baselined offender grown on cognitive'  want-red 'GROWN' "$td/grew_cog.txt" "$measured"
    sed 's|^src/lib.rs::branchy 35 34$|src/lib.rs::branchy 99 99|' "$measured" > "$td/fell.txt"
    cx_row 'a baselined offender that improved'       want-green ''    "$td/fell.txt"     "$measured"

    # ROW 8. STALE. A row whose function is no longer over either threshold.
    { cat "$measured"; printf 'src/lib.rs::tidy 44 44\n'; } > "$td/stale.txt"
    cx_row 'a fixed function whose row was kept'      want-red 'STALE' "$td/stale.txt"    "$measured"

    # ROW 9. THE GREEN CONTROL AT THE OTHER END: a crate with no offender at
    # all, measured for real, against an empty baseline. A checker that redded
    # everything would pass rows 3, 5, 6 and 8.
    mkdir -p "$td/w2"
    if ! cx_measure "$clean" "$td/w2" > "$td/clean_rows.txt" 2> "$td/w2.err"; then
        cx_assert 'a clean crate measures without error' 1 "$(cat "$td/w2.err")"
    else
        cx_assert 'a clean crate measures without error' 0 ''
    fi
    cx_row 'a clean tree against an empty baseline' want-green '' "$td/empty.txt" "$td/clean_rows.txt"

    if [ "$count" -lt 9 ]; then
        printf '  BROKE case table has %s row(s); at least 9 are required\n' "$count"
        fails=$((fails + 1))
    fi
    printf '  %s row(s), %s failure(s)\n' "$count" "$fails"
    if [ "$fails" -ne 0 ]; then
        printf '\nSELF-TEST FAILED\n'
        return 1
    fi
    printf '\nSELF-TEST PASSED (%s/%s)\n' "$count" "$count"
    return 0
}

# The fixture, emitted rather than stored, so the table cannot drift from a
# file nobody looks at. Every number in the header above came from running
# pmat over exactly this text.
cx_write_fixture_lib() {
    local i=0
    printf 'pub fn tidy(a: i32) -> i32 {\n    a + 1\n}\n\n'
    printf 'pub fn branchy(v: &[i32]) -> i32 {\n    let mut n = 0;\n'
    while [ "$i" -lt 34 ]; do
        printf '    if v[%s] > %s {\n        n += %s;\n    }\n' "$i" "$i" "$i"
        i=$((i + 1))
    done
    printf '    n\n}\n\n'
    printf 'pub fn nested(v: &[i32]) -> i32 {\n    let mut n = 0;\n'
    printf '    for a in v {\n        if *a > 0 {\n            for b in v {\n'
    printf '                if *b > 1 {\n                    if *b > 2 {\n'
    printf '                        if *b > 3 {\n                            n += 1;\n'
    printf '                        }\n                    }\n                }\n'
    printf '            }\n        }\n    }\n    n\n}\n\n'
    printf 'pub fn cognitive_only(v: &[i32]) -> i32 {\n    let mut n = 0;\n'
    printf '    for a in v {\n        if *a > 0 {\n            for b in v {\n'
    printf '                if *b > 1 {\n                    if *b > 2 {\n'
    printf '                        if *b > 3 {\n                            if *b > 4 {\n'
    printf '                                n += 1;\n                            }\n'
    printf '                        }\n                    }\n                }\n'
    printf '            }\n        }\n    }\n    n\n}\n'
}

# ---------------------------------------------------------------------------
if [ "${1:-}" = '--selftest' ] || [ "${1:-}" = '--self-test' ]; then
    cx_selftest
    exit $?
fi

printf '=== per-function complexity may only fall (check_complexity_ratchet.sh) ===\n'

if ! command -v pmat > /dev/null 2>&1; then
    printf 'SKIP: pmat is not installed; install it with `cargo install pmat --locked`.\n' >&2
    printf 'This is a hard failure in CI, where the workflow installs it first.\n' >&2
    [ "${CI:-}" = 'true' ] && exit 1
    exit 0
fi
if ! command -v python3 > /dev/null 2>&1; then
    printf 'FAIL: python3 is required to read pmat JSON.\n' >&2
    exit 1
fi

# PROVE THE MECHANISM ENGAGED, do not label the run by intent: print WHICH
# binary produced these numbers. A complexity verdict from an unnamed pmat is a
# confident answer about code you may not be running.
#
# NO CLOCK IS READ HERE, deliberately. The scan costs 8-9s wall on this host
# (10255 files, 11 pmat invocations, measured on 68b059ca), which is why it is
# affordable per-PR -- but the duration is neither printed nor asserted. A
# wall-clock assertion in a required check has failed eleven times in this
# repository (#2671), and `date` in a guard is a DET002 error line that lands
# in a shrink-only lint baseline.
printf 'pmat: %s (%s)\n' "$(command -v pmat)" "$(pmat --version 2>/dev/null | head -1)"

WORK=$(mktemp -d) || exit 1
trap 'rm -rf "${WORK:?}"' EXIT

if ! cx_measure "$REPO_ROOT" "$WORK" > "$WORK/current.txt" 2> "$WORK/measure.err"; then
    sed 's/^/      | /' "$WORK/measure.err" >&2
    printf 'FAIL: the complexity scan did not complete, so growth is UNMEASURED.\n' >&2
    exit 1
fi
sed 's/^/  /' "$WORK/measure.err"

SCANNED=$(grep -c . "$WORK/universe.txt" || true)
CURRENT=$(grep -c . "$WORK/current.txt" || true)
printf '%s .rs file(s) scanned, %s function(s) over cyclomatic>%s or cognitive>%s\n' \
    "$SCANNED" "$CURRENT" "$MAX_CYCLOMATIC" "$MAX_COGNITIVE"

if [ "$SCANNED" -lt "$MIN_RS_FILES" ]; then
    printf '\nFAIL (vacuity): only %s .rs file(s) found, expected %s+.\n' "$SCANNED" "$MIN_RS_FILES"
    printf 'The scan is broken, not the code. Fix it rather than this number.\n'
    exit 1
fi

if [ "${1:-}" = '--update' ]; then
    {
        printf '# complexity_baseline.txt - functions over the pre-commit hook thresholds\n'
        printf '# (cyclomatic > %s or cognitive > %s), as "<path>::<function> <cyclomatic> <cognitive>".\n' \
            "$MAX_CYCLOMATIC" "$MAX_COGNITIVE"
        printf '# SHRINK-ONLY. Regenerate with: bash scripts/check_complexity_ratchet.sh --update\n'
        printf '# Owner: scripts/check_complexity_ratchet.sh (PMAT-746).\n'
        cat "$WORK/current.txt"
    } > "$BASELINE"
    printf 'baseline rewritten: %s row(s)\n' "$CURRENT"
    exit 0
fi

if [ ! -f "$BASELINE" ]; then
    printf 'FAIL: %s missing. Run --update once to establish it.\n' "$BASELINE"
    exit 1
fi

RECORDED=$(cx_data "$BASELINE" | grep -c . || true)
printf 'baseline %s row(s)\n' "$RECORDED"

# THE RATCHET IS A PROPERTY OF THE DIFF, NOT OF THE TREE.
#
# Everything below compares the scan against the baseline AS IT STANDS IN THE
# WORKING TREE, and that alone is not a ratchet: NEW and STALE are the only two
# properties a working tree can answer, and a commit that appends a row AND
# lands the matching offender satisfies both at once. Twelve guards in this
# repository failed exactly that probe. So the file is ALSO compared against a
# ref a pull request cannot rewrite.
#
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "${REPO_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1
RATCHET_RC=0
baseline_ratchet_check "$REPO_ROOT" "$BASELINE_REL" keyed2 || RATCHET_RC=$?

VERDICT_RC=0
FINDINGS=$(cx_verdict "$BASELINE" "$WORK/current.txt") || VERDICT_RC=$?

if [ "$VERDICT_RC" -ne 0 ]; then
    printf '\nFAIL: the complexity ratchet moved backwards.\n'
    printf '%s\n' "$FINDINGS"
    printf '\n  NEW    the function is over a threshold and has no row. Split it, or\n'
    printf '         reduce it below cyclomatic %s / cognitive %s. The baseline is\n' \
        "$MAX_CYCLOMATIC" "$MAX_COGNITIVE"
    printf '         SHRINK-ONLY against origin/main: appending a row is REFUSED,\n'
    printf '         because a row and its violation in one commit is the laundering\n'
    printf '         shape this repository has already found twelve times.\n'
    printf '  GROWN  the recorded number is the ceiling. It may fall, never rise.\n'
    printf '  STALE  the function is fixed - delete its row in the same commit, or\n'
    printf '         the next regression at that coordinate lands for free.\n'
    printf '         bash scripts/check_complexity_ratchet.sh --update\n'
fi

if [ "$RATCHET_RC" -ne 0 ] || [ "$VERDICT_RC" -ne 0 ]; then
    exit 1
fi

printf 'PASS (ratcheted): %s recorded offender(s), none new, none grown, none stale.\n' "$RECORDED"
exit 0
