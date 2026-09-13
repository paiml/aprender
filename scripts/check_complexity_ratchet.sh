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
# STALE is the half that made this a ratchet rather than an allowlist WHEN THE
# COMPARAND WAS A FILE. Without it a row survives its own repair, and the next
# regression at that coordinate is admitted for free by a row nobody noticed was
# already spent. That rule is preserved in cx_verdict() and in the case table
# below, and it is the pre-BSE-03 verdict.
#
# WHAT BSE-03 PHASE B CHANGED, AND WHY STALE IS GONE FROM THE VERDICT
# -------------------------------------------------------------------
# The check path no longer compares a measurement of one tree against a FILE.
# It measures TWO REVISIONS with one instrument in one job -- the comparand
# resolved by baseline_ratchet_resolve (origin/main, a ref a pull request cannot
# rewrite) and the merge commit under test -- and diffs the two measurements:
#
#   NEW      over a threshold in the merge tree, absent from the comparand -> RED
#   GROWN    a comparand function whose number rose                        -> RED
#   IMPROVED a comparand function whose number fell                        -> GREEN
#   RESOLVED a comparand function now under both thresholds                -> GREEN
#
# RESOLVED is where STALE used to be, and dropping it costs nothing HERE
# because the comparand is no longer an allowlist a human maintains: it is
# re-measured from origin/main on every run, so a repaired function leaves the
# comparand by itself on the next merge. Nobody has to notice. The property
# defended above -- "a row must not survive its own repair" -- is what makes a
# FILE comparand a ratchet; a MEASURED comparand has no rows to survive.
#
# The three failures this removes are the merge commits: git writes the merged
# baseline file, no author does, and a merge of two individually-legal branches
# carries rows neither branch wrote. A measurement of the merged TREE has no
# such artefact. scripts/complexity_baseline.txt survives as the recorded
# inventory and is still shrink-only against origin/main through
# baseline_ratchet_check (an author may not APPEND to it), but it is no longer
# an input to the verdict.
#
# Model, rows and mutation: docs/audits/threat-model-bse-03-ratchets.md;
# contract contracts/patterns/ratchet-verdict-d2-v1.yaml (APR-RATCHET-D2-001);
# polarity rows: bash scripts/tests/ratchet_semantics_test.sh --class complexity
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
#
# It is applied to BOTH revisions since phase B: a comparand measured over a
# collapsed universe reads as "everything was already broken", which is attack
# A9 re-entering at the comparand end.
#
# CX_MIN_RS_FILES exists so the throwaway fixture in
# scripts/tests/ratchet_semantics_test.sh (four functions, one file) can reach
# the verdict at all. Every run PRINTS the floor it used and says out loud when
# it is not the shipped one, because a vacuity floor nobody can see is a
# vacuity floor nobody can audit.
MIN_RS_FILES="${CX_MIN_RS_FILES:-5000}"

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

    cx_row_d2() { # cx_row_d2 <name> <want-red|want-green> <needle> <base-rows> <merge-rows>
        local name="$1" want="$2" needle="$3" out rc ok=1
        out=$(cx_verdict_d2 "$4" "$5") && rc=0 || rc=$?
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

    # ROWS 10-14. THE D2 VERDICT, over the SAME measured fixture rows. Both
    # polarities again, and row 14 is the discriminator that says the two
    # verdicts are genuinely different functions rather than one renamed: the
    # input that reds as STALE at row 8 is GREEN here, because a comparand that
    # is a MEASUREMENT has no row to keep.
    cx_row_d2 'D2: over a threshold and absent from the comparand' want-red   'NEW'      "$td/empty.txt"    "$measured"
    cx_row_d2 'D2: a comparand function that grew'                 want-red   'GROWN'    "$td/grew_cyc.txt" "$measured"
    cx_row_d2 'D2: a comparand function that fell'                 want-green 'IMPROVED' "$td/fell.txt"     "$measured"
    cx_row_d2 'D2: a comparand function now under both (was STALE→RED)' want-green 'RESOLVED' "$td/stale.txt" "$measured"
    cx_row_d2 'D2: the merge tree measures identically'            want-green ''         "$measured"        "$measured"

    if [ "$count" -lt 14 ]; then
        printf '  BROKE case table has %s row(s); at least 14 are required\n' "$count"
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
# THE D2 NORMALISER (BSE-03 phase B). measure(<rev>) over a PRISTINE
# materialisation of that revision, taken from the object store.
#
# `git archive <rev> | tar -x --wildcards '*.rs'` is the materialisation: it
# writes exactly the .rs universe of that revision and nothing else (0.7 s and
# 10265 files for HEAD of this repository on 2026-09-07, byte-identical in count
# to `git ls-files -- '*.rs'`), it cannot be contaminated by the working tree,
# and unlike `git worktree add` it leaves no administrative state behind for a
# failed run to strand. pmat needs no Cargo.toml to read a file list, so the
# extracted tree is a complete instrument input.
#
# The path handed to the resolver is a PRESENCE test, not the universe: the
# universe is measured from the tree. `src` is carried by every revision of this
# repository and by the fixture.
CX_COMPARAND_PATH='src'

cx_materialise() { # cx_materialise <root> <rev> <dest> -> the .rs tree of <rev>
    local root="$1" rev="$2" dest="$3"
    mkdir -p "$dest" || return 1
    git -C "$root" archive --format=tar "$rev" 2>/dev/null \
        | tar -x -C "$dest" --wildcards '*.rs' 2>/dev/null
}

# One measurement of one revision. Writes into <work>:
#   tree/            the materialised .rs universe
#   rows.txt         the offender rows
#   universe.txt     the scanned file list (cx_measure writes it)
#   tool_version.txt the pmat that produced rows.txt, captured INSIDE the tree
#
# The version is captured per measurement rather than once per run because the
# claim being asserted is "these two row sets came from one binary" (attack A6),
# and a single capture at the top of the script cannot witness that.
cx_measure_rev() { # cx_measure_rev <root> <rev> <work>
    local root="$1" rev="$2" work="$3"
    mkdir -p "$work" || return 1
    if ! cx_materialise "$root" "$rev" "$work/tree"; then
        printf 'FAIL: could not materialise %s from the object store; the measurement did not happen.\n' "$rev" >&2
        return 1
    fi
    ( cd "$work/tree" && pmat --version 2>/dev/null | head -1 ) > "$work/tool_version.txt"
    cx_measure "$work/tree" "$work" > "$work/rows.txt" 2> "$work/measure.err"
}

# The recorded instrument, BSE-10a's form: a `# tool_version=<...>` header line
# in the baseline file. Absent is NOT a pass and NOT a mismatch; the caller says
# which.
cx_recorded_tool_version() { # cx_recorded_tool_version <file>
    sed -n 's/^#[[:space:]]*tool_version=[[:space:]]*//p' "$1" 2>/dev/null | head -1
}

# THE D2 VERDICT. Two MEASUREMENTS in, findings out; rc 1 iff a RED finding.
# Improvements are printed as NOTE lines and are never a failure -- there is no
# lower bound here, which is exactly what distinguishes this from cx_verdict.
cx_verdict_d2() { # cx_verdict_d2 <base-rows> <merge-rows>
    local out red
    out=$( { cx_data "$1" | sed 's/^/B /'
             cx_data "$2" | sed 's/^/C /'; } | LC_ALL=C awk '
        $1 == "B" { bcyc[$2] = $3; bcog[$2] = $4; base[$2] = 1; next }
        $1 == "C" {
            cur[$2] = 1
            if (!($2 in base)) {
                printf "RED    NEW      %s  cyclomatic %s cognitive %s  (over a threshold, absent from the comparand)\n", $2, $3, $4
                next
            }
            if ($3+0 > bcyc[$2]+0) { printf "RED    GROWN    %s  cyclomatic %s -> %s\n", $2, bcyc[$2], $3 }
            if ($4+0 > bcog[$2]+0) { printf "RED    GROWN    %s  cognitive %s -> %s\n", $2, bcog[$2], $4 }
            if ($3+0 < bcyc[$2]+0) { printf "NOTE   IMPROVED %s  cyclomatic %s -> %s\n", $2, bcyc[$2], $3 }
            if ($4+0 < bcog[$2]+0) { printf "NOTE   IMPROVED %s  cognitive %s -> %s\n", $2, bcog[$2], $4 }
        }
        END {
            for (k in base) {
                if (!(k in cur)) {
                    printf "NOTE   RESOLVED %s  under both thresholds in the merge tree; the comparand is a MEASUREMENT, so there is no row to delete\n", k
                }
            }
        }
    ' | LC_ALL=C sort)
    if [ -n "$out" ]; then
        printf '%s\n' "$out"
    fi
    red=$(printf '%s\n' "$out" | grep -c '^RED ' || true)
    [ "$red" -eq 0 ]
}

# ---------------------------------------------------------------------------

# ── CB-200 (pmat comply, the TDG grade gate) ─────────────────────────────────
# pmat 3.36.0 reads `.pmat-gates.toml [tdg] baseline` and reports CB-200 as
# Warn ("debt held flat") while the count of functions below min_grade is at
# or under it, Fail above it. Nothing in pmat compares that number with the
# one on origin/main, so a branch could raise it and merge (PMAT-937 review).
# The number is therefore MIRRORED in scripts/cb200_baseline.txt, the two must
# agree, and the file is shrink-only against origin/main through the same
# baseline_ratchet_check every other baseline in this tree goes through.
CB200_REL='scripts/cb200_baseline.txt'

cb200_toml_value() { # <gates.toml> -> the [tdg] baseline integer, or ""
    python3 - "$1" <<'PY'
import re, sys
s = open(sys.argv[1], encoding="utf-8").read()
sec = s.split("[tdg]", 1)[1].split("\n[", 1)[0] if "[tdg]" in s else ""
m = re.search(r"^\s*baseline\s*=\s*(\d+)", sec, re.M)
print(m.group(1) if m else "")
PY
}

cb200_pair_check() { # <gates.toml> <mirror file> -> 0 when they agree, 1 otherwise (prints why)
    local toml file
    toml=$(cb200_toml_value "$1")
    if [ -z "$toml" ]; then
        printf 'FAIL: %s carries no [tdg] baseline; pmat comply would report CB-200 as Fail, not as a held ratchet.\n' "$1"
        return 1
    fi
    if [ ! -f "$2" ]; then
        printf 'FAIL: %s missing; it mirrors [tdg] baseline so the CB-200 count is ratcheted, not typed.\n' "$2"
        return 1
    fi
    file=$(grep -vE '^[[:space:]]*(#|$)' "$2" | tr -d '[:space:]')
    if [ "$toml" != "$file" ]; then
        printf 'FAIL: [tdg] baseline = %s but %s says %s; the two move together, and only the file is ratcheted against origin/main.\n' "$toml" "$2" "$file"
        return 1
    fi
    printf 'ok    CB-200 baseline %s ([tdg] baseline == %s)\n' "$toml" "$2"
    return 0
}

cb200_selftest() { # both polarities of the pair check, on throwaway files; returns the broke count
    local td pass=0 broke=0
    td=$(mktemp -d) || return 1
    printf '[tdg]\nbaseline = 609\n' > "$td/gates.toml"
    printf '609\n' > "$td/mirror.txt"
    if cb200_pair_check "$td/gates.toml" "$td/mirror.txt" > /dev/null 2>&1; then
        printf '  ok    %-38s %s\n' cb200_baseline_equal 'toml == mirror'; pass=$((pass + 1))
    else
        printf '  BROKE %-38s %s\n' cb200_baseline_equal 'equal values were refused'; broke=$((broke + 1))
    fi
    printf '[tdg]\nbaseline = 610\n' > "$td/gates.toml"
    if cb200_pair_check "$td/gates.toml" "$td/mirror.txt" > /dev/null 2>&1; then
        printf '  BROKE %-38s %s\n' cb200_baseline_raised 'a raised toml value passed'; broke=$((broke + 1))
    else
        printf '  ok    %-38s %s\n' cb200_baseline_raised 'toml 610 vs mirror 609 refused'; pass=$((pass + 1))
    fi
    printf '[tdg]\nmin_grade = "B"\n' > "$td/gates.toml"
    if cb200_pair_check "$td/gates.toml" "$td/mirror.txt" > /dev/null 2>&1; then
        printf '  BROKE %-38s %s\n' cb200_baseline_absent 'a toml with no baseline passed'; broke=$((broke + 1))
    else
        printf '  ok    %-38s %s\n' cb200_baseline_absent 'no [tdg] baseline refused'; pass=$((pass + 1))
    fi
    rm -rf "${td:?}"
    printf '  CB-200: %s passed, %s broken\n' "$pass" "$broke"
    return "$broke"
}

if [ "${1:-}" = '--selftest' ] || [ "${1:-}" = '--self-test' ]; then
    cx_selftest; cx_rc=$?
    cb200_selftest; cb_rc=$?
    [ "$cx_rc" -eq 0 ] && [ "$cb_rc" -eq 0 ] && exit 0
    exit 1
fi

printf '=== per-function complexity may only fall (check_complexity_ratchet.sh) ===\n'

if ! command -v pmat > /dev/null 2>&1; then
    printf 'ENV: pmat is not on PATH; the fleet pin installs it (tools.toml; CI never installs tools).\n' >&2
    printf 'Without the analyser this guard cannot decide, so it refuses to pass (exit 2, never 0).\n' >&2
    exit 2
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

# ---------------------------------------------------------------------------
# --update KEEPS THE WORKING TREE AS ITS SUBJECT. It is a recorder, not a
# verdict: it writes down what the tree you are sitting on measures, and it is
# the only path that may write the baseline file. It also records the
# INSTRUMENT (BSE-10a's `tool_version=` header), because the check path asserts
# against it and a header nothing writes is a control nothing can hold.
if [ "${1:-}" = '--update' ]; then
    if ! cx_measure "$REPO_ROOT" "$WORK" > "$WORK/current.txt" 2> "$WORK/measure.err"; then
        sed 's/^/      | /' "$WORK/measure.err" >&2
        printf 'FAIL: the complexity scan did not complete, so nothing was recorded.\n' >&2
        exit 1
    fi
    sed 's/^/  /' "$WORK/measure.err"
    SCANNED=$(grep -c . "$WORK/universe.txt" || true)
    CURRENT=$(grep -c . "$WORK/current.txt" || true)
    if [ "$SCANNED" -lt "$MIN_RS_FILES" ]; then
        printf 'FAIL (vacuity): only %s .rs file(s) found, expected %s+. Nothing was written.\n' \
            "$SCANNED" "$MIN_RS_FILES" >&2
        exit 1
    fi
    {
        printf '# complexity_baseline.txt - functions over the pre-commit hook thresholds\n'
        printf '# (cyclomatic > %s or cognitive > %s), as "<path>::<function> <cyclomatic> <cognitive>".\n' \
            "$MAX_CYCLOMATIC" "$MAX_COGNITIVE"
        printf '# SHRINK-ONLY. Regenerate with: bash scripts/check_complexity_ratchet.sh --update\n'
        printf '# Owner: scripts/check_complexity_ratchet.sh (PMAT-746).\n'
        printf '# tool_version=%s\n' "$(pmat --version 2>/dev/null | head -1)"
        cat "$WORK/current.txt"
    } > "$BASELINE"
    printf 'baseline rewritten: %s row(s), tool_version=%s\n' \
        "$CURRENT" "$(pmat --version 2>/dev/null | head -1)"
    exit 0
fi

# ---------------------------------------------------------------------------
# PREFLIGHT. The comparand is resolved and PRINTED before a single file is
# measured (threat model P5): an unresolvable comparand invalidates the
# measurement, so paying for the measurement first only buys a more expensive
# way to say the same thing.
#
# shellcheck source=scripts/lib_baseline_ratchet.sh
. "${REPO_ROOT}/scripts/lib_baseline_ratchet.sh" || exit 1

RESOLUTION=$(baseline_ratchet_resolve "$REPO_ROOT" "$BASELINE_RATCHET_BASE_REF" "$CX_COMPARAND_PATH")
MODE=${RESOLUTION%%$'\t'*}
REF=${RESOLUTION##*$'\t'}
case "$MODE" in
    UNRESOLVABLE)
        printf 'FAIL PREFLIGHT: cannot resolve the comparand ref <%s>, so complexity is UNMEASURED against anything. That is not "no growth", and it is not degraded to comparing this branch against itself.\n' "$REF" >&2
        printf '     In CI, before this guard runs:  git fetch --no-tags --depth=1 origin +refs/heads/main:refs/remotes/origin/main\n' >&2
        exit 1 ;;
    ABSENT | BOOTSTRAP)
        printf 'FAIL PREFLIGHT: <%s> carries no %s/ tree, so there is no comparand to measure. A missing comparand is not "no growth".\n' "$REF" "$CX_COMPARAND_PATH" >&2
        exit 1 ;;
esac
BASE_SHA=$(git -C "$REPO_ROOT" rev-parse --verify --quiet "${REF}^{commit}") || BASE_SHA=""
MERGE_SHA=$(git -C "$REPO_ROOT" rev-parse --verify --quiet 'HEAD^{commit}') || MERGE_SHA=""
if [ -z "$BASE_SHA" ] || [ -z "$MERGE_SHA" ]; then
    printf 'FAIL PREFLIGHT: <%s> and HEAD did not both resolve to a commit (base=%q merge=%q).\n' \
        "$REF" "$BASE_SHA" "$MERGE_SHA" >&2
    exit 1
fi
BASE_SHORT=$(git -C "$REPO_ROOT" rev-parse --short "$BASE_SHA" 2>/dev/null) || BASE_SHORT="$BASE_SHA"
MERGE_SHORT=$(git -C "$REPO_ROOT" rev-parse --short "$MERGE_SHA" 2>/dev/null) || MERGE_SHORT="$MERGE_SHA"
BASE_DATE=$(git -C "$REPO_ROOT" show -s --format=%cI "$BASE_SHA" 2>/dev/null) || BASE_DATE='<no date>'

printf 'complexity — D2 comparand diff (BSE-03 phase B): the verdict is a function of two REVISIONS and the instrument, of nothing else on disk\n'
printf '  comparand   %-12s %s  %s\n' "$MODE" "$BASE_SHORT" "$BASE_DATE"
printf '  merge       %-12s %s  (HEAD)\n' 'HEAD' "$MERGE_SHORT"
printf '  floor       %s .rs file(s), applied to BOTH revisions\n' "$MIN_RS_FILES"
if [ "$BASELINE_RATCHET_BASE_REF" != 'origin/main' ]; then
    printf '  OVERRIDDEN  comparand set via BASELINE_RATCHET_BASE_REF=%s — NOT a protected ref\n' \
        "$BASELINE_RATCHET_BASE_REF"
fi
if [ "${CX_MIN_RS_FILES:-}" != '' ]; then
    printf '  OVERRIDDEN  vacuity floor set via CX_MIN_RS_FILES=%s — this is the fixture seam, NOT the shipped floor of 5000\n' \
        "$CX_MIN_RS_FILES"
fi
DIRTY=$(git -C "$REPO_ROOT" status --porcelain -- '*.rs' | grep -c . || true)
if [ "$DIRTY" -ne 0 ]; then
    printf '  UNCOMMITTED %s .rs path(s) differ between the working tree and %s. The verdict below is over the COMMIT, not over your edits; CI checks out pristine, where the two are equal by construction.\n' \
        "$DIRTY" "$MERGE_SHORT"
fi

# ---------------------------------------------------------------------------
# MEASUREMENT: ONE INSTRUMENT, TWO REVISIONS, in this job.
if ! cx_measure_rev "$REPO_ROOT" "$BASE_SHA" "$WORK/base"; then
    sed 's/^/      | /' "$WORK/base/measure.err" >&2 2>/dev/null || true
    printf 'FAIL: the comparand %s could not be measured, so growth is UNMEASURED. That is a broken check, not a clean tree.\n' "$BASE_SHORT" >&2
    exit 1
fi
if ! cx_measure_rev "$REPO_ROOT" "$MERGE_SHA" "$WORK/merge"; then
    sed 's/^/      | /' "$WORK/merge/measure.err" >&2 2>/dev/null || true
    printf 'FAIL: the merge tree %s could not be measured, so growth is UNMEASURED.\n' "$MERGE_SHORT" >&2
    exit 1
fi

BASE_FILES=$(grep -c . "$WORK/base/universe.txt" || true)
MERGE_FILES=$(grep -c . "$WORK/merge/universe.txt" || true)
BASE_ROWS=$(grep -c . "$WORK/base/rows.txt" || true)
MERGE_ROWS=$(grep -c . "$WORK/merge/rows.txt" || true)
printf '  measured    base  %s .rs file(s), %s function(s) over cyclomatic>%s or cognitive>%s\n' \
    "$BASE_FILES" "$BASE_ROWS" "$MAX_CYCLOMATIC" "$MAX_COGNITIVE"
printf '  measured    merge %s .rs file(s), %s function(s) over cyclomatic>%s or cognitive>%s\n' \
    "$MERGE_FILES" "$MERGE_ROWS" "$MAX_CYCLOMATIC" "$MAX_COGNITIVE"
printf '  universe    base=%s merge=%s delta=%s  (A8: an exclusion added by the PR shrinks one side only, and that is a finding, never an improvement)\n' \
    "$BASE_FILES" "$MERGE_FILES" "$(printf '%+d' "$((MERGE_FILES - BASE_FILES))")"
printf '  polarity    the verdict is about the DIFF of two measurements: a number that ROSE is a regression (RED); a number that FELL, or a function that left the set entirely, is an IMPROVEMENT (GREEN). There is NO lower bound and nothing to delete.\n'

if [ "$BASE_FILES" -lt "$MIN_RS_FILES" ] || [ "$MERGE_FILES" -lt "$MIN_RS_FILES" ]; then
    printf '\nFAIL (vacuity): base scanned %s .rs file(s) and merge scanned %s, expected %s+ at BOTH revisions.\n' \
        "$BASE_FILES" "$MERGE_FILES" "$MIN_RS_FILES"
    printf 'The scan is broken, not the code. Fix it rather than this number.\n'
    exit 1
fi

# THE INSTRUMENT IS ASSERTED, NOT LOGGED (threat model A6). Two measurements
# made by two different pmats are not a diff: a binary that scores fewer
# functions makes any tree look improved, and printing both versions while
# comparing their output anyway is a guard reporting a result it did not
# measure.
BASE_TOOL=$(cat "$WORK/base/tool_version.txt" 2>/dev/null || true)
MERGE_TOOL=$(cat "$WORK/merge/tool_version.txt" 2>/dev/null || true)
RECORDED_TOOL=$(cx_recorded_tool_version "$BASELINE")
printf '  instrument  base=<%s> merge=<%s> recorded=<%s>\n' "$BASE_TOOL" "$MERGE_TOOL" "$RECORDED_TOOL"
if [ -z "$BASE_TOOL" ] || [ -z "$MERGE_TOOL" ]; then
    printf '\nFAIL (instrument): pmat did not name itself at one of the two revisions (base=%q merge=%q). An unnamed instrument cannot be asserted equal to anything.\n' \
        "$BASE_TOOL" "$MERGE_TOOL"
    exit 1
fi
if [ "$BASE_TOOL" != "$MERGE_TOOL" ]; then
    printf '\nFAIL (instrument): the two measurements were produced by DIFFERENT pmat versions, so their difference is not a complexity diff.\n'
    printf '  base  %s  (%s)\n' "$BASE_TOOL" "$BASE_SHORT"
    printf '  merge %s  (%s)\n' "$MERGE_TOOL" "$MERGE_SHORT"
    exit 1
fi
if [ -n "$RECORDED_TOOL" ] && [ "$RECORDED_TOOL" != "$MERGE_TOOL" ]; then
    printf '\nFAIL (instrument): the recorded tool_version does not match the pmat that ran.\n'
    printf '  recorded %s  (%s tool_version= header)\n' "$RECORDED_TOOL" "$BASELINE_REL"
    printf '  ran      %s  (both revisions)\n' "$MERGE_TOOL"
    printf '  Re-record it in the same commit that moves the toolchain: bash scripts/check_complexity_ratchet.sh --update\n'
    exit 1
fi
if [ -z "$RECORDED_TOOL" ]; then
    printf '  NOT RECORDED %s carries no `# tool_version=` header, so the recorded-instrument leg is UNASSERTED (the two-measurement leg above still holds). `--update` writes it.\n' \
        "$BASELINE_REL"
fi

if [ ! -f "$BASELINE" ]; then
    printf 'FAIL: %s missing. Run --update once to establish it.\n' "$BASELINE" >&2
    exit 1
fi
RECORDED=$(cx_data "$BASELINE" | grep -c . || true)
printf '  inventory   %s recorded row(s) in %s — shrink-only against origin/main, and NOT an input to the verdict\n' \
    "$RECORDED" "$BASELINE_REL"

# ---------------------------------------------------------------------------
# THE VERDICT, over the two measurements.
VERDICT_RC=0
FINDINGS=$(cx_verdict_d2 "$WORK/base/rows.txt" "$WORK/merge/rows.txt") || VERDICT_RC=$?   # RATCHET-MUTATION-POINT — the comparand is the MEASUREMENT of base_sha; the registered mutation of scripts/tests/ratchet_semantics_test.sh --class complexity replaces exactly this line with the pre-BSE-03 `cx_verdict "$BASELINE" ...`, which reads the baseline FILE and reintroduces the STALE lower bound
if [ -n "$FINDINGS" ]; then
    printf '%s\n' "$FINDINGS"
fi

# ---------------------------------------------------------------------------
# THE FILE-LEVEL RATCHETS. Unchanged, and still needed for a reason the verdict
# above no longer covers: scripts/complexity_baseline.txt and
# scripts/cb200_baseline.txt are still READ by humans and by pmat comply, so an
# author appending to either must still be refused against a ref they cannot
# rewrite. What changed is that neither file is an input to the complexity
# verdict any more.
RATCHET_RC=0
baseline_ratchet_check "$REPO_ROOT" "$BASELINE_REL" keyed2 || RATCHET_RC=$?
cb200_pair_check "$REPO_ROOT/.pmat-gates.toml" "$REPO_ROOT/$CB200_REL" || RATCHET_RC=1
baseline_ratchet_check "$REPO_ROOT" "$CB200_REL" count || RATCHET_RC=$?

if [ "$VERDICT_RC" -ne 0 ]; then
    printf '\nFAIL: complexity regressed against %s.\n' "$BASE_SHORT"
    printf '  NEW    the function is over a threshold in %s and was not over one in\n' "$MERGE_SHORT"
    printf '         %s. Split it, or reduce it below cyclomatic %s / cognitive %s.\n' \
        "$BASE_SHORT" "$MAX_CYCLOMATIC" "$MAX_COGNITIVE"
    printf '  GROWN  the comparand measurement is the ceiling. It may fall, never rise.\n'
    printf '  There is nothing to edit in %s to make this pass: the comparand is\n' "$BASELINE_REL"
    printf '  measured from %s, not read from the tree you can write to.\n' "$BASELINE_RATCHET_BASE_REF"
fi

if [ "$RATCHET_RC" -ne 0 ] || [ "$VERDICT_RC" -ne 0 ]; then
    exit 1
fi

printf 'PASS (D2): %s vs %s measured by %s — none new, none grown.\n' \
    "$BASE_SHORT" "$MERGE_SHORT" "$MERGE_TOOL"
exit 0
