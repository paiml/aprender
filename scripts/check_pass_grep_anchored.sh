#!/usr/bin/env bash
# check_pass_grep_anchored.sh - every "zero-count" pass-grep must actually turn RED.
#
# PMAT-CI-PASSGREP-001 closed ONE instance of this defect (ci.yml's
# `grep -q "test result.*0 failed"`, which matched "10 failed" and merged a run
# with 10 failures GREEN). The pattern is a CLASS, not an incident: two more
# live instances survived that fix - `contracts/qwen-story-v1.yaml`'s
# FALSIFY-QWEN-STORY-007 (`grep -qE '0 error'`, satisfied by "10 error(s)") and
# `scripts/dogfood-book.sh` - so a point fix is not enough. This is the ratchet.
#
# METHOD: empirical, not syntactic. Regex-linting regexes is brittle; instead we
# extract each pass-grep pattern from the tree and RUN it against a synthetic
# line in which every count is non-zero. A pass-detector that matches an
# all-failing line fails open, by definition. No heuristic about anchors,
# word boundaries, or leading spaces is required - the probe settles it.
#
# Exit 0 = every pass-grep correctly rejects the failing probe.
# Exit 1 = at least one pass-grep fails open (prints file:line and the pattern).
#
# ---------------------------------------------------------------------------
# 2026-08-31: THREE BLIND SPOTS, ONE MEASURED ESCAPE, AND A SELF-TEST THAT WAS
# PASSING FOR THE WRONG REASON.
#
# (1) THE EXTRACTOR DRAINED ITS OWN INPUT, so NO DOUBLE-QUOTED GREP PATTERN WAS
#     EVER CHECKED -- including `grep -q "test result.*0 failed"`, the exact
#     pattern PMAT-CI-PASSGREP-001 is named for and the one this file's own
#     self-test claims to prove it catches. `extract_patterns` ran two seds,
#     both reading /dev/stdin: the FIRST (the single-quote form) matched
#     nothing on a double-quoted line but CONSUMED the whole stream, so the
#     second read an empty file. Proven, not remembered:
#
#       printf 'grep -q "a 0 failed b"\n' | { sed -nE "...'...'..." /dev/stdin
#                                              sed -nE '..."..."...' /dev/stdin; }
#       -> both seds print nothing
#
#     AND THE SELF-TEST DID NOT NOTICE, because it asserted only that the
#     checker exits non-zero. It did -- via `ERROR: only 0 pass-grep(s) found`,
#     the VACUITY floor. A gate whose proof-of-RED is satisfied by its own
#     blindness is the failure this file exists to name, one level up. The
#     self-test now asserts the WORD `FAIL-OPEN`, not the exit status.
#
# (2) THE UNIVERSE EXCLUDED 61 OF 208 TRACKED .sh FILES. SEARCH_PATHS was
#     (.github/workflows contracts scripts Makefile); 59 shell scripts under
#     crates/, one under evidence/ and one at the root were invisible. That is
#     the `git ls-files`-style free-pass shape one directory up.
#
# (3) THE MEASURED ESCAPE IS NOT A PASS-GREP AT ALL -- IT IS AN INVERTED EXIT
#     STATUS. evidence/perf-061/ramp.sh (#2800) ends:
#
#       echo "=== CB-006-OUT lines (must be 0) ==="
#       cat "$OUT/apr.stdout" "$OUT/apr.stderr" | grep -c "CB-006-OUT"
#
#     grep is the LAST command, so the script EXPORTS grep's status: it exits
#     1 when the log is CLEAN and 0 when it is CONTAMINATED, exactly backwards
#     from the banner one line above. KEYWORDS never matches `grep -c
#     "CB-006-OUT"`, so no pattern-based rule could reach it. The second rule
#     below is about the SHAPE of the last command instead, and it keeps this
#     file's method: the polarity is PROVED by running the real grep against a
#     clean and a dirty input, not asserted from the text.
#
#     MEASURED FALSE-POSITIVE RATE: across all 209 .sh files on origin/main the
#     rule fires ZERO times, and it fires on ramp.sh. A script's exit status is
#     its contract with its caller; grep's status answers a different question
#     ("did the pattern occur"), and under either polarity one of the two
#     outcomes is mislabelled. The remedy is one line -- `exit 0`, or an
#     explicit `[ "$n" -eq 0 ] || exit 1`.

set -euo pipefail

SELF_PATH="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

cd "$(dirname "$0")/.." || exit 1

# --self-test: prove this checker can turn RED, and prove it turns red FOR THE
# REASON CLAIMED. A gate nobody has ever seen fail is indistinguishable from a
# gate that cannot fail -- and a self-test that asserts only "exit != 0" is
# indistinguishable from one that asserts nothing.
#
# THIS SELF-TEST USED TO PASS FOR THE WRONG REASON, AND THAT IS WHY IT NOW
# GREPS THE VERDICT. It ran the checker over a fixture containing the
# pre-PMAT-CI-PASSGREP-001 pattern and asserted a non-zero exit. It got one --
# from `ERROR: only 0 pass-grep(s) found`, the VACUITY floor, because the
# extractor drained its own stdin and could not read a DOUBLE-QUOTED pattern at
# all (header, (1)). The checker had therefore never once demonstrated a
# FAIL-OPEN detection, while printing that it had. Every row below now asserts
# the WORD the verdict must contain, and one row asserts a clean fixture stays
# GREEN so "always red" cannot pass either.
if [ "${1:-}" = "--self-test" ]; then
    TMP=$(mktemp -d)
    cleanup_selftest() {
        if [ -n "$TMP" ] && [ "$TMP" != / ]; then
            rm -rf "$TMP"
        fi
    }
    trap cleanup_selftest EXIT
    st_fail=0
    st_rows=0

    # The run is INLINE, not inside `out=$(run_fixture ...)`: a command
    # substitution is a SUBSHELL, so an rc captured in a helper never reaches
    # the caller. Caught by this table's own first execution.
    st_row() { # st_row <name> <want-red|want-green> <must-contain> <dir>
        local name="$1" want="$2" needle="$3" dir="$4" out rc ok=1
        out=$( cd "$dir" && MIN_EXPECTED=1 SH_MIN_EXPECTED=1 \
               bash scripts/check_pass_grep_anchored.sh 2>&1 ) && rc=0 || rc=$?
        st_rows=$((st_rows + 1))
        case "$want" in
            want-red)   [ "$rc" -ne 0 ] || ok=0 ;;
            want-green) [ "$rc" -eq 0 ] || ok=0 ;;
        esac
        if [ -n "$needle" ] && ! grep -qF -- "$needle" <<< "$out"; then ok=0; fi
        if [ "$ok" -eq 1 ]; then
            printf '  ok    %-10s %s\n' "$want" "$name"
        else
            printf '  FAIL  %-10s %s (rc=%s)\n' "$want" "$name" "$rc"
            printf '%s\n' "$out" | sed 's/^/          | /'
            st_fail=$((st_fail + 1))
        fi
    }
    mk_fixture() { # mk_fixture <dir>
        mkdir -p "$1/scripts" "$1/contracts" "$1/.github/workflows"
        touch "$1/Makefile"
        cp "$SELF_PATH" "$1/scripts/"
    }

    printf -- '--- case table -----------------------------------------------------\n'

    # ROW 1. The pre-PMAT-CI-PASSGREP-001 ci.yml line, VERBATIM. It is DOUBLE
    # QUOTED, which is exactly the form the old extractor could not read, and
    # the verdict must name FAIL-OPEN rather than vacuity.
    mk_fixture "$TMP/r1"
    printf '%s\n' 'grep -q "test result.*0 failed" out.log' > "$TMP/r1/scripts/historical_pattern.sh"
    st_row 'PMAT-CI-PASSGREP-001, double-quoted' want-red 'FAIL-OPEN' "$TMP/r1"

    # ROW 2. The same defect single-quoted -- the form that DID work before, so
    # a regression in the other sed is visible too.
    mk_fixture "$TMP/r2"
    # shellcheck disable=SC2016
    printf '%s\n' "grep -qE 'test result.*0 failed' out.log" > "$TMP/r2/scripts/historical_pattern.sh"
    st_row 'the same pattern single-quoted'    want-red 'FAIL-OPEN' "$TMP/r2"

    # ROW 3. THE DIALECT. `grep -q "0 error(s)"` is BRE, so the parens are
    # literal and "10 error(s)" contains it. Probed as -E it looks safe. This
    # row is the live hook the widened universe found.
    mk_fixture "$TMP/r3"
    printf '%s\n' 'bashrs lint "$f" | grep -q "0 error(s)"' > "$TMP/r3/scripts/bre_pattern.sh"
    st_row 'BRE parens, probed in its own dialect' want-red 'FAIL-OPEN' "$TMP/r3"

    # ROW 4. THE INVERTED EXIT STATUS -- evidence/perf-061/ramp.sh:36, verbatim.
    # No keyword appears anywhere in it, so rule 1 is blind by construction.
    mk_fixture "$TMP/r4"
    { printf '#!/usr/bin/env bash\n'
      printf 'echo "=== CB-006-OUT lines (must be 0) ==="\n'
      printf '%s\n' 'cat "$OUT/apr.stdout" "$OUT/apr.stderr" | grep -c "CB-006-OUT"'
    } > "$TMP/r4/scripts/ramp.sh"
    printf '%s\n' "grep -qE '(^|[^0-9])0 error' log" > "$TMP/r4/scripts/anchored.sh"
    st_row 'final command is a bare grep -c'  want-red 'INVERTED-EXIT' "$TMP/r4"

    # ROW 5. THE GREEN CONTROL, and it is the half that says the table
    # discriminates. An anchored pass-grep, and a script that ends by SAYING
    # what success means. A checker that reds everything passes rows 1-4.
    mk_fixture "$TMP/r5"
    printf '%s\n' "grep -qE '(^|[^0-9])0 error' log" > "$TMP/r5/scripts/anchored.sh"
    { printf '#!/usr/bin/env bash\n'
      printf '%s\n' 'n=$(grep -c "CB-006-OUT" log); [ "$n" -eq 0 ] || exit 1'
      printf 'exit 0\n'
    } > "$TMP/r5/scripts/decided.sh"
    st_row 'anchored pattern + decided exit'  want-green 'OK:' "$TMP/r5"

    # ROW 6. VACUITY MUST NOT LOOK LIKE A CATCH. An empty fixture exits
    # non-zero, and the reason must be the floor, NOT a finding -- this is the
    # exact confusion that let the old self-test pass while blind.
    mk_fixture "$TMP/r6"
    st_row 'empty tree reds as VACUOUS, not as a finding' want-red 'ERROR: only' "$TMP/r6"

    if [ "$st_rows" -lt 6 ]; then
        printf '  FAIL  case table has %s row(s); at least 6 are required\n' "$st_rows"
        st_fail=$((st_fail + 1))
    fi
    printf '  %s row(s), %s failure(s)\n' "$st_rows" "$st_fail"
    if [ "$st_fail" -ne 0 ]; then
        printf 'SELF-TEST FAILED\n' >&2
        exit 1
    fi
    printf 'self-test OK: FAIL-OPEN, dialect, INVERTED-EXIT and the green control all assert.\n'
    exit 0
fi

# A line in which EVERY count is non-zero. Any pattern claiming to detect
# "zero errors / zero failures / zero warnings" must NOT match this.
PROBE='Summary: 10 error(s), 20 warning(s), 30 info(s) :: test result: FAILED. 90 passed; 10 failed; 40 ignored'

# Zero-count pass-detector keywords. A grep pattern mentioning one of these is
# asserting "none of these happened" and is therefore in scope.
KEYWORDS='0 error|0 fail|0 warn|0 issue|0 violation|0 problem'

# THE UNIVERSE. `crates` and `evidence` are here because 61 of 208 tracked .sh
# files lived outside the old four -- 59 under crates/, one under evidence/,
# one at the root -- and the measured escape was the one under evidence/. Root
# level *.sh is expanded rather than named, so a new one arrives in scope
# instead of arriving invisible. `target` is pruned: a build directory is not a
# gate, and a vendored copy of someone else's script is not this repo's defect.
SEARCH_PATHS=(.github/workflows contracts scripts Makefile crates evidence)
shopt -s nullglob
SEARCH_PATHS+=(./*.sh)
shopt -u nullglob
GREP_PRUNE=(--exclude-dir=target --exclude-dir=target_disk --exclude-dir=.git \
            --exclude-dir=node_modules)

# Known zero-count pass-greps in the tree at the time of writing. If the
# extractor finds fewer than this, it has gone blind - see the tail of this
# script.
# 3 on origin/main as of 2026-08-31, and it was 2 before the extractor and the
# universe were repaired -- one pattern (`0 error(s)` in
# crates/aprender-distribute/.githooks/pre-commit) was invisible on BOTH counts.
MIN_EXPECTED="${MIN_EXPECTED:-3}"

VIOLATIONS=0
CHECKED=0

# Pull the quoted pattern out of a grep invocation. Handles the two forms used
# in this tree: grep -q'X' with single quotes and with double quotes. Emits one
# pattern per line.
# TAKES THE LINE AS AN ARGUMENT, NOT A FILE, AND THAT IS THE WHOLE FIX. It used
# to be called as `extract_patterns /dev/stdin` from a pipe, and ran two seds
# against that same path. A stream can only be read once: the first sed drained
# it, the second got EOF, and every double-quoted grep pattern in the
# repository was silently invisible -- see (1) in the header. Herestrings feed
# each sed its own copy.
extract_patterns() {
    # Two passes rather than one, so neither sed script needs escaped quotes:
    # the single-quote form is written inside double quotes and vice versa.
    # Each emits `<flags><TAB><pattern>`; see probe_flags_for below.
    sed -nE "s/.*grep[[:space:]]+((-[A-Za-z]+[[:space:]]+)*)'([^']*)'.*/\1	\3/p" <<< "$1"
    # shellcheck disable=SC2016
    sed -nE 's/.*grep[[:space:]]+((-[A-Za-z]+[[:space:]]+)*)"([^"]*)".*/\1	\3/p' <<< "$1"
}

# THE PROBE MUST SPEAK THE DIALECT THE INVOCATION SPEAKS, or it answers a
# question nobody asked. This checker probed every pattern with `grep -qE`
# while most real gates run plain `grep -q`, which is BRE -- and in BRE `(` and
# `)` are LITERAL. crates/aprender-distribute/.githooks/pre-commit runs
#
#     bashrs lint "$file" 2>&1 | grep -q "0 error(s)"
#
# and `0 error(s)` as BRE is a literal substring of `10 error(s)`, so that hook
# passes a file with ten errors. Under -E the same text is `0 errors` followed
# by nothing of the sort, no match, and this checker called it SAFE. One
# dialect away from the decision is not the decision.
#
# Only the flags that change MATCHING are carried: the matcher selector
# (-E/-F/-P, default BRE), plus -i, -w, -x. -q/-c/-n/-r do not affect whether
# the pattern matches.
probe_flags_for() { # probe_flags_for <flag-string> -> flags, one per line
    case "$1" in *E*) printf -- '-E\n' ;; *F*) printf -- '-F\n' ;; *P*) printf -- '-P\n' ;; esac
    case "$1" in *i*) printf -- '-i\n' ;; esac
    case "$1" in *w*) printf -- '-w\n' ;; esac
    case "$1" in *x*) printf -- '-x\n' ;; esac
}

SELF="scripts/check_pass_grep_anchored.sh"

while IFS= read -r hit; do
    file="${hit%%:*}"
    rest="${hit#*:}"
    lineno="${rest%%:*}"

    # This checker's own header quotes the historical bad patterns verbatim as
    # documentation; they are prose, not gates.
    [ "$file" = "$SELF" ] && continue

    # Re-read just this line and pull its grep pattern(s) out.
    line_text=$(sed -n "${lineno}p" "$file")

    trimmed=$(printf '%s' "$line_text" | sed 's/^[[:space:]]*//')

    # A commented-out grep is not a gate.
    case "$trimmed" in
        '#'*) continue ;;
        *) ;;
    esac

    # In a contract, only the `test:` field is executed by `pv`. Everything
    # else - five_whys, references, if_fails - is prose, and prose routinely
    # quotes the very bad patterns this checker exists to describe. Scanning it
    # would make the gate fire on its own documentation.
    case "$file" in
        contracts/*)
            case "$trimmed" in
                test:*) ;;
                *) continue ;;
            esac
            ;;
        *) ;;
    esac
    patterns=$(extract_patterns "$line_text" || true)

    [ -z "$patterns" ] && continue

    while IFS= read -r rec; do
        [ -z "$rec" ] && continue
        flagstr="${rec%%$'\t'*}"
        pat="${rec#*$'\t'}"
        [ -z "$pat" ] && continue
        # Only patterns that actually assert a zero count are in scope.
        grep -qE "$KEYWORDS" <<< "$pat" || continue

        CHECKED=$((CHECKED + 1))

        pflags=()
        while IFS= read -r pf; do [ -n "$pf" ] && pflags+=("$pf"); done < <(probe_flags_for "$flagstr")

        # THE PROBE. If a "zero errors" detector matches an all-failing line,
        # it can never turn red on that failure mode. Run in the invocation's
        # own dialect -- see probe_flags_for. The here-string is #2804's, and
        # it is not cosmetic: `printf ... | grep -q` hands the producer EPIPE
        # when grep exits at its first match, and under pipefail that reports a
        # MATCH as a non-match -- a fail-open detector that fails open.
        if grep -q "${pflags[@]}" -- "$pat" <<< "$PROBE" 2>/dev/null; then
            VIOLATIONS=$((VIOLATIONS + 1))
            printf 'FAIL-OPEN %s:%s\n' "$file" "$lineno"
            printf '          pattern: %s   (probed as: grep %s)\n' "$pat" "${pflags[*]:-BRE}"
            printf '          matches: %s\n' "$PROBE"
            printf '          fix: anchor the count, e.g. (^|[^0-9])0 error\n'
        fi
    done <<EOF
$patterns
EOF
done <<EOF
$(grep -rnE "${GREP_PRUNE[@]}" "grep.*($KEYWORDS)" "${SEARCH_PATHS[@]}" 2>/dev/null || true)
EOF

# ---------------------------------------------------------------------------
# RULE 2: A SCRIPT MAY NOT EXPORT A BARE GREP'S EXIT STATUS AS ITS OWN.
#
# The measured escape, evidence/perf-061/ramp.sh:36 (#2800). See (3) in the
# header. This rule is about the SHAPE of the last command, not about the
# pattern -- no keyword list could ever have reached `grep -c "CB-006-OUT"` --
# but it keeps this file's method: the polarity is PROVED by running the real
# grep against a clean and a dirty input and printing both statuses. Nothing
# here reasons about anchors, banners or intent.
#
# WHY THE RULE IS UNCONDITIONAL rather than "only under a `must be 0` banner".
# grep's status answers "did the pattern occur". A script's status answers "did
# this script succeed". They are different propositions, so under EITHER
# polarity one of the two outcomes is mislabelled, and which one is a coin
# flip on how the banner was worded. A caller cannot tell. The remedy is one
# line, and it makes the intended polarity explicit:
#
#     n=$(... | grep -c "X"); [ "$n" -eq 0 ] || exit 1      # or a plain exit 0
#
# MEASURED: ZERO hits across all 209 .sh files on origin/main, and it fires on
# ramp.sh. A rule with no false positives on the whole tree costs nothing to
# carry.
#
# RESIDUAL, STATED: the scan is over *.sh. A script with no extension (a git
# hook, say) is out of THIS rule's universe though it is inside the keyword
# rule's, because the keyword rule greps directories and this one needs a file
# list. Widening it means deciding what a "script" is by shebang, which is a
# bigger change than the escape justifies.
INVERTED=0
SCANNED=0

sh_universe() {
    {
        git ls-files '*.sh' 2>/dev/null
        find . -name '*.sh' -type f \
             -not -path './.git/*' -not -path '*/target/*' \
             -not -path '*/target_disk/*' -not -path '*/node_modules/*' \
             2>/dev/null | sed 's|^\./||'
    } | LC_ALL=C sort -u
}

# The last command a script runs is the one whose status it returns. Comments
# and blank lines are not commands.
last_command_of() { # last_command_of <file>
    grep -vE '^[[:space:]]*(#|$)' "$1" 2>/dev/null | tail -1
}

while IFS= read -r f; do
    [ -n "$f" ] || continue
    [ -f "$f" ] || continue
    SCANNED=$((SCANNED + 1))
    lc=$(last_command_of "$f")
    [ -n "$lc" ] || continue

    # Forms where grep's status is CONSUMED rather than exported: a
    # conditional, a list, a substitution. Skipped, not judged.
    case "$lc" in
        *'&&'*|*'||'*|*'$('*|*'`'*) continue ;;
        *'&') continue ;;
    esac
    case "$lc" in
        if\ *|while\ *|until\ *|!\ *) continue ;;
    esac

    # The status of a pipeline is its LAST element's.
    tail_seg="${lc##*|}"
    tail_seg="${tail_seg#"${tail_seg%%[![:space:]]*}"}"
    case "$tail_seg" in
        grep\ *|grep) : ;;
        *) continue ;;
    esac

    # THE POLARITY PROBE. Run the real grep, in its own dialect, against an
    # input that cannot match and one that does. Two different statuses mean
    # the script's exit code is a function of whether the pattern OCCURRED.
    # No `| head -1`: an early-exiting reader hands sed a SIGPIPE and pipefail
    # reports 141 for a successful extraction. Capture whole, then take the
    # first line.
    recs=$(extract_patterns "$lc" || true)
    rec="${recs%%$'\n'*}"
    pat="${rec#*$'\t'}"
    flagstr="${rec%%$'\t'*}"
    pflags=()
    while IFS= read -r pf; do [ -n "$pf" ] && pflags+=("$pf"); done < <(probe_flags_for "$flagstr")

    rc_clean=0
    rc_dirty=0
    if [ -n "$rec" ] && [ -n "$pat" ]; then
        # `if`, not `cmd; rc=$?`: grep returns 1 BY DESIGN on the clean input,
        # and under `set -e` a bare invocation would kill the script before it
        # printed a single finding -- rc=1 with no evidence, which reads
        # exactly like a broken run.
        if grep -q "${pflags[@]}" -- "$pat" </dev/null >/dev/null 2>&1; then rc_clean=0; else rc_clean=$?; fi
        if grep -q "${pflags[@]}" -- "$pat" <<< "$pat" >/dev/null 2>&1; then rc_dirty=0; else rc_dirty=$?; fi
    else
        # Pattern not extractable (a variable, say). The SHAPE is still the
        # defect and grep's contract is not in doubt, so this is reported with
        # the statuses left unmeasured rather than skipped -- "could not
        # measure" is never "no finding" in this repository.
        rc_clean=1
        rc_dirty=0
    fi

    INVERTED=$((INVERTED + 1))
    lineno=$(grep -nvE '^[[:space:]]*(#|$)' "$f" | tail -1 | cut -d: -f1 || true)
    printf 'INVERTED-EXIT %s:%s\n' "$f" "$lineno"
    printf '              final command: %s\n' "$lc"
    printf '              the script EXPORTS this grep status as its own:\n'
    printf '                  input with NO match -> exit %s\n' "$rc_clean"
    printf '                  input WITH a match  -> exit %s\n' "$rc_dirty"
    printf '              fix: say what success means. `exit 0`, or\n'
    printf '                   n=$(... | grep -c "X"); [ "$n" -eq 0 ] || exit 1\n'
done <<EOF
$(sh_universe)
EOF

# Vacuity for rule 2. It finds nothing on a clean tree BY DESIGN, so "0
# findings" cannot be the health signal -- the file count is.
SH_MIN_EXPECTED="${SH_MIN_EXPECTED:-100}"
if [ "$SCANNED" -lt "$SH_MIN_EXPECTED" ]; then
    printf 'ERROR: rule 2 scanned only %s shell script(s) (expected >= %s).\n' \
        "$SCANNED" "$SH_MIN_EXPECTED" >&2
    printf 'The universe collapsed - this half of the check is now vacuous.\n' >&2
    exit 1
fi

if [ "$INVERTED" -gt 0 ]; then
    printf '\n%s script(s) export a bare grep status (of %s scanned).\n' "$INVERTED" "$SCANNED" >&2
    printf 'An inverted exit status is a gate that reports the OPPOSITE of the truth.\n' >&2
    exit 1
fi

if [ "$VIOLATIONS" -gt 0 ]; then
    printf '\n%s pass-grep(s) fail open (of %s checked).\n' "$VIOLATIONS" "$CHECKED" >&2
    printf 'A gate that cannot turn RED on a real regression is worth negative EV.\n' >&2
    exit 1
fi

# Fail-closed observability (the lesson of the dark beat lane): a checker that
# silently examines NOTHING reports the same "OK" as one that examined
# everything. If the extraction regex rots, or the search paths move, this
# must go red rather than congratulate itself on an empty set.
if [ "$CHECKED" -lt "$MIN_EXPECTED" ]; then
    printf 'ERROR: only %s pass-grep(s) found (expected >= %s).\n' \
        "$CHECKED" "$MIN_EXPECTED" >&2
    printf 'The extractor probably stopped matching - this check is now vacuous.\n' >&2
    exit 1
fi

printf 'OK: %s zero-count pass-grep(s) checked, all correctly reject a failing line;\n' "$CHECKED"
printf '    %s shell script(s) scanned, none exports a bare grep exit status.\n' "$SCANNED"
