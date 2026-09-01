#!/usr/bin/env bash
# pr_review_shadow_record.sh - S13.11 rung 1: run the quorum arm in shadow mode
# against ONE live pull request and record the verdict as a sample.
#
# PR-REVIEW-SKILL-002 v2 S13.11 rung 1: "CI runs `--explain` on every PR and records
# PERMIT/REFUSE [Qn]. Merges nothing." S8 wants 30 samples before any threshold is set
# on `autonomy_refusal_rate` or `degraded_share`; this is the thing that produces them.
#
# WHY THIS IS A SCRIPT AND NOT SIX LINES OF YAML
# ----------------------------------------------
# Arm 4 was an inline `run:` block whose first branch was `exit 0`, the key it tested
# for was never shipped, and so it passed every run it ever had - a gate that could not
# fail, inside the job written to prevent gates that cannot fail, invisible because
# inline YAML is reachable from no test. That is written down in
# check_pr_review_arm4.sh's own header. A recorder has the same exposure in a quieter
# form: one that silently records nothing still reports success, and rung 2 then reads
# "0 samples" as "the fleet never refuses" instead of "the recorder never ran". So this
# lives in a script, with `--self-test`, like its neighbours.
#
# RC IS NOT A VERDICT. THIS IS THE ARM'S OWN LESSON, APPLIED TO THE ARM.
# S3.E.4 records that `agy` auto-denied a permission in headless mode and returned
# rc 0, status SUCCESS, an empty response and no structured output - so a consultation
# recorded from an exit code is "a review that never happened counted as one". The same
# is true one level up: an arm script that exits 0 while printing nothing must NOT be
# recorded as PERMIT. Every branch below requires the MATCHING OUTPUT LINE as well as
# the code, and disagreement between them is a defect, never a sample.
#
# THE REFUSAL CLASS IS READ THROUGH A CLOSED VOCABULARY.
# S13's forgery post-mortem is unambiguous: every clause that survived the nine forged
# receipts was a whitelist, every clause that fell was a blacklist, and "the difference
# is not the field, not the author and not the amount of care: it is the DIRECTION of
# the test." The class here is producer-supplied (the arm script prints it), so it is
# matched against the Q1..Q10 vocabulary and an unrecognised class is a DEFECT. A
# recorder that accepted `[Q99]`, `[]` or `[BLOCKING]` would silently widen the very
# vocabulary rung 2 is going to compute a rate over.
#
# EXIT
#   0  a sample was recorded - PERMIT or REFUSE, both are samples
#   1  NO sample could be recorded: the box could not answer (arm rc 2), the arm
#      contradicted itself (rc without its line), or the class is not in the
#      vocabulary. Loud, because a silent zero is indistinguishable from a quiet fleet.
#
# THIS SCRIPT CANNOT MERGE ANYTHING. It passes --explain, which S13 documents as the
# verb that "NEVER calls `gh pr merge`", and the CI job that invokes it holds a
# read-only token so the capability is absent as well as unused.
#
# USAGE
#   pr_review_shadow_record.sh --pr <N> [--head <sha>] [--out <dir>]
#   pr_review_shadow_record.sh --self-test
#
# ENVIRONMENT
#   PR_REVIEW_ARM   arm script to invoke (default: scripts/pr_review_quorum_arm.sh)
#   No value of it turns a check off: an arm pointed at a silent stub fails row 4, and
#   one pointed at a lying stub fails rows 6 and 9. Both polarities are --self-test rows.

set -euo pipefail

PROG=${0##*/}
HERE=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$HERE/.." && pwd)

ARM=${PR_REVIEW_ARM:-$REPO_ROOT/scripts/pr_review_quorum_arm.sh}

# S13's own class list. A LIST AND NOT A PATTERN, for the reason MECHANISM_PATHS gives:
# this repository's guard regexes have been wrong six times and a case table caught
# every one.
VALID_CLASSES=' Q1 Q2 Q3 Q4 Q5 Q6 Q7 Q8 Q9 Q10 '

# ---------------------------------------------------------------------------
# safe_rm_scratch <path> <required-substring> - a recursive delete, guarded.
# SEC011, and the same shape check_pr_review_arm4.sh uses: the two ways
# `rm -rf -- "$x"` goes wrong are an EMPTY x and an x that is not ours. Both are
# checked before the expansion, and a path failing either is left alone rather
# than deleted "carefully".
# ---------------------------------------------------------------------------
safe_rm_scratch() {
    local victim=${1:-} must=${2:-}
    [ -n "$victim" ] || return 0
    [ -n "$must" ]   || return 0
    [ "$victim" != "/" ] || return 0
    case "$victim" in
      *"$must"*) if [ -n "$victim" ] && [ "$victim" != "/" ]; then rm -rf -- "$victim"; fi ;;
      *) return 0 ;;
    esac
}

ST_ROOT=''
cleanup_self_test() { safe_rm_scratch "$ST_ROOT" 'shadow-selftest.'; }

die_env()  { echo "$PROG: ENV - $*" >&2; exit 1; }
defect()   { echo "$PROG: DEFECT - $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# record_one <pr> <head> <outdir> - invoke the arm, classify, emit the sample.
# ---------------------------------------------------------------------------
record_one() {
    local pr=$1 head=$2 outdir=$3
    local log rc=0 verdict class line

    [ -f "$ARM" ] || die_env "arm script not found: $ARM"

    log=$(mktemp) || die_env "cannot create a temp file"

    # NEVER read $? through a pipe (Verification Discipline #1): the arm's status is
    # captured directly off the command, and the log is a file, not a pipeline stage.
    set +e
    bash "$ARM" --pr "$pr" --explain >"$log" 2>&1
    rc=$?
    set -e

    case "$rc" in
      0)
        # rc 0 is PERMIT only if the PERMIT line is there too.
        line=$(grep -m1 '^PERMIT ' "$log" || true)
        if [ -z "$line" ]; then
            cat "$log" >&2; rm -f "$log"
            defect "arm exited 0 without a PERMIT line; an exit code is not a verdict (S3.E.4)"
        fi
        verdict=PERMIT
        class='-'
        ;;
      1)
        line=$(grep -m1 '^REFUSE ' "$log" || true)
        if [ -z "$line" ]; then
            cat "$log" >&2; rm -f "$log"
            defect "arm exited 1 without a REFUSE line; a refusal with no name is undiagnosable"
        fi
        # `REFUSE  pr=<n>  [Qk]`
        class=$(printf '%s\n' "$line" | sed -n 's/.*\[\([^]]*\)\].*/\1/p')
        if [ -z "$class" ]; then
            cat "$log" >&2; rm -f "$log"
            defect "REFUSE line carries no class: $line"
        fi
        case "$VALID_CLASSES" in
          *" $class "*) : ;;
          *) cat "$log" >&2; rm -f "$log"
             defect "refusal class '$class' is not one of S13's Q1..Q10; the class is producer-supplied and is read through a closed vocabulary, never a blacklist" ;;
        esac
        verdict=REFUSE
        ;;
      2)
        cat "$log" >&2; rm -f "$log"
        die_env "the arm script could not answer (rc 2). A recorder that swallows this reports zero samples, and zero samples reads as a quiet fleet rather than a broken one"
        ;;
      *)
        cat "$log" >&2; rm -f "$log"
        defect "arm exited $rc, which is not one of S13's documented 0/1/2"
        ;;
    esac

    # The one machine-readable line. Stable field order so a harvester can grep it out
    # of a job log years from now without a parser.
    printf 'S13-SHADOW pr=%s head=%s verdict=%s class=%s arm_rc=%s\n' \
        "$pr" "$head" "$verdict" "$class" "$rc"

    if [ -n "$outdir" ]; then
        mkdir -p -- "$outdir"
        jq -n --arg pr "$pr" --arg head "$head" --arg v "$verdict" \
              --arg c "$class" --argjson rc "$rc" \
              --arg detail "$(cat "$log")" \
              '{pr: $pr, head_sha: $head, verdict: $v, class: $c, arm_rc: $rc,
                rung: 1, spec: "PR-REVIEW-SKILL-002-v2 S13.11", detail: $detail}' \
          > "$outdir/shadow-$pr.json"
    fi

    rm -f "$log"
    return 0
}

# ---------------------------------------------------------------------------
# --self-test - both polarities of every branch above.
#
# Each row installs a STUB arm script with a known rc and a known stdout, so the
# recorder's classification is exercised without a repository, a receipt or a network.
# The rows that must FAIL are the point: a recorder whose RED has never been seen is a
# recorder whose GREEN is a count of invocations.
# ---------------------------------------------------------------------------
self_test() {
    local td fails=0 pass_n=0
    td=$(mktemp -d -t shadow-selftest.XXXXXX) || die_env "cannot create a temp dir"
    ST_ROOT=$td
    trap cleanup_self_test EXIT

    mk_stub() { # <name> <rc> <stdout>
        local f="$td/$1.sh"
        {
            echo '#!/usr/bin/env bash'
            printf 'cat <<'"'"'OUT'"'"'\n%s\nOUT\n' "$3"
            echo "exit $2"
        } > "$f"
        chmod +x "$f"
        printf '%s' "$f"
    }

    row() { # <desc> <expect PASS|FAIL> <stub> [<expect-verdict> <expect-class>]
        local desc=$1 expect=$2 stub=$3 wv=${4:-} wc=${5:-}
        local out rc=0 got_v got_c
        set +e
        out=$(PR_REVIEW_ARM="$stub" bash "$HERE/$PROG" --pr 1 --head deadbeef 2>&1)
        rc=$?
        set -e
        local verdict_ok=1
        if [ "$expect" = PASS ] && [ "$rc" -eq 0 ] && [ -n "$wv" ]; then
            got_v=$(printf '%s\n' "$out" | sed -n 's/.*verdict=\([A-Z-]*\).*/\1/p' | head -1)
            got_c=$(printf '%s\n' "$out" | sed -n 's/.*class=\([A-Za-z0-9-]*\).*/\1/p' | head -1)
            if [ "$got_v" != "$wv" ] || [ "$got_c" != "$wc" ]; then
                verdict_ok=0
            fi
        fi
        if { [ "$expect" = PASS ] && [ "$rc" -eq 0 ] && [ "$verdict_ok" -eq 1 ]; } \
        || { [ "$expect" = FAIL ] && [ "$rc" -ne 0 ]; }; then
            printf 'ok   %s\n' "$desc"
            pass_n=$((pass_n + 1))
        else
            printf 'FAIL %s (expected %s, rc=%s, verdict=%s/%s want %s/%s)\n' \
                "$desc" "$expect" "$rc" "${got_v:-}" "${got_c:-}" "$wv" "$wc"
            printf '     output: %s\n' "$out"
            fails=$((fails + 1))
        fi
    }

    echo "== pr_review_shadow_record.sh --self-test =="

    row 'a PERMIT with its line is a sample' PASS \
        "$(mk_stub permit 0 'PERMIT  pr=1  receipt=evidence/pr-review/1/abc')" PERMIT '-'

    row 'a REFUSE with a valid class is a sample' PASS \
        "$(mk_stub refuse 1 'REFUSE  pr=1  [Q1]
    no such receipt directory')" REFUSE Q1

    row 'every S13 class Q1..Q10 is accepted' PASS \
        "$(mk_stub refuse10 1 'REFUSE  pr=1  [Q10]
    a required check is not green')" REFUSE Q10

    # --- the RED rows. Each is a way the recorder could report a sample it does not have.
    row 'rc 2 (box cannot answer) is LOUD, never a silent zero' FAIL \
        "$(mk_stub env 2 'pr_review_quorum_arm.sh: ENV - cannot run: jq not on PATH.')"

    row 'rc 0 with NO PERMIT line is a defect, not a PERMIT' FAIL \
        "$(mk_stub silent0 0 '')"

    row 'rc 1 with NO REFUSE line is a defect, not a REFUSE' FAIL \
        "$(mk_stub silent1 1 '')"

    row 'a REFUSE with an EMPTY class is a defect' FAIL \
        "$(mk_stub emptyclass 1 'REFUSE  pr=1  []
    something')"

    row 'a class outside Q1..Q10 is a defect (closed vocabulary)' FAIL \
        "$(mk_stub badclass 1 'REFUSE  pr=1  [Q99]
    invented')"

    row 'a non-Q spelling is a defect, not admitted by a blacklist' FAIL \
        "$(mk_stub wordclass 1 'REFUSE  pr=1  [BLOCKING]
    invented')"

    row 'an undocumented rc is a defect, never a sample' FAIL \
        "$(mk_stub weird 7 'PERMIT  pr=1  receipt=x')"

    row 'an absent arm script is ENV, not a pass' FAIL "$td/no-such-arm.sh"

    # A stub that PRINTS permit but EXITS 1 must be read by its rc, and the missing
    # REFUSE line then makes it a defect - the two halves must agree.
    row 'PERMIT text with a refusing rc is a defect (text and rc must agree)' FAIL \
        "$(mk_stub mismatch 1 'PERMIT  pr=1  receipt=x')"

    printf '\n%s passed, %s failed\n' "$pass_n" "$fails"
    [ "$fails" -eq 0 ] || { echo 'SELF-TEST FAILED'; exit 1; }
    exit 0
}

# ---------------------------------------------------------------------------
PR=''; HEAD=''; OUT=''
while [ $# -gt 0 ]; do
  case "$1" in
    --self-test) self_test ;;
    --pr)   PR=${2:-};   shift 2 ;;
    --head) HEAD=${2:-}; shift 2 ;;
    --out)  OUT=${2:-};  shift 2 ;;
    -h|--help) sed -n '2,60p' "$0"; exit 0 ;;
    *) die_env "unknown argument: $1" ;;
  esac
done

[ -n "$PR" ] || die_env "--pr is required"
command -v jq >/dev/null 2>&1 || die_env "jq is not on PATH"

if [ -z "$HEAD" ]; then
  HEAD=${PR_HEAD_SHA:-unknown}
fi

record_one "$PR" "$HEAD" "$OUT"
