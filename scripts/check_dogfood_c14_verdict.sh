#!/usr/bin/env bash
# check_dogfood_c14_verdict.sh -- the dogfood's model-parity row (C14) reads a FAIL as FAIL,
# however much output follows it.
#
# THE DEFECT. scripts/dogfood.sh judged check_model_parity.sh's output with
# `printf '%s\n' "$C14_OUT"` piped into `grep -q -E '^FAIL |^override:'` under `set -o pipefail`. grep -q
# exits at its first match; printf, still writing, takes SIGPIPE, and the pipeline reports 141.
# A FAIL line followed by more than a pipe buffer of output read as NO FAIL, and the row fell
# through to REPORT ("C14 UNMEASURED"), which never fails the dogfood.
#
# WHAT THIS RUNS. The REAL `c14_verdict` function, extracted from scripts/dogfood.sh, over a
# case table; and the call site, which must route the row through it.
#
#   pass               rc 0, a PASS line                     -> PASS
#   fail               rc 1, a FAIL line                     -> FAIL
#   override           rc 1, an override: line               -> FAIL
#   rc0-without-pass   rc 0, no PASS line                    -> REPORT
#   unmeasured         rc 2, neither                         -> REPORT
#   pass-but-rc1       rc 1, a PASS line and a FAIL line     -> FAIL   (the exit code counts)
#   fail-before-2mib   rc 1, a FAIL line, then 2 MiB          -> FAIL   (the SIGPIPE row)
#   pass-before-2mib   rc 0, a PASS line, then 2 MiB          -> PASS
#
# Usage: bash scripts/check_dogfood_c14_verdict.sh [--self-test]
#       --self-test: 0 when the shipped function is GREEN and each planted mutation turns it RED.
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SUBJECT="$ROOT/scripts/dogfood.sh"
SELF_TEST=0
while [ $# -gt 0 ]; do
    case "$1" in
        --self-test) SELF_TEST=1; shift ;;
        -h|--help) awk 'NR == 1 { next } !/^#/ { exit } { sub(/^# ?/, ""); print }' "$0"; exit 0 ;;
        *) printf 'usage: %s [--self-test]\n' "$0" >&2; exit 2 ;;
    esac
done
[ -f "$SUBJECT" ] || { printf 'ENV   no %s\n' "$SUBJECT" >&2; exit 2; }

# judge FILE -> prints one line per failed row; 0 all rows hold, 1 a row failed, 2 no function to judge
judge() {
    local fn got want rc out bad=0
    fn="$(sed -n '/^c14_verdict() {$/,/^}$/p' "$1")"
    [ -n "$fn" ] || { printf 'ENV   no c14_verdict() in %s\n' "$1"; return 2; }
    eval "$fn"
    local big; printf -v big '%2097152s' ''
    while IFS='|' read -r name rc out want; do
        out="${out//\\n/$'\n'}"; out="${out//<2MiB>/$big}"
        C14_VERDICT=unset; c14_verdict "$rc" "$out"; got="$C14_VERDICT"
        [ "$got" = "$want" ] || { printf 'RED   %-18s c14_verdict gave %s, wanted %s\n' "$name" "$got" "$want"; bad=1; }
    done <<'CASES'
pass|0|PASS m1 parity 1.0|PASS
fail|1|FAIL m1 parity 0.2|FAIL
override|1|override: m1 skipped by hand|FAIL
rc0-without-pass|0|measured nothing|REPORT
unmeasured|2|no CUDA device|REPORT
pass-but-rc1|1|PASS m1 parity 1.0\nFAIL m2 parity 0.2|FAIL
fail-before-2mib|1|FAIL m1 parity 0.2\n<2MiB>|FAIL
pass-before-2mib|0|PASS m1 parity 1.0\n<2MiB>|PASS
CASES
    # The row must go through the function: a call site that judges $C14_OUT itself is untested.
    grep -qxF '    c14_verdict "$C14_RC" "$C14_OUT"' "$1" && grep -qxF '    case "$C14_VERDICT" in' "$1" \
        || { printf 'RED   call-site          model-parity is not judged by c14_verdict\n'; bad=1; }
    return "$bad"
}

out="$(judge "$SUBJECT")"; rc=$?
if [ "$SELF_TEST" = 0 ]; then
    [ "$rc" = 0 ] || { printf '%s\n' "$out" >&2; exit "$rc"; }
    printf 'PASS  dogfood C14: c14_verdict reads 8 cases right, and the row goes through it\n'
    exit 0
fi

fails=0
[ "$rc" = 0 ] && echo "self-test: the shipped dogfood.sh -- GREEN (expected)" \
    || { printf 'self-test: the shipped dogfood.sh is RED:\n%s\n' "$out"; fails=1; }
TMP="$(mktemp -d "${TMPDIR:-/tmp}/c14-verdict.XXXXXX")" || exit 2
trap 'rm -rf -- "${TMP:?}"' EXIT
# mutant NAME SED-EXPR -- the expression must change dogfood.sh, and the result must go RED
mutant() {
    sed "$2" "$SUBJECT" > "$TMP/$1.sh"
    if cmp -s "$SUBJECT" "$TMP/$1.sh"; then echo "self-test: mutant $1 -- NOT PLANTED"; fails=1; return; fi
    if judge "$TMP/$1.sh" > /dev/null; then echo "self-test: mutant $1 -- GREEN (the table does not see it)"; fails=1
    else echo "self-test: mutant $1 -- RED (expected)"; fi
}
# The pipe is spelt $P so scripts/check_no_pipe_into_grep_q.sh does not count a mutant as a site.
P='|'
mutant fail-piped   "s/  elif grep -qE '^FAIL |^override:' <<< \"\$2\"; then C14_VERDICT=FAIL/  elif printf '%s\\\\n' \"\$2\" $P grep -qE '^FAIL |^override:'; then C14_VERDICT=FAIL/"
mutant pass-piped   "s/  if \[ \"\$1\" -eq 0 \] \&\& grep -q '^PASS ' <<< \"\$2\"; then/  if [ \"\$1\" -eq 0 ] \&\& printf '%s\\\\n' \"\$2\" $P grep -q '^PASS '; then/"
mutant no-override  "s/'^FAIL |^override:' <<< /'^FAIL ' <<< /"
mutant rc-ignored   "s/  if \[ \"\$1\" -eq 0 \] \&\& grep -q/  if grep -q/"
mutant bypassed     's/    case "\$C14_VERDICT" in/    case PASS in/'
mutant no-call     's/^    c14_verdict "$C14_RC" "$C14_OUT"$/    :/'
[ "$fails" = 0 ] || exit 1
echo "self-test: PASS"
