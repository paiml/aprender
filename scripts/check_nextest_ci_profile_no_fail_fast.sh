#!/usr/bin/env bash
# check_nextest_ci_profile_no_fail_fast.sh -- nextest's [profile.ci] must declare
# fail-fast = false, explicitly (PMAT-3587, the nextest half).
#
# THE DEFECT. fail-fast is a verdict-discarding default: on the first failing test
# nextest cancels every test still queued, and the shard reports the one failure with
# no count of what it did not measure. Measured on a 6-test probe whose 2nd test fails:
#     fail-fast = true    Summary  3/6 tests run: 2 passed, 1 failed
#                         Cancelling due to test failure: 1 test still running
#     fail-fast = false   Summary  6 tests run: 5 passed, 1 failed
#     (key absent)        Summary  3/6 tests run  -- nextest's DEFAULT is fail-fast ON
# The tree paid for this once already: "nextest fail-fast hid every other dark failure:
# 7 rounds for 4 tests." A red run that still measures everything else it was going
# to measure is exactly the run whose data is most wanted.
#
# WHY THIS GUARD READS A TOML KEY AND DOES NOT RUN NEXTEST. The property IS the config
# value, read structurally (tomllib), under the profile CI actually uses -- not a
# substring anywhere in the file. A guard that invoked the build tool would match
# guard_tree.sh's CARGO_RE, be excluded from its --no-cargo run, and need its own
# ci.yml step; the behavioural proof above was run once and recorded in the PR that
# landed this. (This comment once said the tool's name with a space after it, and
# THAT was enough to match CARGO_RE: a substring test satisfied by the documentation
# of the thing it looks for, inside a guard written the same day that was named.)
# `--self-test` covers every polarity: false, true, absent, wrong profile, no profile,
# unparseable, missing file -- and only the first is green.
#
#   check_nextest_ci_profile_no_fail_fast.sh              judge .config/nextest.toml
#   check_nextest_ci_profile_no_fail_fast.sh --self-test  the case table (fixtures, no cargo)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
CONF="${NEXTEST_CONF_OVERRIDE:-$ROOT/.config/nextest.toml}"

# judge <toml> -> 0 when [profile.ci].fail-fast is literally false; 1 otherwise; 2 ENV.
judge() {
    local f=$1
    [ -r "$f" ] || { printf 'ENV   %s: not readable -- cannot judge, not a pass\n' "$f" >&2; return 2; }
    python3 - "$f" <<'PY'
import sys, tomllib
p = sys.argv[1]
try:
    with open(p, "rb") as fh:
        d = tomllib.load(fh)
except Exception as e:  # unparseable config is ENV, never a pass
    print(f"ENV   {p}: does not parse as TOML ({e}) -- cannot judge, not a pass", file=sys.stderr); sys.exit(2)
prof = d.get("profile", {}).get("ci")
if prof is None:
    print(f"FAIL  {p}: no [profile.ci] -- CI runs --profile ci, so the run would take nextest's default, which is fail-fast ON", file=sys.stderr); sys.exit(1)
v = prof.get("fail-fast")
if v is None:
    print(f"FAIL  {p}: [profile.ci] does not set fail-fast -- nextest's default is ON (measured: 3/6 tests run on a 6-test probe)", file=sys.stderr); sys.exit(1)
if v is not False:
    print(f"FAIL  {p}: [profile.ci].fail-fast = {v!r} -- a failing test cancels every test still queued and their verdicts are discarded", file=sys.stderr); sys.exit(1)
print(f"ok    {p}: [profile.ci].fail-fast = false -- a red run still measures everything else")
PY
}

case "${1:-}" in -h|--help) sed -n '2,24p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== nextest [profile.ci] fail-fast guard: case table ==="
    d=$(mktemp -d) || exit 2
    rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
    trap 'rmtree "${d:-}"' EXIT
    bad=0; n=0
    row() { # row WANT_RC LABEL TOML-BODY
        local want=$1 label=$2 body=$3 rc=0; n=$((n + 1))
        printf '%s' "$body" > "$d/c.toml"
        judge "$d/c.toml" > /dev/null 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  %s\n' "$n" "$rc" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s)  %s\n' "$n" "$rc" "$want" "$label" >&2; bad=1; fi
    }
    row 0 "fail-fast = false under [profile.ci] -> PASS"              $'[profile.ci]\nretries = 2\nfail-fast = false\n'
    row 1 "fail-fast = true under [profile.ci] -> RED"                $'[profile.ci]\nretries = 2\nfail-fast = true\n'
    row 1 "key ABSENT under [profile.ci] -> RED (nextest default is ON)" $'[profile.ci]\nretries = 2\n'
    row 1 "false under a DIFFERENT profile only -> RED"               $'[profile.default]\nfail-fast = false\n[profile.ci]\nretries = 2\n'
    row 1 "no [profile.ci] at all -> RED"                             $'[profile.default]\nfail-fast = false\n'
    row 1 "false in a COMMENT, true in the key -> RED (not a substring test)" $'[profile.ci]\n# fail-fast = false\nfail-fast = true\n'
    row 1 "the string \"false\" (a string, not a bool) -> RED"        $'[profile.ci]\nfail-fast = "false"\n'
    row 2 "unparseable TOML -> ENV rc=2, never a pass"                $'[profile.ci\nfail-fast = false\n'
    n=$((n + 1)); rc=0; judge "$d/absent.toml" > /dev/null 2>&1 || rc=$?
    [ "$rc" -eq 2 ] && printf 'ok    row %-2s rc=2  missing file -> ENV rc=2, never a pass\n' "$n" \
        || { printf 'FAIL  row %-2s rc=%s (wanted 2)  missing file -> ENV\n' "$n" "$rc" >&2; bad=1; }
    # and the real config, so a red tree cannot hide behind green fixtures
    n=$((n + 1)); rc=0; judge "$CONF" > /dev/null 2>&1 || rc=$?
    [ "$rc" -eq 0 ] && printf 'ok    row %-2s rc=0  the real %s is GREEN\n' "$n" "${CONF#"$ROOT/"}" \
        || { printf 'FAIL  row %-2s rc=%s  the real %s is RED\n' "$n" "$rc" "${CONF#"$ROOT/"}" >&2; bad=1; }
    [ "$bad" -eq 0 ] && { printf 'SELF-TEST PASSED: %s rows\n' "$n"; exit 0; }
    printf 'SELF-TEST FAILED\n' >&2; exit 1
fi

echo "=== nextest [profile.ci] must not discard verdicts on the first failure (check_nextest_ci_profile_no_fail_fast.sh) ==="
judge "$CONF"; rc=$?
[ "$rc" -eq 0 ] && echo "PASS" || echo "FAIL (rc=$rc)" >&2
exit "$rc"
