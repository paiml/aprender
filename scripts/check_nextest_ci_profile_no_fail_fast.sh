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
# value, read structurally, under the profile CI actually uses -- not a substring
# anywhere in the file. A guard that invoked the build tool would match guard_tree.sh's
# CARGO_RE, be excluded from its --no-cargo run, and need its own ci.yml step; the
# behavioural proof above was run once and recorded in the PR that landed this.
#
# ENV IS NEVER 1. The first cut imported tomllib (python 3.11+) at module level. On
# intel-clean-room-6 (#3626, job 106169486109) python3 is older, the import traceback
# exited 1, and the guard reported "the real .config/nextest.toml is RED" -- an
# environment death read as a code defect; rows 2-7 of the self-test even "passed",
# because they wanted rc=1 anyway. Now: tomllib, else tomli, else a purpose-built reader
# for the one key this guard judges; the shell side accepts a verdict only when the
# judge SAID one (a VERDICT line), and anything else -- a traceback, a dead
# interpreter -- is ENV rc=2 naming the runner. --self-test runs the whole case table
# twice, once per reader, plus the death rows. NO python3 AT ALL is different (#3697):
# the fleet is python-free for automation (infra#708), so that runner prints one
# UNMEASURED line and exits 0 (scripts/lib/python_fleet_state.sh), never a verdict.
#
#   check_nextest_ci_profile_no_fail_fast.sh              judge .config/nextest.toml
#   check_nextest_ci_profile_no_fail_fast.sh --self-test  the case table (fixtures, no cargo)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)" || exit 2
CONF="${NEXTEST_CONF_OVERRIDE:-$ROOT/.config/nextest.toml}"
# shellcheck source=lib/python_fleet_state.sh
. "$ROOT/scripts/lib/python_fleet_state.sh" || exit 2
# test seams (read per call, so a self-test row can set them): NEXTEST_GUARD_PYTHON, the
# interpreter (a dead one reproduces the intel shape); NEXTEST_GUARD_FORCE_FALLBACK=1, a
# real import failure of both TOML libraries (reproduces an old python on a new one).

# judge <toml> -> 0 when [profile.ci].fail-fast is literally false; 1 otherwise; 2 ENV;
# 3 UNMEASURED: this runner has no python3 at all (#3697). The fleet is python-free for
# automation (infra#708), so that is fleet state, never a verdict; a python3 that exists
# and dies is still ENV.
judge() {
    local f=$1 out rc=0 PYTHON="${NEXTEST_GUARD_PYTHON:-python3}" pyrc=0
    [ -r "$f" ] || { printf 'ENV   %s: not readable -- cannot judge, not a pass\n' "$f" >&2; return 2; }
    PY_FLEET_PYTHON="$PYTHON" py_fleet_state check_nextest_ci_profile_no_fail_fast || pyrc=$?
    [ "$pyrc" -eq 0 ] || return "$pyrc"
    out=$(NEXTEST_GUARD_RUNNER="${RUNNER_NAME:-unknown}" "$PYTHON" - "$f" 2>&1 <<'PY'
import os, re, sys
p = sys.argv[1]
runner = os.environ.get("NEXTEST_GUARD_RUNNER", "unknown")
if os.environ.get("NEXTEST_GUARD_FORCE_FALLBACK") == "1":
    # make `import tomllib` / `import tomli` raise for real, so the fallback is the
    # path an old interpreter takes and not a flag that skips around it
    sys.modules["tomllib"] = None
    sys.modules["tomli"] = None

def verdict(code, tag, msg):
    print("VERDICT %s %s: %s" % (tag, p, msg))
    sys.exit(code)

def load_with_lib(path):
    """(profile.ci table or None, reader name) via tomllib/tomli; None,None when neither imports."""
    for name in ("tomllib", "tomli"):
        try:
            mod = __import__(name)
        except ImportError:
            continue
        with open(path, "rb") as fh:
            d = mod.load(fh)
        ci = d.get("profile", {}).get("ci")
        if ci is not None and not isinstance(ci, dict):
            raise ValueError("profile.ci is a %s, not a table" % type(ci).__name__)
        return ci, name
    return None, None

_QUOTED = re.compile(r'"(?:[^"\\]|\\.)*"|\'[^\']*\'')
_HEADER = re.compile(r'^\[(\[)?\s*([A-Za-z0-9_.\-"\' ]+?)\s*\](\])?$')
_KEY = re.compile(r'^([A-Za-z0-9_\-]+|"[^"]+"|\'[^\']+\')\s*=\s*(.*)$')

def load_minimal(path):
    """Purpose-built reader for ONE question: under the table [profile.ci], what is the
    value of fail-fast? Table headers set the current table; every other line must be a
    single-line `key = value` that is balanced and opens no multi-line construct -- such
    lines are only SKIPPED unless they are the judged key. Anything it does not
    understand raises, and the caller reports ENV: it never guesses. Returns
    (None, ...) when no [profile.ci] table exists, else the dict of judged keys."""
    tables = {}
    cur = None
    with open(path, "r", encoding="utf-8") as fh:
        for n, raw in enumerate(fh, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            m = _HEADER.match(line)
            if m:
                aot = bool(m.group(1)) or bool(m.group(3))
                if bool(m.group(1)) != bool(m.group(3)):
                    raise ValueError("line %d: unbalanced table header" % n)
                name = m.group(2).replace('"', "").replace("'", "")
                if aot and name == "profile.ci":
                    raise ValueError("line %d: [[profile.ci]] is an array of tables, not the profile" % n)
                cur = name + ("[]" if aot else "")
                if not aot and cur in tables:
                    raise ValueError("line %d: duplicate table [%s]" % (n, name))
                tables.setdefault(cur, {})
                continue
            m = _KEY.match(line)
            if not m:
                raise ValueError("line %d: not a table header or a `key = value` line" % n)
            key, val = m.group(1).strip('"\''), m.group(2)
            if "." in key:
                raise ValueError("line %d: dotted key %r is outside this reader's scope" % (n, key))
            if val.startswith('"""') or val.startswith("'''"):
                raise ValueError("line %d: multi-line string is outside this reader's scope" % n)
            bare = _QUOTED.sub("", val)
            if bare.count("[") != bare.count("]") or bare.count("{") != bare.count("}"):
                raise ValueError("line %d: value continues past the line -- outside this reader's scope" % n)
            if bare.count('"') % 2 or bare.count("'") % 2:
                raise ValueError("line %d: unterminated string" % n)
            if cur is None:
                raise ValueError("line %d: top-level key %r is outside this reader's scope" % (n, key))
            if cur != "profile.ci":
                continue
            v = bare.split("#", 1)[0].strip() if not val.lstrip().startswith(('"', "'")) else val.strip()
            if v == "true":
                v = True
            elif v == "false":
                v = False
            elif re.match(r'^-?\d+$', v):
                v = int(v)
            elif len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            elif key == "fail-fast":
                raise ValueError("line %d: fail-fast = %r is not a value this reader can type" % (n, v))
            if key in tables[cur]:
                raise ValueError("line %d: duplicate key %r" % (n, key))
            tables[cur][key] = v
    return tables.get("profile.ci")

try:
    try:
        prof, how = load_with_lib(p)
        if how is None:
            prof, how = load_minimal(p), "purpose-built reader (no tomllib/tomli on this interpreter)"
    except Exception as e:
        verdict(2, "ENV", "cannot be parsed on runner=%s (%s: %s) -- cannot judge, not a pass and not a fail"
                % (runner, type(e).__name__, e))
    if prof is None:
        verdict(1, "FAIL", "no [profile.ci] -- CI runs --profile ci, so the run would take nextest's default, "
                "which is fail-fast ON (reader=%s)" % how)
    v = prof.get("fail-fast")
    if v is None:
        verdict(1, "FAIL", "[profile.ci] does not set fail-fast -- nextest's default is ON (measured: 3/6 tests "
                "run on a 6-test probe) (reader=%s)" % how)
    if v is not False:
        verdict(1, "FAIL", "[profile.ci].fail-fast = %r -- a failing test cancels every test still queued and "
                "their verdicts are discarded (reader=%s)" % (v, how))
    verdict(0, "ok", "[profile.ci].fail-fast = false -- a red run still measures everything else "
            "(reader=%s, runner=%s)" % (how, runner))
except SystemExit:
    raise
except BaseException as e:  # the judge itself died: say so, as ENV
    verdict(2, "ENV", "judge crashed on runner=%s (%s: %s) -- cannot judge, not a pass and not a fail"
            % (runner, type(e).__name__, e))
PY
    ) || rc=$?
    # A verdict is something the judge SAID. rc=1 with no VERDICT line is the interpreter
    # dying (a traceback before the first statement, a SyntaxError on an old python, no
    # python3 at all) -- the exact shape that read as "RED" on intel-clean-room-6.
    case "$out" in
        *"VERDICT ok "*|*"VERDICT FAIL "*|*"VERDICT ENV "*) ;;
        *) printf 'ENV   %s: the judge did not judge (interpreter=%s rc=%s runner=%s) -- not a pass and not a fail\n%s\n' \
               "$f" "$PYTHON" "$rc" "${RUNNER_NAME:-unknown}" "$out" >&2; return 2 ;;
    esac
    case "$rc" in
        0) printf '%s\n' "${out#VERDICT }" ;;
        1|2) printf '%s\n' "${out#VERDICT }" >&2 ;;
        *) printf 'ENV   %s: judge rc=%s is not a verdict (runner=%s)\n%s\n' "$f" "$rc" "${RUNNER_NAME:-unknown}" "$out" >&2; return 2 ;;
    esac
    return "$rc"
}

case "${1:-}" in -h|--help) sed -n '2,33p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;; esac

if [ "${1:-}" = "--self-test" ]; then
    echo "=== nextest [profile.ci] fail-fast guard: case table ==="
    d=$(mktemp -d) || exit 2
    rmtree() { case "${1:-}" in ''|/) return 0 ;; *) [ -d "$1" ] && rm -rf -- "$1" ;; esac; return 0; }
    trap 'rmtree "${d:-}"' EXIT
    bad=0; n=0
    row() { # row WANT_RC LABEL TOML-BODY   (reader named by $READER, set by table())
        local want=$1 label=$2 body=$3 rc=0; n=$((n + 1))
        printf '%s' "$body" > "$d/c.toml"
        judge "$d/c.toml" > /dev/null 2>&1 || rc=$?
        if [ "$rc" = "$want" ]; then printf 'ok    row %-2s rc=%s  [%s] %s\n' "$n" "$rc" "$READER" "$label"
        else printf 'FAIL  row %-2s rc=%s (wanted %s)  [%s] %s\n' "$n" "$rc" "$want" "$READER" "$label" >&2; bad=1; fi
    }
    table() { # the same table under BOTH readers; an old-python runner must judge the same property
        row 0 "fail-fast = false under [profile.ci] -> PASS"              $'[profile.ci]\nretries = 2\nfail-fast = false\n'
        row 1 "fail-fast = true under [profile.ci] -> RED"                $'[profile.ci]\nretries = 2\nfail-fast = true\n'
        row 1 "key ABSENT under [profile.ci] -> RED (nextest default is ON)" $'[profile.ci]\nretries = 2\n'
        row 1 "false under a DIFFERENT profile only -> RED"               $'[profile.default]\nfail-fast = false\n[profile.ci]\nretries = 2\n'
        row 1 "no [profile.ci] at all -> RED"                             $'[profile.default]\nfail-fast = false\n'
        row 1 "false in a COMMENT, true in the key -> RED (not a substring test)" $'[profile.ci]\n# fail-fast = false\nfail-fast = true\n'
        row 1 "the string \"false\" (a string, not a bool) -> RED"        $'[profile.ci]\nfail-fast = "false"\n'
        row 1 "false under [[profile.ci.overrides]] only -> RED (an override entry is not the profile)" \
            $'[profile.ci]\nretries = 2\n[[profile.ci.overrides]]\nfilter = "test(/x/)"\nfail-fast = false\n'
        row 0 "false after an inline table and before a sub-table -> PASS (the real file's shape)" \
            $'[profile.ci]\nretries = 2\nfail-fast = false\nslow-timeout = { period = "60s", terminate-after = 20 }\nstatus-level = "slow"\n[profile.ci.junit]\npath = "junit.xml"\n'
        row 2 "[[profile.ci]] (an array of tables, not the profile) -> ENV rc=2, never a pass" $'[[profile.ci]]\nfail-fast = false\n'
        row 2 "unparseable TOML -> ENV rc=2, never a pass"                $'[profile.ci\nfail-fast = false\n'
        # and the real config, so a red tree cannot hide behind green fixtures
        n=$((n + 1)); rc=0; judge "$CONF" > /dev/null 2>&1 || rc=$?
        [ "$rc" -eq 0 ] && printf 'ok    row %-2s rc=0  [%s] the real %s is GREEN\n' "$n" "$READER" "${CONF#"$ROOT/"}" \
            || { printf 'FAIL  row %-2s rc=%s  [%s] the real %s is RED\n' "$n" "$rc" "$READER" "${CONF#"$ROOT/"}" >&2; bad=1; }
    }
    # the readers are python: on a runner with none, the table is fleet state (#3697), and the
    # python-free rows below still run
    pyrc=0; PY_FLEET_PYTHON="${NEXTEST_GUARD_PYTHON:-python3}" py_fleet_state check_nextest_ci_profile_no_fail_fast 2> "$d/py.state" || pyrc=$?
    if [ "$pyrc" -eq 0 ]; then
        READER=library;  table
        READER=fallback; NEXTEST_GUARD_FORCE_FALLBACK=1 table
        # what the purpose-built reader must REFUSE rather than guess (ENV, never a pass)
        READER=fallback
        NEXTEST_GUARD_FORCE_FALLBACK=1 row 2 "a multi-line string hiding a [profile.ci] header -> ENV, never a guess (a line reader that skipped it would say PASS)" \
            $'[profile.default]\nnote = """title"\n[profile.ci]\nfail-fast = false\nx = """"\n'
        NEXTEST_GUARD_FORCE_FALLBACK=1 row 2 "a value continuing past its line -> ENV, never a guess" \
            $'[profile.ci]\nfail-fast = false\nslow-timeout = { period = "60s",\n  terminate-after = 20 }\n'
        NEXTEST_GUARD_FORCE_FALLBACK=1 row 2 "fail-fast = fals (not a value it can type) -> ENV, never a guess" \
            $'[profile.ci]\nfail-fast = fals\n'
    elif [ "$pyrc" -eq 3 ]; then
        cat "$d/py.state"
        printf 'UNMEASURED runner=%s reason=no-interpreter -- the reader case table (both readers) needs python3 and did not run here (#3697)\n' "${RUNNER_NAME:-unknown}"
    else
        cat "$d/py.state"; bad=1
    fi
    py_fleet_state_self_test "$d" || bad=1
    # the death rows: the shape that read as RED on intel-clean-room-6
    n=$((n + 1)); rc=0; printf '[profile.ci]\nfail-fast = false\n' > "$d/c.toml"
    NEXTEST_GUARD_PYTHON=/bin/false judge "$d/c.toml" > /dev/null 2>&1 || rc=$?
    [ "$rc" -eq 2 ] && printf 'ok    row %-2s rc=2  interpreter exits 1 with no verdict -> ENV rc=2, never 1\n' "$n" \
        || { printf 'FAIL  row %-2s rc=%s (wanted 2)  interpreter exits 1 with no verdict must be ENV, not RED\n' "$n" "$rc" >&2; bad=1; }
    # no interpreter at all is fleet state (#3697): UNMEASURED rc=3 naming it, never ENV and never
    # a verdict; the run then exits 0 with that line, and guard_tree surfaces it (#3651)
    n=$((n + 1)); rc=0
    out=$(NEXTEST_GUARD_PYTHON="$d/no-such-interpreter" judge "$d/c.toml" 2>&1) || rc=$?
    [ "$rc" -eq 3 ] && grep -q "^UNMEASURED runner=.* reason=no-interpreter interpreter=$d/no-such-interpreter " <<< "$out" \
        && ! grep -qE '^(ENV|VERDICT|ok )' <<< "$out" \
        && printf 'ok    row %-2s rc=3  no interpreter at all (rc=127) -> UNMEASURED naming it, never ENV (#3697)\n' "$n" \
        || { printf 'FAIL  row %-2s rc=%s (wanted 3 + UNMEASURED)  no interpreter must be fleet state: %s\n' "$n" "$rc" "$out" >&2; bad=1; }
    n=$((n + 1)); rc=0
    out=$(NEXTEST_GUARD_PYTHON="$d/no-such-interpreter" NEXTEST_CONF_OVERRIDE="$d/c.toml" bash "$0" 2>&1) || rc=$?
    [ "$rc" -eq 0 ] && grep -q '^UNMEASURED runner=.* reason=no-interpreter ' <<< "$out" && ! grep -q '^PASS' <<< "$out" \
        && printf 'ok    row %-2s rc=0  the RUN with no interpreter exits 0 with the UNMEASURED line and no PASS line (#3697)\n' "$n" \
        || { printf 'FAIL  row %-2s rc=%s  the run with no interpreter: %s\n' "$n" "$rc" "$out" >&2; bad=1; }
    n=$((n + 1)); rc=0; judge "$d/absent.toml" > /dev/null 2>&1 || rc=$?
    [ "$rc" -eq 2 ] && printf 'ok    row %-2s rc=2  missing file -> ENV rc=2, never a pass\n' "$n" \
        || { printf 'FAIL  row %-2s rc=%s (wanted 2)  missing file -> ENV\n' "$n" "$rc" >&2; bad=1; }
    [ "$bad" -eq 0 ] && { printf 'SELF-TEST PASSED: %s rows\n' "$n"; exit 0; }
    printf 'SELF-TEST FAILED\n' >&2; exit 1
fi

echo "=== nextest [profile.ci] must not discard verdicts on the first failure (check_nextest_ci_profile_no_fail_fast.sh) ==="
judge "$CONF"; rc=$?
[ "$rc" -eq 3 ] && exit 0   # UNMEASURED: the line is printed, fleet state, never a pass line (#3697)
[ "$rc" -eq 0 ] && echo "PASS" || echo "FAIL (rc=$rc)" >&2
exit "$rc"
