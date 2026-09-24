# python_fleet_state.sh — can THIS runner run the python a guard needs? (#3697)
#
# The fleet is python-free for automation (infra#708): the sovereign-ci image ships no python3, and
# intel's python3.10 has no TOML reader. A guard whose reader is python therefore meets three
# different runners, and only one of them is a verdict:
#
#   the interpreter runs and has the modules   -> the guard judges, as it always did
#   no interpreter, or a module not installed  -> FLEET STATE: one UNMEASURED line, the guard exits 0
#                                                 and guard_tree surfaces the line under PASS (#3651)
#   the interpreter exists and dies otherwise  -> ENV, rc 2, RED: a broken install is not fleet state
#
# That is the #3692 ruling (pathonly) and the #3695 one (parity denominator), in one place instead of
# a copy per guard. A guard that CAN drop python should do that instead, as #3694 did; this is for the
# readers that stay python (a YAML tree, a TOML table, a python library under test).
#
# SOURCED, so it is option-neutral: no `set`, and failure is a return status
# (check_sourced_libs_option_neutral.sh):
#
#   . "$ROOT/scripts/lib/python_fleet_state.sh" || exit 2
#   py_fleet_state <guard> [module ...]     0 ok · 3 UNMEASURED (line on stderr) · 2 ENV (line on stderr)
#   py_fleet_state_self_test <scratch-dir>  the case table; a consuming guard's --self-test runs it
#
# The interpreter is ${PY_FLEET_PYTHON:-python3}; PY_FLEET_PYTHON is the test seam.

py_fleet_state() {
    local guard=$1 py=${PY_FLEET_PYTHON:-python3} out rc=0 mod
    shift
    # A probe, not the guard's own program: it imports each module by name, and ONLY a
    # ModuleNotFoundError is "not installed". Any other failure propagates and is ENV.
    out=$("$py" -c '
import importlib, sys
for m in sys.argv[1:]:
    try:
        importlib.import_module(m)
    except ModuleNotFoundError:
        print("no-module " + m)
        sys.exit(3)
print("py-fleet-ok")
' "$@" 2>&1) || rc=$?
    case "$rc:$out" in
        0:py-fleet-ok) return 0 ;;
        126:*|127:*)
            printf 'UNMEASURED runner=%s reason=no-interpreter interpreter=%s guard=%s -- this runner has no %s, so the guard did not judge; fleet state, not a pass (#3697)\n' \
                "${RUNNER_NAME:-unknown}" "$py" "$guard" "$py" >&2
            return 3 ;;
        3:no-module\ *)
            mod=${out#no-module }
            printf 'UNMEASURED runner=%s reason=no-module module=%s interpreter=%s guard=%s -- %s cannot import %s, so the guard did not judge; fleet state, not a pass (#3697)\n' \
                "${RUNNER_NAME:-unknown}" "$mod" "$py" "$guard" "$py" "$mod" >&2
            return 3 ;;
        *)
            printf 'ENV   %s: the interpreter %s exists and failed (rc=%s): %s -- not fleet state, RED\n' \
                "$guard" "$py" "$rc" "${out:-<no output>}" >&2
            return 2 ;;
    esac
}

# The case table. Every classification row runs on a stub, so it is measured on a runner with no
# python at all; the two rows that need the real interpreter say UNMEASURED where there is none.
py_fleet_state_self_test() {
    local td=$1 bad=0 rc out
    [ -n "$td" ] && [ -d "$td" ] || { printf 'FAIL  py_fleet_state_self_test needs a scratch dir, got %s\n' "${td:-<empty>}"; return 1; }
    mkdir -p "$td/pyfs" || return 1
    printf '#!/bin/sh\necho "python3: command not found" >&2\nexit 127\n' > "$td/pyfs/py127"
    printf '#!/bin/sh\necho "no-module yaml"\nexit 3\n' > "$td/pyfs/pynomod"
    printf '#!/bin/sh\necho "Traceback (most recent call last): boom" >&2\nexit 1\n' > "$td/pyfs/pycrash"
    printf '#!/bin/sh\nexit 0\n' > "$td/pyfs/pysilent"
    chmod 755 "$td/pyfs"/py* || return 1

    # pyfs_row WANT_RC MUST_MATCH MUST_NOT_MATCH LABEL -- PY_FLEET_PYTHON set by the caller
    pyfs_row() {
        rc=0; out=$(py_fleet_state pyfs-probe yaml 2>&1) || rc=$?
        if [ "$rc" = "$1" ] && grep -qE -- "$2" <<<"$out" && ! grep -qE -- "$3" <<<"$out"; then
            printf 'ok    py_fleet_state rc=%s  %s\n' "$rc" "$4"
        else
            printf 'FAIL  py_fleet_state rc=%s (wanted %s, /%s/, not /%s/)  %s: %s\n' "$rc" "$1" "$2" "$3" "$4" "$out"
            bad=1
        fi
    }
    PY_FLEET_PYTHON="$td/pyfs/absent" pyfs_row 3 '^UNMEASURED runner=.* reason=no-interpreter interpreter=.*/absent guard=pyfs-probe' '^ENV' \
        "no interpreter on PATH -> UNMEASURED no-interpreter, never ENV"
    PY_FLEET_PYTHON="$td/pyfs/py127" pyfs_row 3 '^UNMEASURED runner=.* reason=no-interpreter' '^ENV' \
        "an interpreter that exits 127 (not found) -> UNMEASURED no-interpreter"
    PY_FLEET_PYTHON="$td/pyfs/pynomod" pyfs_row 3 '^UNMEASURED runner=.* reason=no-module module=yaml ' '^ENV' \
        "a module that is not installed -> UNMEASURED no-module naming it"
    PY_FLEET_PYTHON="$td/pyfs/pycrash" pyfs_row 2 '^ENV   pyfs-probe: .*rc=1' 'UNMEASURED' \
        "an interpreter that dies (rc 1) -> ENV rc 2, RED, never UNMEASURED"
    PY_FLEET_PYTHON="$td/pyfs/pysilent" pyfs_row 2 '^ENV   pyfs-probe: .*rc=0' 'UNMEASURED' \
        "an interpreter that answers nothing -> ENV rc 2, never ok"

    # The real interpreter: a module that cannot exist is no-module, and no modules at all is ok.
    rc=0; out=$(py_fleet_state pyfs-probe 2>&1) || rc=$?
    if [ "$rc" = 3 ]; then
        printf 'UNMEASURED runner=%s reason=no-interpreter -- the two real-interpreter rows of py_fleet_state_self_test did not run (#3697)\n' "${RUNNER_NAME:-unknown}"
    elif [ "$rc" = 0 ]; then
        printf 'ok    py_fleet_state rc=0  the real interpreter with no modules -> ok\n'
        rc=0; out=$(py_fleet_state pyfs-probe py_fleet_state_no_such_module_3697 2>&1) || rc=$?
        if [ "$rc" = 3 ] && grep -qE '^UNMEASURED runner=.* reason=no-module module=py_fleet_state_no_such_module_3697 ' <<<"$out"; then
            printf 'ok    py_fleet_state rc=3  the real interpreter, a module that does not exist -> UNMEASURED no-module\n'
        else
            printf 'FAIL  py_fleet_state rc=%s  the real interpreter, a module that does not exist: %s\n' "$rc" "$out"; bad=1
        fi
    else
        printf 'FAIL  py_fleet_state rc=%s  the real interpreter with no modules: %s\n' "$rc" "$out"; bad=1
    fi
    return "$bad"
}
